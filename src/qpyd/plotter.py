import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from .datatypes import StateProperties


class Plotter:
    """
    Class to handle visualization and saving of simulation results.
    
    Uses a data-driven approach to map physical observables to their respective 
    visual configurations (colormaps, labels). Provides individual public methods 
    for plotting specific physical observables like stability diagrams (heatmaps), 
    1D slices, and spin configurations.
    """

    # Central registry mapping the keys from `get_results()` to plotting metadata.
    _OBSERVABLES = {
        'I':   {'cmap': 'seismic', 'label': '$I$'},
        'G':   {'cmap': 'magma',   'label': '$\\partial I / \\partial V_b$'},
        'N_avg': {'cmap': 'viridis', 'label': '$\\langle N \\rangle$'},
        'P':   {'cmap': 'plasma',  'label': '$P_{{{state_idx}}}$'}
    }
    
    def __init__(self):
        """Initializes the Plotter with default matplotlib settings."""
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        """
        Configures global matplotlib parameters for figures.

        Returns
        -------
        None
        """
        plt.rcParams['svg.fonttype'] = 'none'
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['axes.labelsize'] = 18
        matplotlib.rcParams['xtick.labelsize'] = 18
        matplotlib.rcParams['ytick.labelsize'] = 18
        matplotlib.rcParams['axes.titlesize'] = 18

    def _add_labels(
        self, 
        im: AxesImage, 
        cbar_title: str, 
        title: str | None = None, 
        axes_units: list[str] | None = None
    ) -> None:
        """
        Helper to add titles, labels, and colorbars to 2D plots.

        Parameters
        ----------
        im : AxesImage
            The image object returned by `plt.imshow`.
        cbar_title : str
            Title string for the colorbar axis.
        title : str, optional
            Main plot title. Default is None.
        axes_units : list of str, optional
            Units for the [X, Y] axes. Default is None.
        """
        x_label = "$V_g$"
        y_label = "$V_b$"
        
        if axes_units is not None:
            if axes_units[0]: x_label += f" ({axes_units[0]})"
            if axes_units[1]: y_label += f" ({axes_units[1]})"
            
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        cbar = plt.colorbar(im)
        cbar.ax.set_title(cbar_title, fontsize=18, pad=10)
        
        if title is not None:
            plt.title(title)
            
        plt.tight_layout()
    
    def _get_unique_filename(self, filename: str) -> str:
        """
        Helper to generate a unique filename to avoid overwriting existing files.

        Parameters
        ----------
        filename : str
            Requested file name including extension.

        Returns
        -------
        str
            A safe file name with an appended integer if required.
        """
        if not os.path.exists(filename):
            return filename
            
        base, ext = os.path.splitext(filename)
        counter = 1
        new_filename = f"{base}_{counter}{ext}"
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{base}_{counter}{ext}"
            
        return new_filename

    def plot_heatmap(
        self, 
        data: np.ndarray, 
        extent: tuple[float, float, float, float], 
        cbar_title: str, 
        title: str, 
        axes_units: list[str] | None = None, 
        cmap: str = 'viridis', 
        v_range: tuple[float, float] | None = None, 
        save_format: str | None = None, 
        save_name: str | None = None
    ) -> None:
        """
        Generic method to plot any 2D data as a colored heatmap.

        Parameters
        ----------
        data : np.ndarray
            2D numeric array to plot.
        extent : tuple of float
            Plot boundaries formatted as `(min_Vg, max_Vg, min_Vb, max_Vb)`.
        cbar_title : str
            Title for the colorbar.
        title : str
            Main title of the plot.
        axes_units : list of str, optional
            Units for the X and Y axes. Default is None.
        cmap : str, optional
            Colormap name mapping to a valid Matplotlib colormap. Default is 'viridis'.
        v_range : tuple of float, optional
            Limits `(vmin, vmax)` for the color scale. If None, it autoscales 
            to the min/max of the data array.
        save_format : str, optional
            File extension to save the figure (e.g., 'png', 'svg', 'pdf'). 
            If None, the figure is not saved.
        save_name : str, optional
            Filename base without extension. If None and `save_format` is provided, 
            a name is automatically generated from the title and a timestamp.
        """
        if v_range is not None:
            vmin, vmax = v_range
        else:
            vmin = np.min(data)
            vmax = np.max(data)
            if vmin == vmax:
                vmax += 1e-12

        fig = plt.figure(figsize=(8, 6))
        ax = plt.imshow(data.T, extent=extent, vmin=vmin, vmax=vmax,
                        origin='lower', aspect='auto', cmap=cmap)
                        
        self._add_labels(ax, cbar_title, title, axes_units)
        plt.show()

        if save_format is not None:
            if save_name is None:
                clean_title = title.replace(" ", "_").replace("/", "-").replace("$", "").replace("\\", "")
                save_name = f"{clean_title}_{int(time.time())}"
            filename = f"{save_name}.{save_format}"
            unique_filename = self._get_unique_filename(filename)
            plt.savefig(unique_filename, format=save_format)
            print(f"Saved figure to {unique_filename}")

    def plot_map(
        self, 
        results: dict[str, np.ndarray], 
        quantity: str, 
        extent: tuple[float, float, float, float], 
        state_idx: int = 0, 
        v_range: tuple[float, float] | None = None, 
        title: str | None = None, 
        axes_units: list[str] | None = None, 
        quantity_units: str | None = None,
        custom_cmap: str | None = None,
        save_format: str | None = None, 
        save_name: str | None = None
    ) -> None:
        """
        Unified method to plot a stability diagram for any known observable.

        Parameters
        ----------
        results : dict of str to np.ndarray
            The dictionary returned by `sim.get_results()`.
        quantity : str
            The physical quantity to plot: 'I' (Current), 'G' (Conductance), 
            'N_avg' (Occupation), or 'P' (Probability).
        extent : tuple of float
            Plot boundaries formatted as `(min_Vg, max_Vg, min_Vb, max_Vb)`.
        state_idx : int, optional
            If quantity is 'P', this selects which state probability to plot. Default is 0.
        v_range : tuple of float, optional
            Custom limits `(vmin, vmax)` for the color axis. Default is None.
        title : str, optional
            Main title of the plot. Defaults to a generic title based on the quantity.
        axes_units : list of str, optional
            Units for the X and Y axes. Default is None.
        quantity_units : str, optional
            Units for the plotted quantity. Default is None.
        custom_cmap : str, optional
            Override the default colormap for the specified quantity.
        save_format : str, optional
            File format to save. If None, does not save.
        save_name : str, optional
            Filename base.
        """
        if quantity not in self._OBSERVABLES:
            raise ValueError(f"Unknown quantity '{quantity}'. Available: {list(self._OBSERVABLES.keys())}")
            
        meta = self._OBSERVABLES[quantity]
        cmap = custom_cmap or meta['cmap']
        data = results[quantity]
        
        if quantity == 'P':
            data = data[:, :, state_idx]
            label = meta['label'].format(state_idx=state_idx + 1)
        else:
            label = meta['label']

        if quantity_units:
            label += f" ({quantity_units})"
            
        self.plot_heatmap(data, extent, label, title, axes_units, cmap, v_range, save_format, save_name)

    def plot_slice(
        self, 
        results: dict[str, np.ndarray], 
        quantity: str | np.ndarray, 
        cut_type: str, 
        fixed_value: float, 
        state_idx: int = 0, 
        axes_units: list[str] | None = None, 
        quantity_units: str | None = None,
        color: str = 'black', 
        save_name: str | None = None, 
        save_format: str | None = None, 
        ax: Axes | None = None, 
        custom_ylabel: str | None = None, 
        label: str | None = None
    ) -> Axes:
        """
        Plots a 1D slice (vertical or horizontal) of the simulation results.

        Parameters
        ----------
        results : dict of str to np.ndarray
            The dictionary returned by `sim.get_results()`.
        quantity : str or np.ndarray
            The physical quantity to plot: 'I', 'G', 'N_avg', 'P', OR a custom 2D numpy array.
        cut_type : str
            'horizontal' (Sweep Vg, fixed Vb) or 'vertical' (Sweep Vb, fixed Vg).
        fixed_value : float
            The voltage value at which to extract the 1D slice.
        state_idx : int, optional
            If quantity is 'P', this selects which state probability to plot. Default is 0.
        axes_units : list of str, optional
            Units for the voltage axes `[Vg_unit, Vb_unit]`. Used to label the x-axis and the title. Default is None.
        quantity_units : str, optional
            Units for the plotted quantity (y-axis). Default is None.
        color : str, optional
            Color of the plot line. Default is 'black'.
        save_name : str, optional
            Filename base to save. Default is None.
        save_format : str, optional
            File format to save. Default is None.
        ax : Axes or None, optional
            An existing Matplotlib axes object to plot onto. If None, a new figure is created.
        custom_ylabel : str, optional
            Override the default y-axis label entirely. Default is None.
        label : str, optional
            Legend label for the plotted line. Default is None.
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the plot.
        """
        Vgs = results['Vgs']
        Vbs = results['Vbs']
        
        if isinstance(quantity, str):
            if quantity not in self._OBSERVABLES:
                raise ValueError(f"Unknown quantity string. Choose from: {list(self._OBSERVABLES.keys())}")
                
            data = results[quantity]
            if quantity == 'P':
                data = data[:, :, state_idx]
                ylabel = self._OBSERVABLES['P']['label'].format(state_idx=state_idx + 1)
            else:
                ylabel = self._OBSERVABLES[quantity]['label']
        elif isinstance(quantity, np.ndarray):
            data = quantity
            ylabel = "Custom Data"
        else:
            raise TypeError("quantity must be a string or a 2D numpy array.")

        # Add quantity units to the y-axis label if provided
        if quantity_units:
            ylabel += f" ({quantity_units})"

        # custom_ylabel overrides everything if provided
        if custom_ylabel: 
            ylabel = custom_ylabel

        if ax is None: 
            fig, current_ax = plt.subplots(figsize=(8, 6))
        else: 
            current_ax, fig = ax, ax.figure

        if cut_type == 'horizontal':
            idx = (np.abs(Vbs - fixed_value)).argmin()
            actual_val = Vbs[idx]
            x_axis = Vgs
            y_axis = data[:, idx] 
            
            xlabel = "$V_g$" + (f" ({axes_units[0]})" if axes_units and axes_units[0] else "")
            title = f"$V_b \\approx {actual_val:.3f}$" + (f" {axes_units[1]}" if axes_units and axes_units[1] else "")

        elif cut_type == 'vertical':
            idx = (np.abs(Vgs - fixed_value)).argmin()
            actual_val = Vgs[idx]
            x_axis = Vbs
            y_axis = data[idx, :]
            
            if data.shape[1] == len(Vbs) - 1: 
                x_axis = x_axis[:-1]
                
            xlabel = "$V_b$" + (f" ({axes_units[1]})" if axes_units and axes_units[1] else "")
            title = f"$V_g \\approx {actual_val:.3f}$" + (f" {axes_units[0]}" if axes_units and axes_units[0] else "")
        else: 
            raise ValueError("cut_type must be 'horizontal' or 'vertical'")
            
        current_ax.plot(x_axis, y_axis, color=color, linewidth=2, label=label)
        current_ax.set_xlim(x_axis[0], x_axis[-1])
        current_ax.set_xlabel(xlabel)
        current_ax.set_ylabel(ylabel)
        
        if not current_ax.get_title():
            current_ax.set_title(title)
            
        current_ax.grid(True, alpha=0.3)
        
        if label: 
            current_ax.legend(frameon=False)
            
        fig.tight_layout()
        
        if save_name:
            filename = f"{save_name}.{save_format}"
            unique_filename = self._get_unique_filename(filename)
            fig.savefig(unique_filename, format=save_format)
            print(f"Slice saved to {unique_filename}")
            
        return current_ax

    def plot_spin_configuration(
        self, 
        orbital_energies: list[float], 
        state_dict: StateProperties, 
        title: str = "Quantum State Orbital Configuration"
    ) -> None:
        """
        Visualizes the orbital spin configurations of a many-body quantum state.

        Dynamically generates side-by-side subplots to represent superpositions (if applicable). 
        Reads directly from the 'single_particle_kets' output generated by the exact diagonalization solver.

        Parameters
        ----------
        orbital_energies : list of float
            List of orbital energies corresponding to the spatial levels (y-axis heights).
        state_dict : StateProperties
            Dictionary containing the state properties, specifically the 'single_particle_kets' 
            which is a list of tuples `(amplitude, ket_string, ket_array)` 
            representing the constituent single-particle states of the many-body state.
        title : str, optional
            Title of the overall plot. Default is "Quantum State Superposition".
            
        Raises
        ------
        ValueError
            If the length of the single-particle occupation array is not exactly twice 
            the length of the `orbital_energies` list.
        """
        single_particle_kets = state_dict['single_particle_kets']
        
        if not single_particle_kets:
            print("Warning: The provided state_dict contains no single particle states to plot.")
            return

        expected_ket_length = 2 * len(orbital_energies)
        actual_ket_length = len(single_particle_kets[0][2]) # occ_array of the first vector
        
        if actual_ket_length != expected_ket_length:
            raise ValueError(
                f"Dimension mismatch: The single-particle ket length ({actual_ket_length}) "
                f"must be exactly twice the number of orbital energies ({expected_ket_length})."
            )

        num_states = len(single_particle_kets)
        fig, axes = plt.subplots(1, num_states, figsize=(3 * num_states, 5), sharey=True)
        
        if num_states == 1:
            axes = [axes]
            
        x_min, x_max = 0.2, 0.8
        
        # Calculate y-axis margins 
        if len(orbital_energies) > 1:
            dy = max(orbital_energies) - min(orbital_energies)
            margin = dy * 0.2 if dy > 0 else 1.0
        else:
            margin = 1.0
            
        y_min = min(orbital_energies) - margin
        y_max = max(orbital_energies) + margin
        
        for idx, (coeff, _, occ_array) in enumerate(single_particle_kets):
            ax = axes[idx]
            
            # Format the amplitude coefficient
            if isinstance(coeff, complex) and coeff.imag == 0:
                coeff = float(coeff.real)
            if isinstance(coeff, float) and coeff.is_integer():
                coeff = int(coeff)
                
            coeff_str = f"{coeff:.3g}" if isinstance(coeff, float) else str(np.round(coeff, 3))
            
            if not str(coeff_str).startswith("-"):
                coeff_str = f"+ {coeff_str}"
            else:
                coeff_str = f"- {coeff_str[1:]}"

            for i, E in enumerate(orbital_energies):
                # Draw the orbital energy level
                ax.hlines(E, x_min, x_max, colors='black', linewidth=2)
                
                # Retrieve spin up/down for the i-th spatial orbital
                up_occ = occ_array[2 * i]
                down_occ = occ_array[2 * i + 1]
                
                symbol = ""
                if up_occ == 1 and down_occ == 1:
                    symbol = "↑↓"
                elif up_occ == 1:
                    symbol = "↑"
                elif down_occ == 1:
                    symbol = "↓"
                
                if symbol:
                    ax.text(0.5, E, symbol, ha='center', va='center', 
                            fontsize=35, color='blue', weight='bold')
                    
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            
            ax.set_title(f"Amplitude: {coeff_str}", fontsize=14)
            
            if idx == 0:
                ax.set_ylabel("Single-Particle Energy")
        title_str = title + f"\nEnergy = {state_dict['energy']}"
        fig.suptitle(title_str, fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()

    def save_results(self, results: dict[str, np.ndarray], filename: str = "rate_equations_results.npz") -> None:
        """
        Saves the numerical results dictionary to a compressed .npz file.

        Parameters
        ----------
        results : dict of str to np.ndarray
            Dictionary of numerical grid results to save.
        filename : str, optional
            Output filename. Default is "rate_equations_results.npz".
        """
        np.savez_compressed(filename, **results)
        print(f"Saved results to {filename}")