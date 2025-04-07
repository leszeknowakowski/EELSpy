import hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, mkQApp
import numpy as np
import os
from PyQt5.QtCore import Qt, QRectF, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QMainWindow, QDockWidget, QMenuBar, QToolBar,
                             QStatusBar, QAction, QFileDialog, QHBoxLayout, QMdiArea, QMdiSubWindow, QLabel,
                             QPushButton, QGridLayout, QListWidget, QLineEdit, QFormLayout, QComboBox, QCheckBox)
from PyQt5.QtGui import QCursor, QFont, QPen, QBrush
#from scipy import optimize
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz
import time

font = {'color': 'b', 'font-size': '14pt'}
my_font = QFont("Times", 10, QFont.Bold)
my_font.setBold(True)


def set_plot_fonts(plot, color='w', font="Times", size=10):
    """
    Set fonts for the axis labels and ticks of a pyqtgraph plot.

    Parameters:
        plot (pyqtgraph.PlotItem): The plot to modify.
        color (str): The color of the text.
        font (str): The font family.
        size (int): The font size.
    """
    for label in ['left', 'right', 'top', 'bottom']:
        my_font = QFont(font, size, QFont.Bold)
        plot.getAxis(label).setTickFont(my_font)
        plot.getAxis(label).setTextPen(color)


class EELS:
    """
    Class for handling Electron Energy Loss Spectroscopy (EELS) data.
    """
    def __init__(self, file):
        """
        Load EELS data from a file.

        Parameters:
            file (str): Path to the EELS data file.
        """
        loaded = hs.load(file)
        # Always work with a list
        self.signals = loaded if isinstance(loaded, list) else [loaded]

        # Initialize attributes
        self.eels_highloss = None
        self.eels_lowloss = None
        self.haadf = None
        self.edx = None

        for signal in self.signals:
            name = signal.metadata['General']['title'].lower()
            signal_type = type(signal).__name__.lower()

            if 'high' in name and 'eels' in name:
                self.eels_highloss = signal
            elif 'low' in name and 'eels' in name:
                self.eels_lowloss = signal
            elif 'haadf' in name or 'stemhaadf' in signal_type:
                self.haadf = signal
            elif 'edx' in name or 'eds' in name or 'eds' in signal_type:
                self.edx = signal
            elif 'eels' in name or 'eels' in signal_type:
                # Fallback for unnamed EELS
                if self.eels_lowloss is None:
                    self.eels_lowloss = signal
                elif self.eels_highloss is None:
                    self.eels_highloss = signal

        # Just in case: store all signals
        self.all_signals = self.signals
        #s = hs.load(file)
        ######################################################### change later!!! ############################
        #self.haadf = s[0]  # High-angle annular dark field image
        #self.eels_lowloss = s[1]  # Low-loss EELS spectrum
        #self.eels_highloss = s[2]  # High-loss EELS spectrum

        #self.eels_highloss = s
    def get_spectral_resolution(self):
        """
        Get the energy loss spectral resolution.

        Returns:
            float: The spectral resolution in energy loss.
        """
        return self.eels_highloss.axes_manager["Energy loss"].scale

    def get_offset(self):
        """
        Get the energy loss offset.

        Returns:
            float: The energy loss offset.
        """
        return self.eels_highloss.axes_manager["Energy loss"].offset


class MainWindow(QMainWindow):
    """
    Main application window for EELS visualization.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the main window, menus, toolbars, status bar, and MDI area.
        """
        print('starting')
        super().__init__(*args, **kwargs)
        self.setWindowTitle('EELSpy')
        self.resize(1200, 1000)

        # Setup menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        file_menu = self.menu_bar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(lambda : self.save_data("small_test_file", (1,5), (1,5)))
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        eels_menu = self.menu_bar.addMenu("EELS")
        calculate_shift_action = QAction("Calculate Shift", self)
        calculate_shift_action.triggered.connect(self.calculate_shift)
        eels_menu.addAction(calculate_shift_action)

        peak_setting_action = QAction("Peak Setting", self)
        peak_setting_action.triggered.connect(self.peak_settings)
        eels_menu.addAction(peak_setting_action)

        shift_setting_action = QAction("Shift Setting", self)
        shift_setting_action.triggered.connect(self.shift_settings)
        eels_menu.addAction(shift_setting_action)

        # Setup toolbar
        self.tool_bar = QToolBar("Main Toolbar", self)
        self.addToolBar(self.tool_bar)
        self.tool_bar.addAction(open_action)
        self.tool_bar.addAction(exit_action)

        # Setup status bar
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Setup MDI area
        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)

        # Create main graphics window
        self.graphics_window = pg.GraphicsLayoutWidget()
        self.main_subwindow = QMdiSubWindow()
        self.main_subwindow.setWidget(self.graphics_window)
        self.mdi_area.addSubWindow(self.main_subwindow)
        self.main_subwindow.show()

        self.spectrum_subwindow = None
        self.spectrum_plot = None
        self.data = None
        self.selected_pixel_roi = None  # Store the selected pixel for highlighting
        self.plots = {}

    def open_file(self):
        """
        Open a file dialog to select an EELS data file and load it.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "DM4 Files (*.dm4);;All Files (*)")
        if file_name:
            self.load_data(file_name)

    def load_data(self, file):
        """
        Load EELS data from a file and display it as an image.

        Parameters:
            file (str): Path to the EELS data file.
        """
        # TODO: load files many times with functionality
        # TODO: more flexible loading, with choosing the data to process
        print('loading data')
        self.path = file
        self.filename = os.path.basename(self.path)
        self.main_subwindow.setWindowTitle(f"{self.filename} EELS intensity")
        self.data = EELS(self.path)
        summed_data = np.sum(self.data.eels_highloss.data, axis=2)
        min_val, max_val = np.min(summed_data), np.max(summed_data)
        rescaled_data = (summed_data - min_val) / (max_val - min_val) * 100
        item = self.create_matrix(np.transpose(rescaled_data))
        self.add_plot(item, self.graphics_window, gradient="CET-L1", name="mainw")
        self.status_bar.showMessage(f"Loaded file: {file}")

    def save_data(self, name, x_region, y_region):
        #save only highloss now
        data = self.data.eels_highloss.data[x_region[0]:x_region[1], y_region[0]:y_region[1]]
        self.data.eels_highloss.data = data
        self.data.eels_highloss.save(name)

    def create_matrix(self, data):
        image_item = pg.ImageItem()
        image_item.setImage(data)
        image_item.mouseClickEvent = self.on_map_left_clicked
        return image_item

    def add_plot(self, item, parent, gradient="CET-D1A", name="", **kwargs):
        """
        Add a plot with an image item to a pyqtgraph parent widget.

        All other keyword arguments are passed to the ColorBarItem.
        """
        # TODO: add many plots, add title properly, force rectangular shape of pixels
        parent.clear()
        view = parent.addPlot()
        view.setLabel("left", "y dimension")
        view.setLabel("bottom", "x dimension")
        view.invertY(True)
        view.setDefaultPadding(0.0)
        view.addItem(item)
        view.showAxes(True, showValues=(True, True, False, False), size=20)
        view.getAxis('bottom').setHeight(20)
        view.getAxis('left').setWidth(50)

        width, height = item.image.shape
        base_size = 800
        aspect_ratio = width / height
        vb = view.getViewBox()
        vb.setAspectLocked(True)
        plot_width = base_size
        plot_height = int(base_size / aspect_ratio)
        self.main_subwindow.resize(plot_width, plot_height)
        self.main_subwindow.setMaximumHeight(1200)


        colorMap = pg.colormap.get(gradient)
        bar = pg.ColorBarItem(colorMap=colorMap, **kwargs)
        bar.setImageItem(item, insert_in=view)

        self.plots[name] = {
            "item": item,
            "plotItem": view,
            "roi": None,
        }

    def on_map_left_clicked(self, event):
        """
        Handle mouse clicks on the image to select a pixel region of interest (ROI).
        """
        # TODO: red ROI color edge when hovered
        if self.data is None:
            return
        if event.button() == 1 and event.modifiers() == Qt.ShiftModifier:
            clicked_item = event.currentItem
            clicked_name = None

            # Find which plot this item belongs to
            for name, info in self.plots.items():
                if info["item"] is clicked_item:
                    clicked_name = name
                    break

            if clicked_name is None:
                return

            plot_info = self.plots[clicked_name]
            pos = event.pos()
            x, y = int(pos.x()), int(pos.y())

            if plot_info["roi"] is None:
                roi = pg.RectROI([x - 0.5, y - 0.5], [1, 1], pen=pg.mkPen('r', width=2))
                plot_info["plotItem"].addItem(roi)
                roi.sigRegionChanged.connect(lambda: self.update_spectrum_from_roi(clicked_name))
                plot_info["roi"] = roi
            else:
                plot_info["roi"].setPos([x - 0.5, y - 0.5])

            self.update_spectrum_from_roi(clicked_name)

    def update_spectrum_from_roi(self, name):
        """
        Extracts all spectra inside the selected region of interest (ROI) and updates the spectrum plot.
        If no data or ROI is available, the function returns without modification.
        """
        # TODO: maybe auto fitting when ROI is moved?
        if self.data is None or name not in self.plots:
            return

        plot_info = self.plots[name]
        roi = plot_info["roi"]
        if roi is None:
            return

        roi_mask = roi.getArrayRegion(self.data.eels_highloss.data, plot_info["item"], axes=(1, 0))
        if roi_mask is not None and roi_mask.size > 0:
            summed_spectrum = np.sum(roi_mask, axis=(1, 0))
        else:
            summed_spectrum = np.zeros_like(self.data.eels_highloss.data[0, 0])
        # normalize
        summed_spectrum = summed_spectrum/max(summed_spectrum)

        spectrum_res = self.data.get_spectral_resolution()
        spectrum_offset = self.data.get_offset()
        x_values = np.array(range(len(summed_spectrum))) * spectrum_res + spectrum_offset

        if self.spectrum_subwindow is None:
            self.spectrum_plot = SpectrumPlot(self)
            self.spectrum_subwindow = QMdiSubWindow()
            self.spectrum_subwindow.setWindowTitle(f"{self.filename} EELS spectrum")
            self.spectrum_subwindow.setWidget(self.spectrum_plot)
            self.mdi_area.addSubWindow(self.spectrum_subwindow)
            self.spectrum_subwindow.show()

        pt = roi.pos()
        self.spectrum_plot.update_plot(x_values, summed_spectrum, pt.__reduce__()[1])

    def calculate_shift(self):
        """
        Initiates the shift calculation process using ShiftCalculator.
        Displays progress in the status bar and connects the completion signal
        to plot the shift map.
        """
        # TODO: many fitting functions, especially peak functions
        self.worker = ShiftCalculator(self.data, self.spectrum_plot)
        self.worker.progress_signal.connect(lambda c, d: self.status_bar.showMessage(f"progress: {c} out of {d}"))
        self.worker.end_signal.connect(lambda time: self.status_bar.showMessage(f"done, elapsed time: {time}"))
        self.worker.start()
        self.worker.end_signal.connect(lambda: self.plot_shift_map())

    def plot_shift_map(self, type="peak"):
        """
        Generates and displays a shift map after the shift calculation process completes.
        The shift map is displayed in a new subwindow within the MDI area.
        """
        # TODO:  dispatch from main EELS image plot!
        self.shift_window = pg.GraphicsLayoutWidget()
        self.shift_subwindow = QMdiSubWindow()
        self.shift_subwindow.setWidget(self.shift_window)
        self.shift_window.setWindowTitle(f"{self.filename} EELS {type} shift map")
        self.spectrum_subwindow.setWindowTitle(f'{type} shift image')
        self.mdi_area.addSubWindow(self.shift_subwindow)
        self.shift_subwindow.show()

        # Create and add the shift map to the new window
        if type == "edge":
            item = self.create_matrix(np.transpose(self.worker.intersects))
        elif type == "peak":
            item = self.create_matrix(np.transpose(self.worker.peaks_maximums))
        elif type == "maximum":
            item = self.create_matrix(np.transpose(self.worker.maximums))
        self.add_plot(item, self.shift_window, limits=(-1,1), rounding=0.01, label="shift")

    def get_peak_fitter(self):
        try:
            fitter = self.peak_settings_window.get_peak_fitter()
        except:
            self.peak_settings()
            fitter = self.peak_settings_window.get_peak_fitter()
        return fitter

    def peak_settings(self):
        self.peak_settings_window = PeakFitterConfigWindow()
        self.peak_settings_window.show()

    def shift_settings(self):
        self.shift_settings_window = ShiftCalculatorSettings()
        self.shift_settings_window.show()


class ShiftCalculatorSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shift Calculator Settings")
        self.setGeometry(100, 100, 300, 250)

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Dropdown for model selection
        self.shift_type_box = QComboBox()
        self.shift_type_box.addItems(["edge", "peak"])
        form_layout.addRow("Calcualte shift regarding to:", self.shift_type_box)

        layout.addLayout(form_layout)
        self.setLayout(layout)

    def get_shift_settings(self):
        return self.shift_type_box.currentText()

class PeakFitterConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PeakFitter Configuration")
        self.setGeometry(100, 100, 300, 250)
        self.peak_fitter = None

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Dropdown for model selection
        self.model_box = QComboBox()
        self.model_box.addItems(["gaussian", "lorentzian", "voigt", "pseudo_voigt"])
        form_layout.addRow("Model:", self.model_box)

        # LineEdits for float/int parameters
        self.prominence_input = QLineEdit("0.1")
        form_layout.addRow("Peak Prominence:", self.prominence_input)

        self.max_iter_input = QLineEdit("10")
        form_layout.addRow("Max Iterations:", self.max_iter_input)

        self.tol_input = QLineEdit("1e-4")
        form_layout.addRow("Tolerance:", self.tol_input)

        # Checkbox for verbose
        self.verbose_check = QCheckBox("Verbose Output")
        form_layout.addRow(self.verbose_check)

        # Button to create PeakFitter
        self.create_button = QPushButton("Create PeakFitter")
        self.create_button.clicked.connect(self.create_peak_fitter)
        layout.addLayout(form_layout)
        layout.addWidget(self.create_button)

        # Label to show result
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def create_peak_fitter(self):
        try:
            model = self.model_box.currentText()
            peak_prominence = float(self.prominence_input.text())
            max_iter = int(self.max_iter_input.text())
            tol = float(self.tol_input.text())
            verbose = self.verbose_check.isChecked()

            self.peak_fitter = {
                "model": model,
                "peak_prominence": peak_prominence,
                "max_iter": max_iter,
                "tol": tol,
                "verbose": verbose
            }
            self.status_label.setText("PeakFitter created successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create PeakFitter:\n{e}")

    def get_peak_fitter(self):
        """Return a PeakFitter instance based on current settings."""
        model = self.model_box.currentText()
        peak_prominence = float(self.prominence_input.text())
        max_iter = int(self.max_iter_input.text())
        tol = float(self.tol_input.text())
        verbose = self.verbose_check.isChecked()

        return PeakFitter(
            model=model,
            peak_prominence=peak_prominence,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )

class ShiftCalculator(QThread):
    progress_signal = pyqtSignal(int, int)
    end_signal = pyqtSignal(float)

    def __init__(self, data, spectrum_plot):
        super().__init__()
        self.data = data
        self.spectrum_plot = spectrum_plot
        self.intersects = []
        self.maximums = []
        self.peaks_maximums = []

    def run(self):
        self._calculate_shift(mode="peak")

    def _calculate_shift(self, mode):
        """
        Generalized internal method to perform shift calculations for maximum, peak, or edge.

        Parameters:
        mode : str
            One of 'max', 'peak', or 'edge' to indicate the type of shift to calculate.
        """
        tic = time.time()
        spectrum_res = self.data.get_spectral_resolution()
        spectrum_offset = self.data.get_offset()
        counter = 0
        data_array = self.data.eels_highloss.data
        data_length = len(data_array) * len(data_array[0])
        reference_peak_center = self.spectrum_plot.saved_peaks['Peak_1']['energy'] if mode in ("max", "peak") else None #change hard coding!!

        result_matrix = []

        for i, line in enumerate(data_array):
            line_results = []
            for j, spectrum in enumerate(line):
                counter += 1
                x_values = np.array(range(len(spectrum))) * spectrum_res + spectrum_offset
                self.spectrum_plot.update_plot(x_values, spectrum, (i, j), update_roi=False, plot=False)

                if mode == "max":
                    self.spectrum_plot.fit_background(plot=False)
                    max_val = self.spectrum_plot.find_max()
                    value = (reference_peak_center - max_val) if self.significant_signal("peak") else np.nan

                elif mode == "peak":
                    self.spectrum_plot.fit_background(plot=False)
                    fit_result = self.spectrum_plot.fit_peak(plot=False, view_center=False)
                    if self.significant_signal("peak") and fit_result['fit_successful']:
                        value = reference_peak_center - fit_result['center']
                    else:
                        value = np.nan

                elif mode == "edge":
                    self.spectrum_plot.fit_edge(plot=False)
                    intersect = self.spectrum_plot.fit_background(plot=False)
                    value = (intersect - 825.8) * 100 if self.significant_signal("edge") else 0

                if i == 50 and j == 50:
                    print("stop")

                line_results.append(value)
                self.progress_signal.emit(counter, data_length)
                time.sleep(0.0005 if mode in ("max", "peak") else 0.001)

            result_matrix.append(line_results)

        result_matrix = np.array(result_matrix)

        if mode == "max":
            self.maximums = result_matrix
        elif mode == "peak":
            self.peaks_maximums = result_matrix
        elif mode == "edge":
            self.intersects = result_matrix

        toc = time.time()
        self.end_signal.emit(toc - tic)

    def peak_calculation(self):
        self._calculate_shift("peak")

    def maximum_calculator(self):
        self._calculate_shift("max")

    def edge_calculation(self):
        self._calculate_shift("edge")

    def significant_signal(self, type="peak"):
        bg_noise = self.spectrum_plot.bg_fit_noise_level
        signal = (
            self.spectrum_plot.egde_signal_strength if type == "edge"
            else self.spectrum_plot.peak_signal_strength
        )
        return bg_noise * 6 < signal


class LinearFitResult:
    """
    Represents the result of a linear fit, storing the slope and intercept values.
    """
    def __init__(self, slope: float, intercept: float) -> None:
        """
        Initializes a LinearFitResult instance with a given slope and intercept.
        """
        self.slope: float = slope
        self.intercept: float = intercept

    def evaluate(self, x: float) -> float:
        """
        Evaluates the linear function at a given x-value.
        """
        return self.slope * x + self.intercept

    def __str__(self) -> str:
        """
        Returns a string representation of the linear equation.
        """
        return f"y = {self.slope:.6f}x + {self.intercept:.6f}"



class PeakFitter:
    """
    Fits a single peak in spectral data using Gaussian, Lorentzian,
    Voigt, or pseudo-Voigt models. Supports iterative fitting,
    region-of-interest filtering, and automatic parameter estimation.

    Parameters
    ----------
    model : str
        The peak model to use: 'gaussian', 'lorentzian', 'voigt', or 'pseudo_voigt'.
    peak_prominence : float
        Minimum prominence of peaks used for initial guess.
    max_iter : int
        Maximum number of iterations for refitting until convergence.
    tol : float
        Relative change tolerance for convergence.
    verbose : bool
        If True, print convergence details.
    """
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    from scipy.special import wofz

    def __init__(self, model='gaussian', peak_prominence=0.1, max_iter=10, tol=1e-4, verbose=False):
        self.model = model
        self.peak_prominence = peak_prominence
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self._init_models()

    def _init_models(self):
        # Define model functions and their param counts
        self.model_funcs = {
            'gaussian': (lambda x, a, c, w, y0: y0 + a * np.exp(-(x - c)**2 / (2 * w**2)), 3),
            'lorentzian': (lambda x, a, c, w, y0: y0 + a * (w**2 / ((x - c)**2 + w**2)), 3),
            'voigt': (lambda x, a, c, s, g, y0: y0 + a * np.real(wofz(((x - c) + 1j * g) / (s * np.sqrt(2)))) / (s * np.sqrt(2*np.pi)), 4),
            'pseudo_voigt': (
                lambda x, a, c, w, eta, y0: y0 + eta * (w**2 / ((x - c)**2 + w**2)) + (1 - eta) * a * np.exp(-(x - c)**2 / (2 * w**2)), 4
            )
        }

    def _initial_guess(self, x, y):
        peaks, _ = find_peaks(y, prominence=self.peak_prominence)
        if len(peaks) == 0:
            raise ValueError("No peaks found for initial guess.")

        idx = peaks[0]
        cen = x[idx]
        amp = y[idx]
        wid = (max(x) - min(x)) / 10
        y0 = min(y)

        if self.model in ['gaussian', 'lorentzian']:
            return [amp, cen, wid, y0]
        elif self.model == 'voigt':
            return [amp, cen, wid, wid / 2, y0]
        elif self.model == 'pseudo_voigt':
            return [amp, cen, wid, 0.5, y0]

    def _compute_parameters(self, popt):
        """Calculate useful peak parameters from fit."""
        if self.model == 'gaussian':
            a, c, w, y0 = popt
            fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(w)
            area = a * abs(w) * np.sqrt(2 * np.pi)
            max_val = a
        elif self.model == 'lorentzian':
            a, c, w, y0 = popt
            fwhm = 2 * abs(w)
            area = np.pi * a * abs(w)
            max_val = a
        elif self.model == 'voigt':
            a, c, s, g, y0 = popt
            fwhm = 0.5346 * 2 * g + np.sqrt(0.2166 * (2 * g)**2 + (2.3548 * s)**2)
            max_val = a * np.real(wofz(0)) / (s * np.sqrt(2 * np.pi))
            area = a  # Approximate
        elif self.model == 'pseudo_voigt':
            a, c, w, eta, yo = popt
            fwhm = 2 * w  # Approximate
            area = a * (eta * np.pi * w + (1 - eta) * w * np.sqrt(2 * np.pi))
            max_val = a
        else:
            raise ValueError("Unknown model")

        return {
            'amplitude': a,
            'center': c,
            'FWHM': fwhm,
            'maximum': max_val,
            'area': area
        }

    def fit(self, xdata, ydata, roi=None):
        """
        Fit a peak to the provided x and y data.

        Parameters
        ----------
        xdata : np.ndarray
            X-axis data (e.g., energy or wavelength).
        ydata : np.ndarray
            Y-axis data (e.g., intensity).
        roi : tuple or None
            Region of interest as (xmin, xmax). If None, fit the entire range.

        Returns
        -------
        popt : list
            Optimal parameters for the fit.
        yfit : np.ndarray
            Evaluated fitted function.
        xfit : np.ndarray
            X values used for fitting (ROI-restricted if applicable).
        yfit_data : np.ndarray
            Y values used for fitting.
        peak_info : dict
            Computed peak parameters (center, height, FWHM, area, etc.)
        """
        if self.model not in self.model_funcs:
            raise ValueError(f"Unsupported model: {self.model}")

        func, _ = self.model_funcs[self.model]

        if roi is not None:
            xmin, xmax = roi
            mask = (xdata >= xmin) & (xdata <= xmax)
            xdata = xdata[mask]
            ydata = ydata[mask]
            if len(xdata) < 5:
                raise ValueError("ROI too narrow or no data in range.")

        p0 = self._initial_guess(xdata, ydata)
        prev = np.array(p0)

        for i in range(self.max_iter):
            try:
                popt, _ = curve_fit(func, xdata, ydata, p0=prev)
            except RuntimeError:
                raise FitNotConverged("Fit did not converge.")

            delta = np.abs(popt - prev) / (np.abs(prev) + 1e-8)
            if self.verbose:
                print(f"[Iter {i+1}] popt = {popt}, Δ = {delta}")

            if np.all(delta < self.tol):
                break
            prev = popt
        else:
            if self.verbose:
                print("Warning: Maximum iterations reached without convergence.")

        yfit = func(xdata, *popt)
        peak_info = self._compute_parameters(popt)
        self.amplitude = np.max(yfit) - np.min(yfit)

        return popt, yfit, xdata, ydata, peak_info


class SpectrumPlot(QWidget):
    """
    A widget for plotting and analyzing spectral data.

    This class provides functionality to display spectra, fit background and edge regions,
    determine intersection points, find maximum and save edge data.
    """

    def __init__(self, main_window):
        """
        Initializes the SpectrumPlot widget.

        Creates the main layout, plot widget, controls for background fitting,
        intersection calculation, and saving edge data.
        """
        super().__init__()
        self.main_window = main_window

        # Main layout
        main_layout = QVBoxLayout()

        # Pixel position label
        self.label = QLabel("", self)
        main_layout.addWidget(self.label)

        # Plot widget
        self.plot_widget = pg.PlotWidget(background='w')
        self.plot_widget.setWindowTitle("EELS Spectrum Plot")
        self.plot_widget.setLabel("left", "counts", **font)
        self.plot_widget.setLabel("bottom", "energy", units="keV", **font)
        set_plot_fonts(self.plot_widget, size=11, color='k')
        main_layout.addWidget(self.plot_widget)

        # Controls layout
        controls_layout = QGridLayout()

        # Background fitting controls
        self.fit_bg_button = QPushButton("Fit Background")
        self.fit_bg_button.clicked.connect(lambda: self.fit_background(plot=True))
        controls_layout.addWidget(self.fit_bg_button, 0, 0)

        self.reset_bg_button = QPushButton("Reset Background")
        self.reset_bg_button.clicked.connect(self.reset_background)
        controls_layout.addWidget(self.reset_bg_button, 0, 1)

        # Edge Intersection control section
        self.fit_edge_buuton = QPushButton("Fit Edge")
        self.fit_edge_buuton.clicked.connect(lambda: self.fit_edge(plot=True))
        controls_layout.addWidget(self.fit_edge_buuton, 1, 0)

        controls_layout.addWidget(QLabel("Intersection Energy:"), 1, 1)
        self.intersection_value = QLineEdit()
        self.intersection_value.setReadOnly(True)
        controls_layout.addWidget(self.intersection_value, 1, 2)

        self.edge_name_input = QLineEdit()
        self.edge_name_input.setPlaceholderText("Edge name")
        controls_layout.addWidget(self.edge_name_input, 1, 3)

        self.save_edge_button = QPushButton("Save Edge")
        self.save_edge_button.clicked.connect(self.save_edge)
        controls_layout.addWidget(self.save_edge_button, 1, 4)

        # Saved edges list
        self.saved_edges_list = QListWidget()
        controls_layout.addWidget(QLabel("Saved Edges:"), 3, 0)
        controls_layout.addWidget(self.saved_edges_list, 3, 0, 1, 4)

        # peak fit controls
        self.fit_peak_button = QPushButton("Fit Peak")
        self.fit_peak_button.clicked.connect(lambda :self.fit_peak(plot=True))
        controls_layout.addWidget(self.fit_peak_button, 4, 0)

        controls_layout.addWidget(QLabel("Peak Energy:"), 4, 1)
        self.max_peak_value = QLineEdit()
        self.max_peak_value.setReadOnly(True)
        controls_layout.addWidget(self.max_peak_value, 4, 2)

        self.peak_name_input = QLineEdit()
        self.peak_name_input.setPlaceholderText("Peak name")
        controls_layout.addWidget(self.peak_name_input, 4, 3)
        controls_layout.addWidget(self.max_peak_value, 4, 4)

        self.save_peak_button = QPushButton("Save Edge")
        self.save_peak_button.clicked.connect(self.save_peak)
        controls_layout.addWidget(self.save_peak_button, 4, 4)

        self.saved_peak_list = QListWidget()
        controls_layout.addWidget(QLabel("Saved Edges:"), 5, 0)
        controls_layout.addWidget(self.saved_peak_list, 5, 0, 1, 4)

        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

        # Data storage
        self.xaxis = None
        self.spectrum = None
        self.bg_roi = None
        self.edge_roi = None
        self.bg_fit = None
        self.edge_fit = None
        self.bg_fit_line = None
        self.edge_fit_line = None
        self.peak_line = None
        self.intersection_point = None
        self.spectrum_curve = None
        self.saved_edges = {}
        self.saved_peaks = {}
        self.peak_line

    def update_plot(self, xaxis, energies, position, update_roi=True, plot=True):
        """
        Updates the plot with new spectral data.

        Args:
            xaxis (array-like): X-axis values (e.g., energy in keV).
            energies (array-like): Corresponding spectral intensities.
            position (tuple): Pixel position label to display.
            update_roi (bool, optional): Whether to update the ROIs (default: True).
        """
        self.xaxis = xaxis
        self.spectrum = energies
        if plot:
            if self.spectrum_curve is not None:
                self.spectrum_curve.setData(xaxis, energies)
            else:
                self.spectrum_curve = self.plot_widget.plot(xaxis, energies, pen='k')
            self.label.setText(f"Pixel: {position}")

            if update_roi:
                self.update_roi(xaxis, energies)

    def update_roi(self, xaxis, energies):
        """
        Updates or creates regions of interest (ROIs) for fitting.

        Args:
            xaxis (array-like): X-axis values.
            energies (array-like): Corresponding spectral intensities.
        """
        # Create ROIs if they don't exist
        if self.bg_roi is None:
            # Create background ROI around 20% of the x-axis range
            x_range = xaxis[-1] - xaxis[0]
            roi_width = x_range * 0.05
            start_x = xaxis[700] + x_range * 0.01

            # Find y values within this x range
            mask = (xaxis >= start_x) & (xaxis <= start_x + roi_width)
            if np.any(mask):
                y_values = energies[mask]
                min_y = np.min(y_values)
                max_y = np.max(y_values)
                height = max_y - min_y

                # Create ROI
                self.bg_roi = pg.LinearRegionItem(
                    values=[start_x, start_x + roi_width],
                    brush=pg.mkBrush(128, 128, 255, 50),
                    pen=pg.mkPen('b', width=2),
                    movable=True
                )
                self.plot_widget.addItem(self.bg_roi)
                self.bg_roi.sigRegionChanged.connect(self.on_bg_roi_changed)

        # Create edge ROI if it doesn't exist
        if self.edge_roi is None:
            # Create edge ROI around 60% of the x-axis range
            x_range = xaxis[-1] - xaxis[0]
            roi_width = x_range * 0.02
            start_x = xaxis[725] + x_range * 0.1

            self.edge_roi = pg.LinearRegionItem(
                values=[start_x, start_x + roi_width],
                brush=pg.mkBrush(255, 128, 128, 50),
                pen=pg.mkPen('r', width=2),
                movable=True,
                hoverPen='y'

            )
            self.plot_widget.addItem(self.edge_roi)
            self.edge_roi.sigRegionChanged.connect(self.on_edge_roi_changed)

        # Reset fits
        self.bg_fit = None
        self.edge_fit = None
        if self.bg_fit_line is not None:
            self.plot_widget.removeItem(self.bg_fit_line)
            self.bg_fit_line = None
        if self.edge_fit_line is not None:
            self.plot_widget.removeItem(self.edge_fit_line)
            self.edge_fit_line = None
        if self.intersection_point is not None:
            self.plot_widget.removeItem(self.intersection_point)
            self.intersection_point = None

        self.intersection_value.setText("")

    def fit_background(self, plot=True):
        """
        Fits a linear background to the selected region and optionally plots it.

        Args:
            plot (bool, optional): Whether to plot the fitted background (default: True).

        Returns:
            float or None: Intersection point if both background and edge fits are available.
        """
        if self.xaxis is None or self.spectrum is None or self.bg_roi is None:
            return

        # Get the region boundaries
        min_x, max_x = self.bg_roi.getRegion()

        # Find the indices that fall within the region
        mask = (self.xaxis >= min_x) & (self.xaxis <= max_x)
        x_values = self.xaxis[mask]
        y_values = self.spectrum[mask]

        if len(x_values) < 2:  # Need at least 2 points for linear fit
            return

        # Fit a line to the data
        params, residuals, rank, sing, rcond = np.polyfit(x_values, y_values, 1, full=True)
        slope, intercept = params

        # Store the fit
        self.bg_fit = LinearFitResult(slope, intercept)
        self.bg_fit_noise_level = np.std(y_values - (slope*x_values + intercept))

        if plot==True:
            # Update the background fit line
            if self.bg_fit_line is not None:
                self.plot_widget.removeItem(self.bg_fit_line)

            # Create a line that spans the entire x-axis
            x_min, x_max = self.xaxis[0], self.xaxis[-1]
            x_fit = np.array([x_min, x_max])
            y_fit = self.bg_fit.evaluate(x_fit)

            # Plot the fit line
            self.bg_fit_line = self.plot_widget.plot(x_fit, y_fit, pen=pg.mkPen('b', width=2, style=Qt.DashLine))

        # If we have both fits, calculate intersection
        if self.edge_fit is not None:
            intersection = self.calculate_intersection(plot)
            return intersection

    def on_bg_roi_changed(self):
        """
        Callback function for changes in the background ROI.
        Does nothing, as the background fit is explicitly triggered by a button press.
        """
        # Background fit must be explicitly triggered by button press
        pass

    def on_edge_roi_changed(self):
        """
        Callback function for changes in the edge ROI.
        Automatically triggers the edge fitting process.
        """
        # Automatically update edge fit when ROI changes
        #self.fit_edge()
        pass

    def fit_edge(self, plot=True):
        """
        Fits a linear edge function to the selected region and optionally plots it.

        Args:
            plot (bool, optional): Whether to plot the fitted edge (default: True).

        Returns:
            float or None: Intersection point if both background and edge fits are available.
        """
        if self.xaxis is None or self.spectrum is None or self.edge_roi is None:
            return

        # Get the region boundaries
        min_x, max_x = self.edge_roi.getRegion()

        # Find the indices that fall within the region
        mask = (self.xaxis >= min_x) & (self.xaxis <= max_x)
        x_values = self.xaxis[mask]
        y_values = self.spectrum[mask]

        if len(x_values) < 2:  # Need at least 2 points for linear fit
            return

        # Fit a line to the data
        params, residuals,rank, sing, rcond = np.polyfit(x_values, y_values, 1, full=True)
        slope, intercept = params

        # Store the fit
        self.edge_fit = LinearFitResult(slope, intercept)
        self.egde_signal_strength = np.max(y_values) - np.min(y_values)
        if plot == True:
            # Update the edge fit line
            if self.edge_fit_line is not None:
                self.plot_widget.removeItem(self.edge_fit_line)

            # Create a line that spans the entire x-axis
            x_min, x_max = self.xaxis[0], self.xaxis[-1]
            x_fit = np.array([x_min, x_max])
            y_fit = self.edge_fit.evaluate(x_fit)

            # Plot the fit line
            self.edge_fit_line = self.plot_widget.plot(x_fit, y_fit, pen=pg.mkPen('r', width=2, style=Qt.DashLine))

        # If we have both fits, calculate intersection
        if self.bg_fit is not None:
            intersection = self.calculate_intersection(plot)
            return intersection

    def fit_peak(self, model='pseudo_voigt', plot=True, view_center=True):

        #fitter = PeakFitter(model, max_iter=100, verbose=True)
        fitter = self.main_window.get_peak_fitter()
        try:
            popt, yfit, xfit, yroi, info = fitter.fit(self.xaxis, self.spectrum, roi=self.edge_roi.getRegion())
            self.peak_signal_strength = fitter.amplitude
            if plot == True:
                if self.peak_line is not None:
                    self.plot_widget.removeItem(self.peak_line)
                # Plot the fit line
                self.peak_line = self.plot_widget.plot(
                    xfit,
                    yfit,
                    pen=pg.mkPen('g', width=2, style=Qt.DashLine)
                )

            center = info['center']
            if view_center:
                self.max_peak_value.setText(f"{center:.2f}")
            #self.peak_signal_strength = info['maximum']
            return {
                'fit_successful': True,
                'popt': popt,
                'yfit': yfit,
                'xfit': xfit,
                'yroi': yroi,
                'info': info,
                'center': center
            }
        except FitNotConverged:
            center = 'NaN'
            #print("Fit not converged")
            if view_center:
                self.max_peak_value.setText(center)
        return {
            'fit_successful': False,
            'center': center
        }

    def reset_background(self):
        """
        Resets the background fit and removes related graphical elements.
        """
        if self.bg_fit_line is not None:
            self.plot_widget.removeItem(self.bg_fit_line)
            self.bg_fit_line = None

        self.bg_fit = None

        # Remove intersection point if it exists
        if self.intersection_point is not None:
            self.plot_widget.removeItem(self.intersection_point)
            self.intersection_point = None

        self.intersection_value.setText("")

    def find_maximum(self, ):
        max = np.max(self.spectrum)
        return max

    def calculate_intersection(self, plot=True):
        """
        Calculates the intersection point between the background and edge fits.

        Args:
            plot (bool, optional): Whether to plot the intersection point (default: True).

        Returns:
            float or None: The intersection x-coordinate, or None if no valid intersection exists.
        """
        if self.bg_fit is None or self.edge_fit is None:
            return

        # Solve for the intersection point: bg_slope*x + bg_intercept = edge_slope*x + edge_intercept
        # (bg_slope - edge_slope)*x = edge_intercept - bg_intercept
        # x = (edge_intercept - bg_intercept) / (bg_slope - edge_slope)

        if self.bg_fit.slope == self.edge_fit.slope:
            # Lines are parallel, no intersection
            return

        x_intersect = (self.edge_fit.intercept - self.bg_fit.intercept) / (self.bg_fit.slope - self.edge_fit.slope)
        y_intersect = self.bg_fit.evaluate(x_intersect)

        # Display the intersection point
        if plot == True:
            if self.intersection_point is not None:
                self.plot_widget.removeItem(self.intersection_point)

            self.intersection_point = pg.ScatterPlotItem()
            self.intersection_point.addPoints(x=[x_intersect], y=[y_intersect],
                                              brush=pg.mkBrush('g'), size=10, symbol='o')
            self.plot_widget.addItem(self.intersection_point)

            # Update the value display
            self.intersection_value.setText(f"{x_intersect:.6f}")

        return x_intersect

    def save_edge(self):
        """
        Saves the detected edge energy along with its associated ROIs.
        """
        if not self.intersection_value.text():
            return

        edge_name = self.edge_name_input.text().strip()
        if not edge_name:
            edge_name = f"Edge_{len(self.saved_edges) + 1}"

        energy_value = float(self.intersection_value.text())

        min_bg, max_bg = self.bg_roi.getRegion()
        min_edge, max_edge = self.edge_roi.getRegion()

        # Save the edge
        self.saved_edges[edge_name] = {"energy": energy_value, "min_bg": min_bg, "max_bg": max_bg, "min_edge": min_edge, "max_edge": max_edge}

        # Add to the list
        self.saved_edges_list.addItem(f"{edge_name}: {energy_value:.2f} keV, bg ROI: min {min_bg:.2f}, max {max_bg:.2f}, edge ROI: min {min_edge:.2f} max {max_edge:.2f}")

        # Clear the edge name field
        self.edge_name_input.clear()

    def save_peak(self):
        """
        Saves the detected peak energy along with its associated ROIs.
        """
        if not self.max_peak_value.text():
            return
        peak_name = self.peak_name_input.text().strip()
        if not peak_name:
            peak_name = f"Peak_{len(self.saved_peaks)+1}"

        peak_energy = float(self.max_peak_value.text())

        # maybe change ROI to peak-specific
        min_bg, max_bg = self.bg_roi.getRegion()
        min_edge, max_edge = self.edge_roi.getRegion()

        self.saved_peaks[peak_name] = {"energy": peak_energy, "min_bg": min_bg, "max_bg": max_bg, "min_edge": min_edge, "max_edge": max_edge}
        self.saved_peak_list.addItem(f"{peak_name}: {peak_energy:.2f} eV, bg ROI: min {min_bg:.2f}, max {max_bg:.2f}, edge ROI: min {min_edge:.2f} max {max_edge:.2f}")

    def calculate_shift(self, edge):
        print("calculating shift...")

class FitNotConverged(Exception):
    "Fit did not converge."


if __name__ == '__main__':
    tic = time.time()
    mkQApp("Main")
    main_window = MainWindow()
    #main_window.load_data(r"D:\OneDrive - Uniwersytet Jagielloński\Studia\python\EELSpy\pythonProject1\small_test_file.hspy")
    main_window.load_data(r"D:\OneDrive - Uniwersytet Jagielloński\Studia\python\EELSpy\pythonProject1\tests\reference\STEM SI6.dm4")
    main_window.show()
    toc = time.time()
    print(toc - tic)
    pg.exec()
