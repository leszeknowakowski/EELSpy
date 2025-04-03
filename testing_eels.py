import hyperspy.api as hs
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, mkQApp
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QMainWindow, QDockWidget, QMenuBar, QToolBar,
                             QStatusBar, QAction, QFileDialog, QHBoxLayout, QMdiArea, QMdiSubWindow, QLabel,
                             QPushButton, QGridLayout, QListWidget, QLineEdit)
from PyQt5.QtGui import QCursor, QFont, QPen, QBrush
from scipy import optimize
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
        s = hs.load(file)
        self.haadf = s[0]  # High-angle annular dark field image
        self.eels_lowloss = s[1]  # Low-loss EELS spectrum
        self.eels_highloss = s[2]  # High-loss EELS spectrum

    def get_spectral_resolution(self):
        """
        Get the energy loss spectral resolution.

        Returns:
            float: The spectral resolution in energy loss.
        """
        return self.eels_lowloss.axes_manager["Energy loss"].scale

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
        super().__init__(*args, **kwargs)
        self.setWindowTitle('EELSpy')
        self.resize(1200, 1000)

        # Setup menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        file_menu = self.menu_bar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        calculate_shift_action = QAction("Calculate Shift", self)
        calculate_shift_action.triggered.connect(self.calculate_shift)
        file_menu.addAction(calculate_shift_action)

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
        self.data = EELS(file)
        summed_data = np.sum(self.data.eels_highloss.data, axis=2)
        min_val, max_val = np.min(summed_data), np.max(summed_data)
        rescaled_data = (summed_data - min_val) / (max_val - min_val) * 100
        item = self.create_matrix(np.transpose(rescaled_data))
        self.add_plot(item, self.graphics_window, name="mainw")
        self.status_bar.showMessage(f"Loaded file: {file}")

    def create_matrix(self, data):
        """
        Create a pyqtgraph image item from matrix data.
        """
        self.corrMatrix = data
        self.correlogram = pg.ImageItem()
        self.correlogram.setImage(self.corrMatrix)
        self.correlogram.mouseClickEvent = self.on_map_left_clicked
        return self.correlogram

    def add_plot(self, item, parent, name=""):
        """
        Add a plot with an image item to a pyqtgraph parent widget.
        """
        parent.clear()
        self.plotItem = parent.addPlot()
        self.plotItem.setLabel("left", "y dimension")
        self.plotItem.setLabel("bottom", "x dimension")
        self.plotItem.invertY(True)
        self.plotItem.setDefaultPadding(0.0)
        self.plotItem.addItem(item)
        self.plotItem.showAxes(True, showValues=(True, True, False, False), size=20)
        self.plotItem.getAxis('bottom').setHeight(10)

        colorMap = pg.colormap.get("CET-D1A")
        bar = pg.ColorBarItem(colorMap=colorMap)
        bar.setImageItem(item, insert_in=self.plotItem)

    def on_map_left_clicked(self, event):
        """
        Handle mouse clicks on the image to select a pixel region of interest (ROI).
        """
        if self.data is None:
            return
        if event.button() == 1 and event.modifiers() == Qt.ShiftModifier:
            pos = event.pos()
            x, y = int(pos.x()), int(pos.y())

            if self.selected_pixel_roi is None:
                self.selected_pixel_roi = pg.RectROI([x - 0.5, y - 0.5], [1, 1], pen=pg.mkPen('r', width=2))
                self.plotItem.addItem(self.selected_pixel_roi)
                self.selected_pixel_roi.sigRegionChanged.connect(self.update_spectrum_from_roi)
            else:
                self.selected_pixel_roi.setPos([x - 0.5, y - 0.5])
            self.update_spectrum_from_roi()

    def update_spectrum_from_roi(self):
        """
        Extracts all spectra inside the selected region of interest (ROI) and updates the spectrum plot.
        If no data or ROI is available, the function returns without modification.
        """
        if self.data is None or self.selected_pixel_roi is None:
            return

        # Extract the region of interest (ROI) from the high-loss EELS data
        roi_mask = self.selected_pixel_roi.getArrayRegion(self.data.eels_highloss.data, self.correlogram, axes=(1, 0))

        if roi_mask is not None and roi_mask.size > 0:
            # Sum spectra across the ROI
            summed_spectrum = np.sum(roi_mask, axis=(1, 0))
        else:
            # If ROI is empty, create a zero spectrum
            summed_spectrum = np.zeros_like(self.data.eels_highloss.data[0, 0])

        # Get spectral resolution and offset from the data
        spectrum_res = self.data.get_spectral_resolution()
        spectrum_offset = self.data.get_offset()
        x_values = np.array(range(len(summed_spectrum))) * spectrum_res + spectrum_offset

        # Create a new spectrum window if it does not exist
        if self.spectrum_subwindow is None:
            self.spectrum_plot = SpectrumPlot()
            self.spectrum_subwindow = QMdiSubWindow()
            self.spectrum_subwindow.setWidget(self.spectrum_plot)
            self.mdi_area.addSubWindow(self.spectrum_subwindow)
            self.spectrum_subwindow.show()

        # Get the position of the selected ROI and update the spectrum plot
        pt = self.selected_pixel_roi.pos()
        self.spectrum_plot.update_plot(x_values, summed_spectrum, pt.__reduce__()[1])

    def calculate_shift(self):
        """
        Initiates the shift calculation process using ShiftCalculator.
        Displays progress in the status bar and connects the completion signal
        to plot the shift map.
        """
        self.worker = ShiftCalculator(self.data, self.spectrum_plot)
        self.worker.progress_signal.connect(lambda c, d: self.status_bar.showMessage(f"progress: {c} out of {d}"))
        self.worker.end_signal.connect(lambda time: self.status_bar.showMessage(f"done, elapsed time: {time}"))
        self.worker.start()
        self.worker.end_signal.connect(self.plot_shift_map)

    def plot_shift_map(self):
        """
        Generates and displays a shift map after the shift calculation process completes.
        The shift map is displayed in a new subwindow within the MDI area.
        """
        self.shift_window = pg.GraphicsLayoutWidget()
        self.shift_subwindow = QMdiSubWindow()
        self.shift_subwindow.setWidget(self.shift_window)
        self.mdi_area.addSubWindow(self.shift_subwindow)
        self.shift_subwindow.show()

        # Create and add the shift map to the new window
        item = self.create_matrix(np.transpose(self.worker.intersects))
        self.add_plot(item, self.shift_window)


class ShiftCalculator(QThread):
    """
    A worker thread that calculates the shift map for EELS spectra.
    It iterates over the spectral data, fits edges and backgrounds, and stores the computed shifts.
    """
    progress_signal = pyqtSignal(int, int)  # Signal to update progress
    end_signal = pyqtSignal(float)  # Signal emitted when processing is complete

    def __init__(self, data, spectrum_plot):
        """
        Initializes the ShiftCalculator with spectral data and a spectrum plot object.

        Parameters:
        data : object
            The dataset containing EELS spectra.
        spectrum_plot : object
            The plot widget for spectrum visualization.
        """
        super().__init__()
        self.data = data
        self.spectrum_plot = spectrum_plot
        self.intersects = []

    def run(self):
        """
        Executes the shift calculation process.
        Iterates over the spectra, fits edges and backgrounds, and computes intersections.
        Emits progress updates and signals completion with elapsed time.
        """
        tic = time.time()
        spectrum_res = self.data.get_spectral_resolution()
        spectrum_offset = self.data.get_offset()
        counter = 0
        data_length = len(self.data.eels_highloss.data) * len(self.data.eels_highloss.data[0])

        for i, line in enumerate(self.data.eels_highloss.data):
            line_intersects = []
            for j, spectrum in enumerate(line):
                counter += 1
                x_values = np.array(range(len(spectrum))) * spectrum_res + spectrum_offset
                self.spectrum_plot.update_plot(x_values, spectrum, (i, j), update_roi=False)
                self.spectrum_plot.fit_edge(plot=False)

                if j == 50 and i == 50:
                    print("stop")

                intersect = self.spectrum_plot.fit_background(plot=False)
                if self.significant_signal():
                    line_intersects.append((intersect - 825.8) * 100)
                else:
                    line_intersects.append(0)

                self.progress_signal.emit(counter, data_length)  # Emit progress update
                time.sleep(0.001)

            self.intersects.append(line_intersects)

        toc = time.time()
        self.intersects = np.array(self.intersects)
        self.end_signal.emit(toc - tic)

    def significant_signal(self):
        """
        Determines whether the detected signal is significant compared to background noise.

        Returns:
        bool : True if the signal is significant, False otherwise.
        """
        bg_noise = self.spectrum_plot.bg_fit_noise_level
        signal = self.spectrum_plot.egde_signal_strength
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


class SpectrumPlot(QWidget):
    """
    A widget for plotting and analyzing spectral data.

    This class provides functionality to display spectra, fit background and edge regions,
    determine intersection points, and save edge data.
    """

    def __init__(self):
        """
        Initializes the SpectrumPlot widget.

        Creates the main layout, plot widget, controls for background fitting,
        intersection calculation, and saving edge data.
        """
        super().__init__()

        # Main layout
        main_layout = QVBoxLayout()

        # Pixel position label
        self.label = QLabel("", self)
        main_layout.addWidget(self.label)

        # Plot widget
        self.plot_widget = pg.PlotWidget(background='w')
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

        # Intersection control section
        controls_layout.addWidget(QLabel("Intersection Energy:"), 1, 0)
        self.intersection_value = QLineEdit()
        self.intersection_value.setReadOnly(True)
        controls_layout.addWidget(self.intersection_value, 1, 1)

        self.edge_name_input = QLineEdit()
        self.edge_name_input.setPlaceholderText("Edge name")
        controls_layout.addWidget(self.edge_name_input, 1, 2)

        self.save_edge_button = QPushButton("Save Edge")
        self.save_edge_button.clicked.connect(self.save_edge)
        controls_layout.addWidget(self.save_edge_button, 1, 3)

        self.calculate_shift_btn = QPushButton("Calculate Shift!")
        self.calculate_shift_btn.clicked.connect(self.calculate_shift)
        controls_layout.addWidget(self.calculate_shift_btn, 2, 1)

        # Saved edges list
        self.saved_edges_list = QListWidget()
        controls_layout.addWidget(QLabel("Saved Edges:"), 3, 0)
        controls_layout.addWidget(self.saved_edges_list, 3, 0, 1, 4)

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
        self.intersection_point = None
        self.spectrum_curve = None
        self.saved_edges = {}

    def update_plot(self, xaxis, energies, position, update_roi=True):
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
        if self.spectrum_curve is not None:
            self.spectrum_curve.setData(xaxis, energies)
        else:
            self.spectrum_curve = self.plot_widget.plot(xaxis, energies, pen='k')
        self.label.setText(f"Pixel: {position}")

        if update_roi:
            self.update_roi(xaxis, energies)

    def update_roi(self, xaxis, energies):
        """
        Updates or creates regions of interest (ROIs) for background and edge fitting.

        Args:
            xaxis (array-like): X-axis values.
            energies (array-like): Corresponding spectral intensities.
        """
        # Create ROIs if they don't exist
        if self.bg_roi is None:
            # Create background ROI around 20% of the x-axis range
            x_range = xaxis[-1] - xaxis[0]
            roi_width = x_range * 0.2
            start_x = xaxis[0] + x_range * 0.1

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
            roi_width = x_range * 0.2
            start_x = xaxis[0] + x_range * 0.6

            self.edge_roi = pg.LinearRegionItem(
                values=[start_x, start_x + roi_width],
                brush=pg.mkBrush(255, 128, 128, 50),
                pen=pg.mkPen('r', width=2),
                movable=True
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
        self.fit_edge()

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

    def calculate_shift(self, edge):
        print("calculating shift...")



if __name__ == '__main__':
    mkQApp("Main")
    main_window = MainWindow()
    #main_window.load_data("D:\\OneDrive - Uniwersytet Jagielloński\\Studia\\python\\EELSpy\\pythonProject1\\tests\\reference\\STEM SI_eels_O_Mn_La_do_Ca.dm4")
    main_window.show()
    pg.exec()
