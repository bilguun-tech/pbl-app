import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QProgressBar,  # Added QProgressBar for loading state
)

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

import matplotlib.pyplot as plt
from analysis_methods.just_plot import just_plot
from analysis_methods.additive_method import additive_method
from analysis_methods.arima import arima
from analysis_methods.ETS_model import ETS_model


class AnalysisThread(QThread):
    analysis_complete = pyqtSignal(object)

    def __init__(self, analysis_method, data, selected_column):
        super().__init__()
        self.analysis_method = analysis_method
        self.data = data
        self.selected_column = selected_column

    def run(self):
        plot = self.analysis_method(self.data, self.selected_column)
        self.analysis_complete.emit(plot)


class DataAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Data Analyzer")
        self.setGeometry(100, 100, 800, 400)

        self.data = None
        self.file_label = QLabel("No file selected")
        self.import_button = QPushButton("Import File", self)
        self.import_button.clicked.connect(self.import_file)

        # Analysis method selection using QComboBox
        self.method_label = QLabel("Select Analysis Method:")
        self.method_combobox = QComboBox(self)
        self.method_combobox.addItems(
            ["Just Plot", "Additive method", "Arima method", "ETS model"]
        )

        # Column name selection using QComboBox
        self.column_label = QLabel("Select Column Name:")
        self.column_combobox = QComboBox(self)

        # Button to start analysis
        self.start_analysis_button = QPushButton("Start Analysis", self)
        self.start_analysis_button.clicked.connect(self.start_analysis)

        # Loading indicator
        self.loading_indicator = QProgressBar(self)
        self.loading_indicator.setMinimum(0)
        self.loading_indicator.setMaximum(0)
        self.loading_indicator.setTextVisible(False)
        self.loading_indicator.hide()

        # Widgets for displaying graphs
        self.graph1_label = QLabel("Graph")

        # Layout setup
        layout = QHBoxLayout(self)

        # Left side layout for graphs
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.graph1_label)
        layout.addLayout(left_layout)

        # Right side layout for file import, method selection, and analysis button
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.file_label)
        right_layout.addWidget(self.import_button)
        right_layout.addWidget(self.method_label)
        right_layout.addWidget(self.method_combobox)
        right_layout.addWidget(self.column_label)
        right_layout.addWidget(self.column_combobox)
        right_layout.addWidget(self.start_analysis_button)
        right_layout.addWidget(self.loading_indicator)
        right_layout.addStretch(1)
        layout.addLayout(right_layout)

        self.setLayout(layout)

    def import_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            self.file_label.setText(f"File: {file_path}")
            # Read CSV file using pandas
            try:
                # Store the data as an attribute
                self.data = pd.read_csv(file_path)

                # Populate column_combobox with column names
                self.column_combobox.clear()
                self.column_combobox.addItems(
                    self.data.columns[1:]
                )  # Exclude the first column

            except Exception as e:
                # Handle any potential errors during reading the CSV file
                print(f"Error reading CSV file: {e}")

    def start_analysis(self):
        # Get the selected analysis method from the combobox
        selected_method = self.method_combobox.currentText()
        selected_column = self.column_combobox.currentText()

        # Map the selected method to the corresponding function
        method_mapping = {
            "Just Plot": just_plot,
            "Additive method": additive_method,
            "Arima method": arima,
            "ETS model": ETS_model,
        }

        analysis_method = method_mapping.get(selected_method)

        if analysis_method:
            self.show_loading_state(True)
            self.analysis_thread = AnalysisThread(
                analysis_method, self.data, selected_column
            )
            self.analysis_thread.analysis_complete.connect(self.analysis_complete)
            self.analysis_thread.start()

    def show_loading_state(self, state):
        if state:
            self.loading_indicator.show()
        else:
            self.loading_indicator.hide()

    def analysis_complete(self, plot):
        self.show_loading_state(False)

        if plot:
            if isinstance(plot, plt.Figure):
                self.show_matplotlib_plot(plot)
            else:
                print("Invalid plot type")

    def show_matplotlib_plot(self, plot):
        # Clear existing plots
        self.graph1_label.clear()

        # Display the plots in the QLabel widgets
        self.graph1_label.setPixmap(self.plot_to_pixmap(plot))

    def plot_to_pixmap(self, plot):
        # Create a figure and render it to a pixmap
        figure = plot.figure
        canvas = figure.canvas
        canvas.draw()  # Ensure the figure is drawn

        # Convert the rendered figure to a NumPy array
        width, height = figure.get_size_inches() * figure.get_dpi()
        buf = canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 4)

        # Create a QImage from the NumPy array
        qimage = QImage(img.data, int(width), int(height), QImage.Format_RGBA8888)

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(qimage)

        return pixmap


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
