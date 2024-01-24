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
)
from PyQt5.QtGui import QImage, QPixmap

import matplotlib.pyplot as plt
from analysis_methods.method1 import (
    method1,
)

from analysis_methods.method2 import (
    method2,
)

from analysis_methods.method3 import (
    method3,
)


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
        self.method_combobox.addItems(["Method 1", "Method 2", "Method 3"])

        # Button to start analysis
        self.start_analysis_button = QPushButton("Start Analysis", self)
        self.start_analysis_button.clicked.connect(self.start_analysis)

        # Widgets for displaying graphs
        self.graph1_label = QLabel("Graph 1")

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
        right_layout.addWidget(self.start_analysis_button)
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

            except Exception as e:
                # Handle any potential errors during reading the CSV file
                print(f"Error reading CSV file: {e}")

    def start_analysis(self):
        # Get the selected analysis method from the combobox
        selected_method = self.method_combobox.currentText()

        # Map the selected method to the corresponding function
        method_mapping = {"Method 1": method1, "Method 2": method2, "Method 3": method3}

        analysis_method = method_mapping.get(selected_method)

        if analysis_method:
            plot = analysis_method(self.data)

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
