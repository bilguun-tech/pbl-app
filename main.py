import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
)
from PyQt5.QtGui import QImage, QPixmap

import matplotlib.pyplot as plt
from analysis_methods import (
    method1,
    method2,
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

        self.method_label = QLabel("Select Analysis Method:")
        self.method_buttons = []
        for i, method in enumerate([method1, method2, method3]):
            button = QPushButton(f"Method {i + 1}", self)
            button.clicked.connect(lambda _, m=method: self.analyze_and_display(m))
            self.method_buttons.append(button)

        # Widgets for displaying graphs
        self.graph1_label = QLabel("Graph 1")
        self.graph2_label = QLabel("Graph 2")

        # Layout setup
        layout = QHBoxLayout(self)

        # Left side layout for graphs
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.graph1_label)
        left_layout.addWidget(self.graph2_label)
        layout.addLayout(left_layout)

        # Right side layout for file import and method selection
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.file_label)
        right_layout.addWidget(self.import_button)
        right_layout.addWidget(self.method_label)
        right_layout.addStretch(1)
        for button in self.method_buttons:
            right_layout.addWidget(button)
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

    def analyze_and_display(self, analysis_method):
        data = []  # Replace with your data or data loading code
        plot = analysis_method(data)

        if plot:
            if isinstance(plot, plt.Figure):
                self.show_matplotlib_plot(plot)
            else:
                print("Invalid plot type")

    def show_matplotlib_plot(self, plot):
        # Clear existing plots
        self.graph1_label.clear()
        self.graph2_label.clear()

        # Display the plots in the QLabel widgets
        self.graph1_label.setPixmap(self.plot_to_pixmap(plot))
        self.graph2_label.setPixmap(self.plot_to_pixmap(plot))

    def plot_to_pixmap(self, plot):
        buf = plot.canvas.buffer_rgba()
        qimage = QImage(
            buf,
            plot.canvas.get_width_height()[0],
            plot.canvas.get_width_height()[1],
            QImage.Format_RGBA8888,
        )
        pixmap = QPixmap.fromImage(qimage)
        return pixmap


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
