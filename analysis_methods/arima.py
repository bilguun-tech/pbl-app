import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

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
from analysis_methods.just_plot import just_plot
from analysis_methods.additive_method import additive_method
from analysis_methods.arima import arima
from analysis_methods.ETS_model import ETS_model


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
        # Get the selected analysis method and column name from the combobox
        selected_method = self.method_combobox.currentText()
        if selected_method == "Additive method":
            selected_period = self.period_combobox.currentText()

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
            # Perform analysis directly in the main thread
            plot = analysis_method(self.data, selected_column)
            self.analysis_complete(plot)

    def analysis_complete(self, plot):
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


def arima(df, column_name):
    df["date"] = df.iloc[:, 0]
    df = df[[column_name, "date"]]
    df.set_index("date", inplace=True)

    # 可視化
    # df.plot()
    # plt.show()

    # ADF検定（原系列）定常過程かどうかを検定する
    dftest = adfuller(df)
    # print('ADF Statistic: %f' % dftest[0])
    print("p-value: %f" % dftest[1])
    # print('Critical values :')
    # for k, v in dftest[4].items():
    #    print('\t', k, v)

    # プログラムのURL:https://toukei-lab.com/python_stock
    if dftest[1] <= 0.05:
        print("p-value <= 0.05")
        print("データは定常過程です")
    else:
        print("データは定常過程ではありません")
        # ARIMAモデル データ準備
        train_data, test_data = df[0 : int(len(df) * 0.7)], df[int(len(df) * 0.7) :]
        train_data = train_data[column_name].values
        test_data = test_data[column_name].values

        # ARIMAモデル実装
        # train_data = df["close"].values
        model = ARIMA(train_data, order=(6, 1, 0))
        model_fit = model.fit()
        print(model_fit.summary())

        # ARIMAモデル 予測
        history = [x for x in train_data]
        model_predictions = []

        for time_point in range(len(test_data)):
            # ARIMAモデル 実装
            model = ARIMA(history, order=(6, 1, 0))
            model_fit = model.fit()
            # 予測データの出力
            output = model_fit.forecast()
            yhat = output[0]
            model_predictions.append(yhat)
            # トレーニングデータの取り込み
            true_test_value = test_data[time_point]
            history.append(true_test_value)
        # サブプロットの設定
        fig, axs = plt.subplots()

        # 実測値の描画
        axs.plot(test_data, color="Red", label="Measured")
        # 予測値の描画
        axs.plot(model_predictions, color="Blue", label="Prediction")
        axs.set_title(" ARIMA model", fontname="MS Gothic")
        axs.set_xlabel("Date", fontname="MS Gothic")
        axs.set_ylabel("Amazon stock price", fontname="MS Gothic")
        axs.legend(prop={"family": "MS Gothic"})

        # axs.set_title('Prediction', fontname="MS Gothic")

        return fig
