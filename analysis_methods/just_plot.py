from pyqtgraph import PlotWidget, PlotDataItem, mkPen
import pandas as pd


def just_plot(data, column_name):
    plot_widget = PlotWidget()
    plot_widget.setBackground("w")

    # Filter out rows with empty values in the specified column
    filtered_data = data.dropna(subset=[column_name])

    if "年" in data.columns:  # 年ごとのデータの場合
        # Convert x values to datetime objects
        x = filtered_data.iloc[:, 0]

        # Create PlotDataItem
        y = filtered_data[column_name]
        plot_data = PlotDataItem(x, y, pen=mkPen("r", width=2))
        plot_widget.addItem(plot_data)
    elif "date" in data.columns:  # 1日ごとの場合
        # Convert x values to datetime objects
        try:
            x = pd.to_datetime(filtered_data.iloc[:, 0])
        except ValueError as e:
            print(f"Error converting x values to datetime: {e}")
            return plot_widget

        # Create PlotDataItem
        y = filtered_data[column_name]
        plot_data = PlotDataItem(x, y, pen=mkPen("r", width=2))
        plot_widget.addItem(plot_data)

        # Define a function to update ticks based on zoom level
        def update_ticks(event):
            # Get current x-axis range
            x_range = plot_widget.viewRange()[0]
            x_min, x_max = x_range

            # Calculate the time range of the visible portion
            time_range = x_max - x_min

            print(time_range)
            # Choose tick spacing based on the time range
            if time_range <= 1000000000 * 3600 * 24 * 7:  # 1 week or less
                tick_spacing = 1  # 1 day
                print("1 week or less")
            elif time_range <= 1000000000 * 3600 * 24 * 30:  # 1 month or less
                tick_spacing = 7  # 1 week
                print("1 month or less")
            else:
                tick_spacing = 30  # 1 month
                print("1 month or more")

            # Set ticks with the chosen spacing
            axis = plot_widget.getAxis("bottom")
            axis.setTicks(
                [
                    [
                        (v.timestamp() * 1000000000, v.strftime("%Y-%m-%d"))
                        for i, v in enumerate(x)
                        if i % tick_spacing == 0
                    ]
                ]
            )

        # Connect the update_ticks function to the sigXRangeChanged signal
        plot_widget.sigXRangeChanged.connect(update_ticks)

        # Initial tick setup
        update_ticks(None)

    return plot_widget
