import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Tuple
from operator import itemgetter
from collections import Counter

def plot_list_of_curves(
    list_of_x_vals,
    list_of_y_vals,
    list_of_colors,
    list_of_curve_labels,
    x_label=None,
    y_label=None,
    title=None
):
    plt.figure(figsize=(11, 7))
    for i, x_vals in enumerate(list_of_x_vals):
        plt.plot(
            x_vals,
            list_of_y_vals[i],
            list_of_colors[i],
            label=list_of_curve_labels[i]
        )
    plt.axis((
        min(map(min, list_of_x_vals)),
        max(map(max, list_of_x_vals)),
        min(map(min, list_of_y_vals)),
        max(map(max, list_of_y_vals))
    ))
    if x_label is not None:
        plt.xlabel(x_label, fontsize=20)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=20)
    if title is not None:
        plt.title(title, fontsize=25)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.show()

def plot_single_trace_all_processes(
    process1_trace: np.ndarray,
    process2_trace: np.ndarray,
    process3_trace: np.ndarray
) -> None:


    traces_len = len(process1_trace)

    plot_list_of_curves(
        [range(traces_len)] * 3,
        [process1_trace, process2_trace, process3_trace],
        ["r", "b", "g"],
        [
            r"Process 1 ($\alpha_1=0.25$)",
            r"Process 2 ($\alpha_2=0.75$)",
            r"Process 3 ($\alpha_3=1.0$)"
        ],
        "Time Steps",
        "Stock Price",
        "Single-Trace Simulation for Each Process"
    )


def get_terminal_histogram(
    price_traces: np.ndarray
) -> Tuple[Sequence[int], Sequence[int]]:
    pairs = sorted(
        list(Counter(price_traces[:, -1]).items()),
        key=itemgetter(0)
    )
    return [x for x, _ in pairs], [y for _, y in pairs]


def plot_distribution_at_time_all_processes(
    process1_traces: np.ndarray,
    process2_traces: np.ndarray,
    process3_traces: np.ndarray
) -> None:


    num_traces = len(process1_traces)
    time_steps = len(process1_traces[0]) - 1

    x1, y1 = get_terminal_histogram(process1_traces)
    x2, y2 = get_terminal_histogram(process2_traces)
    x3, y3 = get_terminal_histogram(process3_traces)

    plot_list_of_curves(
        [x1, x2, x3],
        [y1, y2, y3],
        ["r", "b", "g"],
        [
            r"Process 1 ($\alpha_1=0.25$)",
            r"Process 2 ($\alpha_2=0.75$)",
            r"Process 3 ($\alpha_3=1.0$)"
        ],
        "Terminal Stock Price",
        "Counts",
        f"Terminal Price Counts (T={time_steps:d}, Traces={num_traces:d})"
    )

if __name__ == '__main__':
    x = np.arange(1, 100)
    y = [0.1 * x + 1.0, 0.001 * (x - 50) ** 2, np.log(x)]
    colors = ["r", "b", "g"]
    labels = ["Linear", "Quadratic", "Log"]
    plot_list_of_curves(
        [x, x, x],
        y,
        colors,
        labels,
        "X-Axis",
        "Y-Axis",
        "Test Plot"
    )
