import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = "outputs/manual_veryfication/class_metrics.csv"
OUTPUT_DIR = "outputs/manualnyobrazek"
OUTPUT_FILE = "class_metrics_overview.png"

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1_score", "F1-score"),
]


def load_metrics(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"Plik nie zawiera danych: {csv_path}")

    required_columns = {"class", *(metric_name for metric_name, _ in METRICS)}
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Brak wymaganych kolumn w pliku CSV: {missing}")

    return rows


def create_plot(rows, output_path):
    class_names = [row["class"] for row in rows]
    x_positions = np.arange(len(class_names))
    bar_width = 0.19
    colors = ["#2878B5", "#F28E2B", "#3A923A", "#B55A5A"]

    fig, ax = plt.subplots(figsize=(12, 7))

    for index, ((metric_name, label), color) in enumerate(zip(METRICS, colors)):
        values = [float(row[metric_name]) for row in rows]
        offset = (index - (len(METRICS) - 1) / 2) * bar_width
        bars = ax.bar(
            x_positions + offset,
            values,
            bar_width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0.90, 1.01)
    ax.set_ylabel("Wartość")
    ax.set_title("Metryki klasyfikacji dla poszczególnych klas")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_metrics(INPUT_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    create_plot(rows, output_path)
    print(f"Zapisano wykres: {output_path}")


if __name__ == "__main__":
    main()
