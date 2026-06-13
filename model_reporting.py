import csv
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def prepare_output_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    return models_dir, metrics_dir


def save_model_with_timestamp(model, models_dir, prefix="banana_model"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model_path = os.path.join(models_dir, f"{prefix}_{timestamp}.keras")
    model.save(model_path)
    return model_path


def save_dataset_preview(train_ds, class_names, output_dir):
    plt.figure(figsize=(10, 10))

    for images, labels in train_ds.take(1):
        count = min(12, images.shape[0])

        for i in range(count):
            plt.subplot(3, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")

    plt.savefig(os.path.join(output_dir, "dataset_preview.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_training_history_plot(history, epochs, output_dir):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_and_save_reports(model, test_ds, test_file_paths, class_names, metrics_dir):
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy:.6f}")

    y_true = collect_labels(test_ds)
    y_logits = model.predict(test_ds, verbose=2)
    y_prob = tf.nn.softmax(y_logits, axis=1).numpy()
    y_pred = np.argmax(y_logits, axis=1)

    confusion_matrix = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=len(class_names),
    ).numpy()

    class_metrics = calculate_class_metrics(confusion_matrix, class_names, y_true, y_prob)
    save_metrics_table(class_metrics, test_loss, test_accuracy, metrics_dir)
    save_predictions_table(test_file_paths, y_true, y_pred, y_prob, class_names, metrics_dir)
    save_mistake_previews(test_file_paths, y_true, y_pred, y_prob, class_names, metrics_dir)
    save_confusion_matrix_plot(confusion_matrix, class_names, metrics_dir)
    save_metrics_plots(class_metrics, metrics_dir)

    return test_loss, test_accuracy


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def safe_filename(name):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def collect_labels(dataset):
    labels_batches = []
    for _, labels in dataset:
        labels_batches.append(labels.numpy())
    return np.concatenate(labels_batches)


def calculate_curve_metrics(y_true_binary, y_score):
    positives = np.sum(y_true_binary == 1)
    negatives = np.sum(y_true_binary == 0)

    if positives == 0 or negatives == 0:
        return {
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 0.0]),
            "precision_curve": np.array([0.0, 0.0]),
            "recall_curve": np.array([0.0, 1.0]),
        }

    sorted_indices = np.argsort(y_score)[::-1]
    sorted_true = y_true_binary[sorted_indices]

    true_positives = np.cumsum(sorted_true)
    false_positives = np.cumsum(1 - sorted_true)
    ranks = np.arange(1, len(sorted_true) + 1)

    tpr = np.concatenate(([0.0], true_positives / positives))
    fpr = np.concatenate(([0.0], false_positives / negatives))
    precision_curve = np.concatenate(([1.0], true_positives / ranks))
    recall_curve = np.concatenate(([0.0], true_positives / positives))

    return {
        "roc_auc": float(np.trapz(tpr, fpr)),
        "pr_auc": float(np.trapz(precision_curve, recall_curve)),
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }


def calculate_class_metrics(confusion_matrix, class_names, y_true, y_prob):
    total = np.sum(confusion_matrix)
    metrics = []

    for class_index, class_name in enumerate(class_names):
        tp = int(confusion_matrix[class_index, class_index])
        fp = int(np.sum(confusion_matrix[:, class_index]) - tp)
        fn = int(np.sum(confusion_matrix[class_index, :]) - tp)
        tn = int(total - tp - fp - fn)

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        npv = safe_divide(tn, tn + fn)
        f1_score = safe_divide(2 * precision * recall, precision + recall)

        curve_metrics = calculate_curve_metrics(
            (y_true == class_index).astype(int),
            y_prob[:, class_index],
        )

        metrics.append({
            "class": class_name,
            "support": int(tp + fn),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": safe_divide(tp + tn, total),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "npv": npv,
            "f1_score": f1_score,
            "false_positive_rate": safe_divide(fp, fp + tn),
            "false_negative_rate": safe_divide(fn, fn + tp),
            "roc_auc": curve_metrics["roc_auc"],
            "pr_auc": curve_metrics["pr_auc"],
            "fpr": curve_metrics["fpr"],
            "tpr": curve_metrics["tpr"],
            "precision_curve": curve_metrics["precision_curve"],
            "recall_curve": curve_metrics["recall_curve"],
        })

    return metrics


def save_metrics_table(metrics, test_loss, test_accuracy, metrics_dir):
    fieldnames = [
        "class",
        "support",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "npv",
        "f1_score",
        "false_positive_rate",
        "false_negative_rate",
        "roc_auc",
        "pr_auc",
    ]

    csv_path = os.path.join(metrics_dir, "class_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({field: row[field] for field in fieldnames})

    txt_path = os.path.join(metrics_dir, "class_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"Test loss: {test_loss:.6f}\n")
        txt_file.write(f"Test accuracy: {test_accuracy:.6f}\n\n")
        txt_file.write(
            "class | support | tp | fp | tn | fn | accuracy | precision | recall | "
            "specificity | npv | f1_score | false_positive_rate | false_negative_rate | "
            "roc_auc | pr_auc\n"
        )
        txt_file.write("-" * 165 + "\n")
        for row in metrics:
            txt_file.write(
                f"{row['class']} | {row['support']} | {row['tp']} | {row['fp']} | "
                f"{row['tn']} | {row['fn']} | {row['accuracy']:.4f} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['specificity']:.4f} | {row['npv']:.4f} | "
                f"{row['f1_score']:.4f} | {row['false_positive_rate']:.4f} | "
                f"{row['false_negative_rate']:.4f} | {row['roc_auc']:.4f} | "
                f"{row['pr_auc']:.4f}\n"
            )


def save_predictions_table(file_paths, y_true, y_pred, y_prob, class_names, metrics_dir):
    fieldnames = [
        "file_path",
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
    ]
    fieldnames += [f"prob_{safe_filename(class_name)}" for class_name in class_names]

    csv_path = os.path.join(metrics_dir, "test_predictions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, file_path in enumerate(file_paths):
            true_index = int(y_true[index])
            predicted_index = int(y_pred[index])
            row = {
                "file_path": file_path,
                "true_class": class_names[true_index],
                "predicted_class": class_names[predicted_index],
                "confidence": float(y_prob[index, predicted_index]),
                "correct": true_index == predicted_index,
            }

            for class_index, class_name in enumerate(class_names):
                row[f"prob_{safe_filename(class_name)}"] = float(y_prob[index, class_index])

            writer.writerow(row)


def save_mistake_previews(file_paths, y_true, y_pred, y_prob, class_names, metrics_dir):
    mistakes_dir = os.path.join(metrics_dir, "mistakes")
    os.makedirs(mistakes_dir, exist_ok=True)

    grouped_mistakes = {}
    for index, file_path in enumerate(file_paths):
        true_index = int(y_true[index])
        predicted_index = int(y_pred[index])

        if true_index == predicted_index:
            continue

        key = (true_index, predicted_index)
        grouped_mistakes.setdefault(key, []).append({
            "file_path": file_path,
            "confidence": float(y_prob[index, predicted_index]),
        })

    save_mistakes_summary(grouped_mistakes, class_names, mistakes_dir)

    for (true_index, predicted_index), mistakes in grouped_mistakes.items():
        true_class = class_names[true_index]
        predicted_class = class_names[predicted_index]
        group_name = f"{safe_filename(true_class)}_as_{safe_filename(predicted_class)}"
        group_dir = os.path.join(mistakes_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        selected_mistakes = mistakes[:40]
        for page_index, start in enumerate(range(0, len(selected_mistakes), 8), start=1):
            save_mistake_preview_page(
                selected_mistakes[start:start + 8],
                true_class,
                predicted_class,
                group_dir,
                page_index,
            )


def save_mistakes_summary(grouped_mistakes, class_names, mistakes_dir):
    summary_path = os.path.join(mistakes_dir, "mistakes_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
        fieldnames = [
            "true_class",
            "predicted_class",
            "total_mistakes",
            "shown_in_previews",
            "folder",
        ]
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()

        for (true_index, predicted_index), mistakes in grouped_mistakes.items():
            true_class = class_names[true_index]
            predicted_class = class_names[predicted_index]
            folder = f"{safe_filename(true_class)}_as_{safe_filename(predicted_class)}"
            writer.writerow({
                "true_class": true_class,
                "predicted_class": predicted_class,
                "total_mistakes": len(mistakes),
                "shown_in_previews": min(len(mistakes), 40),
                "folder": folder,
            })


def save_mistake_preview_page(mistakes, true_class, predicted_class, group_dir, page_index):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for axis in axes:
        axis.axis("off")

    for axis, mistake in zip(axes, mistakes):
        image = tf.keras.utils.load_img(mistake["file_path"])
        axis.imshow(image)
        axis.set_title(
            f"{true_class} -> {predicted_class}\n"
            f"confidence: {mistake['confidence']:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    fig.suptitle(f"{true_class} oznaczony jako {predicted_class}", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        os.path.join(group_dir, f"preview_{page_index:02d}.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_confusion_matrix_plot(confusion_matrix, class_names, metrics_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Prawdziwa klasa",
        xlabel="Przewidziana klasa",
        title="Macierz pomylek",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = confusion_matrix.max() / 2 if confusion_matrix.size else 0
    for row in range(confusion_matrix.shape[0]):
        for column in range(confusion_matrix.shape[1]):
            value = confusion_matrix[row, column]
            color = "white" if value > threshold else "black"
            ax.text(column, row, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(os.path.join(metrics_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        confusion_matrix,
        row_sums,
        out=np.zeros_like(confusion_matrix, dtype=float),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(normalized_matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Prawdziwa klasa",
        xlabel="Przewidziana klasa",
        title="Znormalizowana macierz pomylek",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for row in range(normalized_matrix.shape[0]):
        for column in range(normalized_matrix.shape[1]):
            value = normalized_matrix[row, column]
            color = "white" if value > 0.5 else "black"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(
        os.path.join(metrics_dir, "confusion_matrix_normalized.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_metrics_plots(metrics, metrics_dir):
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "npv",
        "f1_score",
        "false_positive_rate",
        "false_negative_rate",
        "roc_auc",
        "pr_auc",
    ]

    class_labels = [row["class"] for row in metrics]
    x_positions = np.arange(len(class_labels))

    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.1
    for index, metric_name in enumerate(metric_names):
        values = [row[metric_name] for row in metrics]
        offset = (index - (len(metric_names) - 1) / 2) * width
        ax.bar(x_positions + offset, values, width, label=metric_name)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Wartosc")
    ax.set_title("Metryki dla klas")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(metrics_dir, "class_metrics_overview.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    for row in metrics:
        values = [row[metric_name] for metric_name in metric_names]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metric_names, values, color="steelblue")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Wartosc")
        ax.set_title(f"Metryki klasy: {row['class']} (support={row['support']})")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        for index, value in enumerate(values):
            ax.text(index, value + 0.02, f"{value:.2f}", ha="center", va="bottom")

        fig.tight_layout()
        filename = f"class_metrics_{safe_filename(row['class'])}.png"
        fig.savefig(os.path.join(metrics_dir, filename), dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(row["fpr"], row["tpr"], label=f"AUC = {row['roc_auc']:.3f}")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[0].set_title(f"ROC: {row['class']}")
        axes[0].legend(loc="lower right")

        axes[1].plot(
            row["recall_curve"],
            row["precision_curve"],
            label=f"AUC = {row['pr_auc']:.3f}",
        )
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(f"Precision-recall: {row['class']}")
        axes[1].legend(loc="lower left")

        fig.tight_layout()
        filename = f"class_curves_{safe_filename(row['class'])}.png"
        fig.savefig(os.path.join(metrics_dir, filename), dpi=200, bbox_inches="tight")
        plt.close(fig)
