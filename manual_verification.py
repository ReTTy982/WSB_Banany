import argparse
import os

from tensorflow import keras

from model_loading import load_compatible_model
from model_reporting import evaluate_and_save_reports


DEFAULT_DATASET_DIR = "dataset/test"
DEFAULT_OUTPUT_DIR = "outputs/manual_veryfication"
DEFAULT_IMAGE_SIZE = (416, 416)
DEFAULT_BATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wczytaj zapisany model i wykonaj weryfikacje na wybranym datasecie."
    )
    parser.add_argument("model_path", help="Sciezka do zapisanego modelu .keras")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_DIR,
        help=f"Folder datasetu do weryfikacji. Domyslnie: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder na wyniki. Domyslnie: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size. Domyslnie: {DEFAULT_BATCH_SIZE}",
    )
    return parser.parse_args()


def infer_image_size(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) >= 4 and input_shape[1] and input_shape[2]:
        return int(input_shape[1]), int(input_shape[2])

    return DEFAULT_IMAGE_SIZE


def load_dataset(dataset_dir, image_size, batch_size):
    dataset = keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataset, dataset.file_paths, dataset.class_names


def save_summary(output_dir, model_path, dataset_dir, class_names, test_loss, test_accuracy):
    summary_path = os.path.join(output_dir, "verification_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"Model: {model_path}\n")
        summary_file.write(f"Dataset: {dataset_dir}\n")
        summary_file.write(f"Classes: {', '.join(class_names)}\n")
        summary_file.write(f"Test loss: {test_loss:.6f}\n")
        summary_file.write(f"Test accuracy: {test_accuracy:.6f}\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_compatible_model(args.model_path, compile=True)
    image_size = infer_image_size(model)
    dataset, file_paths, class_names = load_dataset(
        args.dataset,
        image_size=image_size,
        batch_size=args.batch_size,
    )
    dataset = dataset.prefetch(1)

    test_loss, test_accuracy = evaluate_and_save_reports(
        model,
        dataset,
        file_paths,
        class_names,
        args.output_dir,
    )
    save_summary(
        args.output_dir,
        args.model_path,
        args.dataset,
        class_names,
        test_loss,
        test_accuracy,
    )
    print(f"Wyniki zapisane do: {args.output_dir}")


if __name__ == "__main__":
    main()
