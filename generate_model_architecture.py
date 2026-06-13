import argparse
import os

from tensorflow import keras

from model_loading import load_compatible_model


DEFAULT_OUTPUT_DIR = "outputs/model_architecture"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wygeneruj diagram architektury zapisanego modelu Keras."
    )
    parser.add_argument("model_path", help="Sciezka do modelu .keras")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder na wyniki. Domyslnie: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--horizontal",
        action="store_true",
        help="Wygeneruj diagram poziomo zamiast pionowo.",
    )
    return parser.parse_args()


def save_model_summary(model, output_path):
    summary_lines = []
    model.summary(print_fn=summary_lines.append)

    with open(output_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(summary_lines))
        summary_file.write("\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_compatible_model(args.model_path, compile=False)
    diagram_path = os.path.join(args.output_dir, "model_architecture.png")
    summary_path = os.path.join(args.output_dir, "model_summary.txt")

    save_model_summary(model, summary_path)

    try:
        keras.utils.plot_model(
            model,
            to_file=diagram_path,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="LR" if args.horizontal else "TB",
            expand_nested=True,
            dpi=200,
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Nie mozna wygenerowac diagramu. Zainstaluj pydot oraz Graphviz."
        ) from error

    print(f"Diagram zapisany do: {diagram_path}")
    print(f"Podsumowanie zapisane do: {summary_path}")


if __name__ == "__main__":
    main()
