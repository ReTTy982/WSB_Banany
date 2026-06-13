import argparse
import os
import shutil
import subprocess
import sys
import warnings


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_NEURAL_NET_DIR = os.path.join(PROJECT_DIR, "third_party", "PlotNeuralNet")
DEFAULT_OUTPUT_DIR = os.path.join("outputs", "plotneuralnet_architecture")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Wygeneruj diagram architektury CNN za pomoca PlotNeuralNet."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder na wyniki. Domyslnie: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def to_feature_box(
    name,
    offset,
    to,
    width,
    height,
    depth,
    fill,
    opacity=1.0,
):
    return rf"""
\pic[shift={{{offset}}}] at {to}
{{Box={{
    name={name},
    caption=,
    xlabel={{{{"", }}}},
    zlabel=,
    fill={fill},
    opacity={opacity},
    height={height},
    width={width},
    depth={depth}
    }}
}};
"""


def to_layer_name(name, title):
    return rf"""
\node[
    anchor=south west,
    align=left,
    font=\scriptsize\bfseries,
    text width=2.6cm
] at ([xshift=0.2cm,yshift=0.25cm]{name}-northeast)
{{{title}}};
"""


def to_shape_label(name, shape):
    return rf"""
\node[
    anchor=north,
    align=center,
    font=\scriptsize,
    text width=2.8cm
] at ([yshift=-0.85cm]{name}-south)
{{{shape}}};
"""


def to_reference_colors():
    return r"""
\def\ConvColor{rgb:yellow,5;red,1.4;white,8}
\def\PoolColor{rgb:red,3;white,7}
\def\DropoutColor{rgb:blue,2;red,2;white,7}
\def\FcColor{rgb:blue,2;white,7}
\def\SoftmaxColor{rgb:green,3;white,7}
"""


def to_legend():
    return r"""
\node[
    draw=orange!75!black,
    fill=\ConvColor,
    minimum width=0.55cm,
    minimum height=0.32cm
] (legend-conv) at (12.5,-6.6) {};
\node[anchor=west,font=\scriptsize] at ([xshift=0.25cm]legend-conv.east)
{convolution + ReLU};

\node[
    draw=red!65!black,
    fill=\PoolColor,
    minimum width=0.55cm,
    minimum height=0.32cm
] (legend-pool) at (12.5,-7.15) {};
\node[anchor=west,font=\scriptsize] at ([xshift=0.25cm]legend-pool.east)
{max pooling};

\node[
    draw=violet!65!black,
    fill=\DropoutColor,
    minimum width=0.55cm,
    minimum height=0.32cm
] (legend-dropout) at (12.5,-7.7) {};
\node[anchor=west,font=\scriptsize] at ([xshift=0.25cm]legend-dropout.east)
{dropout};

\node[
    draw=blue!60!black,
    fill=\FcColor,
    minimum width=0.55cm,
    minimum height=0.32cm
] (legend-dense) at (12.5,-8.25) {};
\node[anchor=west,font=\scriptsize] at ([xshift=0.25cm]legend-dense.east)
{fully connected + ReLU};

\node[
    draw=green!60!black,
    fill=\SoftmaxColor,
    minimum width=0.55cm,
    minimum height=0.32cm
] (legend-softmax) at (12.5,-8.8) {};
\node[anchor=west,font=\scriptsize] at ([xshift=0.25cm]legend-softmax.east)
{softmax};
"""


def build_architecture(layer_helpers_path):
    sys.path.insert(0, PLOT_NEURAL_NET_DIR)
    warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pycore\..*")
    from pycore.tikzeng import (
        to_begin,
        to_cor,
        to_end,
        to_head,
    )

    return [
        to_head(layer_helpers_path),
        to_cor(),
        to_reference_colors(),
        to_begin(),
        to_feature_box(
            "conv1",
            offset="(0,0,0)",
            to="(0,0,0)",
            width=2,
            height=40,
            depth=40,
            fill=r"\ConvColor",
        ),
        to_layer_name("conv1", "conv1"),
        to_shape_label("conv1", "416 x 416 x 16"),
        to_feature_box(
            "pool1",
            offset="(0.35,0,0)",
            to="(conv1-east)",
            width=0.65,
            height=32,
            depth=32,
            fill=r"\PoolColor",
            opacity=0.75,
        ),
        to_feature_box(
            "conv2",
            offset="(1.15,0,0)",
            to="(pool1-east)",
            width=2.5,
            height=32,
            depth=32,
            fill=r"\ConvColor",
        ),
        to_layer_name("conv2", "conv2"),
        to_shape_label("conv2", "208 x 208 x 32"),
        to_feature_box(
            "pool2",
            offset="(0.35,0,0)",
            to="(conv2-east)",
            width=0.65,
            height=24,
            depth=24,
            fill=r"\PoolColor",
            opacity=0.75,
        ),
        to_feature_box(
            "conv3",
            offset="(1.15,0,0)",
            to="(pool2-east)",
            width=3,
            height=24,
            depth=24,
            fill=r"\ConvColor",
        ),
        to_layer_name("conv3", "conv3"),
        to_shape_label("conv3", "104 x 104 x 64"),
        to_feature_box(
            "pool3",
            offset="(0.35,0,0)",
            to="(conv3-east)",
            width=0.65,
            height=17,
            depth=17,
            fill=r"\PoolColor",
            opacity=0.75,
        ),
        to_feature_box(
            "dropout",
            offset="(0.45,0,0)",
            to="(pool3-east)",
            width=0.8,
            height=17,
            depth=17,
            fill=r"\DropoutColor",
            opacity=0.8,
        ),
        to_layer_name("dropout", "Dropout"),
        to_shape_label("dropout", "52 x 52 x 64"),
        to_feature_box(
            "gap",
            offset="(1.4,0,0)",
            to="(dropout-east)",
            width=0.65,
            height=8,
            depth=8,
            fill=r"\ConvColor",
        ),
        to_layer_name("gap", "GlobalAvgPool"),
        to_shape_label("gap", "1 x 1 x 64"),
        to_feature_box(
            "dense",
            offset="(2.6,0,0)",
            to="(gap-east)",
            width=7,
            height=1.5,
            depth=1.5,
            fill=r"\FcColor",
        ),
        to_layer_name("dense", "Dense"),
        to_shape_label("dense", "1 x 1 x 128"),
        to_feature_box(
            "softmax",
            offset="(0.35,0,0)",
            to="(dense-east)",
            width=3,
            height=1.5,
            depth=1.5,
            fill=r"\SoftmaxColor",
        ),
        to_layer_name("softmax", "Softmax"),
        to_shape_label("softmax", "1 x 1 x 4"),
        to_legend(),
        to_end(),
    ]


def write_tex(architecture, tex_path):
    with open(tex_path, "w", encoding="utf-8") as tex_file:
        tex_file.writelines(architecture)


def compile_pdf(output_dir, tex_filename):
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        return None

    subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_filename],
        cwd=output_dir,
        check=True,
    )

    output_stem = os.path.splitext(tex_filename)[0]
    for extension in (".aux", ".log"):
        helper_path = os.path.join(output_dir, output_stem + extension)
        if os.path.exists(helper_path):
            os.remove(helper_path)

    return os.path.join(output_dir, os.path.splitext(tex_filename)[0] + ".pdf")


def convert_pdf_to_png(pdf_path):
    png_path = os.path.splitext(pdf_path)[0] + ".png"
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        subprocess.run(
            [pdftoppm, "-png", "-singlefile", "-r", "200", pdf_path, png_path[:-4]],
            check=True,
        )
        return png_path

    converter = shutil.which("magick") or shutil.which("convert")
    if converter:
        command = [converter]
        if os.path.basename(converter) == "magick":
            command.append("convert")
        command.extend(["-density", "200", pdf_path, png_path])
        subprocess.run(command, check=True)
        return png_path

    return None


def main():
    args = parse_args()

    if not os.path.isdir(PLOT_NEURAL_NET_DIR):
        raise FileNotFoundError(
            "Brak third_party/PlotNeuralNet. Pobierz repozytorium PlotNeuralNet."
        )

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    layer_helpers_path = os.path.relpath(PLOT_NEURAL_NET_DIR, output_dir)
    tex_filename = "banana_cnn_plotneuralnet.tex"
    tex_path = os.path.join(output_dir, tex_filename)
    architecture = build_architecture(layer_helpers_path)
    write_tex(architecture, tex_path)
    print(f"Plik TikZ zapisany do: {tex_path}")

    pdf_path = compile_pdf(output_dir, tex_filename)
    if not pdf_path:
        print("PDF nie zostal wygenerowany: zainstaluj pdflatex (TeX Live).")
        return

    print(f"PDF zapisany do: {pdf_path}")
    png_path = convert_pdf_to_png(pdf_path)
    if png_path:
        print(f"PNG zapisany do: {png_path}")
    else:
        print("PNG nie zostal wygenerowany: zainstaluj pdftoppm lub ImageMagick.")


if __name__ == "__main__":
    main()
