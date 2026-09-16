"""CLI for omle-convert: convert ML models to the OMLE format.

Usage::

    omle-convert model.json      output.omle   # XGBoost or CatBoost JSON
    omle-convert model.ubj       output.omle   # XGBoost UBJSON
    omle-convert model.txt       output.omle   # LightGBM text (.bin too)
    omle-convert model.cbm       output.omle   # CatBoost binary
    omle-convert model.joblib    output.omle   # any pickled estimator
    omle-convert path/to/spark/  output.omle
    omle-convert spark_model.zip output.omle   # .tar.gz and .tgz too

    # Common options
    omle-convert model.json output.omle \\
        --feature-names age,income,score \\
        --target-name label \\
        --class-labels no,yes \\
        --model-name my_clf

    # Spark models (PipelineModel vs single model is auto-detected)
    omle-convert path/to/spark/ output.omle \\
        --feature-names age,income,score

Output format is inferred from the file extension:
  .omle → protobuf binary (default)
  .json            → JSON text
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional


def _parse_list(s: Optional[str]) -> Optional[list[str]]:
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _save(model, output: Path) -> None:
    import omle
    if output.suffix.lower() == ".json":
        omle.save_json(model, output)
    else:
        omle.save(model, output)


def _find_model_root(extracted_dir: str) -> str:
    entries = os.listdir(extracted_dir)
    if len(entries) == 1:
        candidate = os.path.join(extracted_dir, entries[0])
        if os.path.isdir(candidate):
            return candidate
    return extracted_dir


@contextlib.contextmanager
def _open_spark_input(input_path: str):
    """Yield the effective model directory, extracting archives as needed."""
    lower = input_path.lower()
    if lower.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(input_path) as zf:
                zf.extractall(tmp)
            yield _find_model_root(tmp)
    elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(input_path, "r:gz") as tf:
                tf.extractall(tmp)
            yield _find_model_root(tmp)
    else:
        yield input_path


def _is_spark_input(input_path: str) -> bool:
    lower = input_path.lower()
    return (
        os.path.isdir(input_path)
        or lower.endswith(".zip")
        or lower.endswith(".tar.gz")
        or lower.endswith(".tgz")
    )


def _cmd(args: argparse.Namespace) -> None:
    from omle_convert import to_omle as convert
    from omle_convert.spark import from_spark

    input_path = args.input
    output_path = Path(args.output)
    feature_names = _parse_list(args.feature_names)
    model_name = args.model_name

    if _is_spark_input(input_path):
        if not os.path.exists(input_path):
            print(f"Error: input path does not exist: {input_path!r}", file=sys.stderr)
            sys.exit(1)
        try:
            with _open_spark_input(input_path) as model_dir:
                model = from_spark(
                    model_dir,
                    feature_names=feature_names or None,
                )
        except Exception as exc:
            print(f"Error during conversion: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        if model_name:
            model.metadata.name = model_name
    else:
        model = convert(
            input_path,
            feature_names=feature_names,
            target_name=args.target_name,
            class_labels=_parse_list(args.class_labels),
            model_name=model_name,
        )

    _save(model, output_path)
    print(f"Saved OMLE model to {args.output}")


def _build_parser() -> argparse.ArgumentParser:
    from omle_convert import __version__
    parser = argparse.ArgumentParser(
        prog="omle-convert",
        description="Convert ML models to the OMLE format (.omle)",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"omle-convert {__version__}",
    )
    parser.add_argument("input",  help="Path to the source model file or directory")
    parser.add_argument("output", help="Path to write the OMLE model (.omle or .json)")
    parser.add_argument(
        "--feature-names",
        metavar="NAMES",
        default=None,
        help="Comma-separated feature names (e.g. age,income,score)",
    )
    parser.add_argument(
        "--target-name",
        metavar="NAME",
        default="y",
        help="Target variable name (default: y)",
    )
    parser.add_argument(
        "--class-labels",
        metavar="LABELS",
        default=None,
        help="Comma-separated class labels for classification (e.g. no,yes)",
    )
    parser.add_argument(
        "--model-name",
        metavar="NAME",
        default="",
        help="Optional model name stored in metadata",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print full traceback on error",
    )
    parser.set_defaults(func=_cmd)
    return parser


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
