import argparse
from pathlib import Path

from pipeline.basic_features import BasicHandler
from pipeline.category_features import CategoryHandler
from pipeline.cleansing import DataCleaningHandler
from pipeline.complex_features import ComplexHandler
from pipeline.encoding import EncodingHandler
from pipeline.loader import CSVLoader
from pipeline.normalization import NormalizeHandler
from pipeline.save import NumpySaver
from pipeline.split import SplitTargetHandler


def build_pipeline(csv_path: Path, target_column: str = "salary") -> CSVLoader:
    """
    Builds a full pipeline from needed handlers, passing key params to them.

    :param csv_path: path to the input csv file
    :type csv_path: Path
    :param target_column: target column to become target vector Y after analysis
    :type target_column: str
    :return: CSVLoader with set up next handlers
    :rtype: Handler
    """

    loader = CSVLoader(path=str(csv_path))
    cleaner = DataCleaningHandler()
    basic = BasicHandler()
    profile_loc = CategoryHandler()
    education = ComplexHandler()
    positions = NormalizeHandler()
    encoder = EncodingHandler()
    splitter = SplitTargetHandler(target_column=target_column)
    saver = NumpySaver(output_dir=csv_path.parent)

    (loader.set_next(cleaner).set_next(basic)
     .set_next(profile_loc).set_next(education).set_next(positions)
     .set_next(encoder).set_next(splitter).set_next(saver))

    return loader


def parse_args() -> argparse.Namespace:
    """
    Parses args to extract path to the csv file with data.

    :return: parsed args, including csv_path
    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser(description="HH.ru parsing pipeline")
    parser.add_argument("csv_path", type=str, help="Path to hh.csv")
    parser.add_argument(
        "--target",
        type=str,
        default="salary",
        help="Target column name (after feature parsing)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Runs a full parsing and preparing pipeline on the provided via arguments data.

    :return: Creates x_data.npy and y_data.npy near the initial data file
    """

    args = parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    pipeline = build_pipeline(csv_path, target_column=args.target)
    pipeline.handle({})


if __name__ == "__main__":
    main()
