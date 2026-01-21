import sys

from pipeline.loader import CSVLoader
from pipeline.cleansing import DataCleaningHandler
from pipeline.encoding import EncodingHandler
from pipeline.split import SplitTargetHandler
from pipeline.save import NumpySaver


def main():
    path = sys.argv[1]

    pipeline = CSVLoader(
        path,
        DataCleaningHandler(
            EncodingHandler(
                SplitTargetHandler(
                    target_column="Авто",
                    next_handler=NumpySaver(),
                )
            )
        ),
    )

    pipeline.handle({})


if __name__ == "__main__":
    main()
