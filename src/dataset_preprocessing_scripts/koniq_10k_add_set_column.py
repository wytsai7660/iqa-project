# This script takes the output CSV of process_liqe_labels.py for the KonIQ-10k
# dataset and adds a column named "set" that contains one of the following
# strings: "training", "testing", or "validation". The value indicates the
# official split the image belongs. This script requires the CSV downloaded from
# https://github.com/subpic/koniq/blob/master/metadata/koniq10k_distributions_sets.csv.

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pandas import options, read_csv # pyright: ignore[reportUnknownVariableType]

options.mode.copy_on_write = True

def main(args: Namespace):
    labels_data_frame = read_csv(args.labels_path, keep_default_na=False, index_col="filename") # keep_default_na=False makes read_csv treat empty scene types as empty strings
    official_splits_data_frame = read_csv(args.official_splits_path, usecols=["image_name", "set"], index_col="image_name")
    official_splits_data_frame["set"] = official_splits_data_frame["set"].replace("test", "testing") # pyright: ignore[reportUnknownMemberType]
    output_data_frame = labels_data_frame.join(official_splits_data_frame, sort=True, validate="1:1")
    output_data_frame.to_csv(args.output_path)

if __name__ == "__main__":
    argument_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="This script takes the output CSV of process_liqe_labels.py for the KonIQ-10k dataset and adds a column named \"set\".",
        epilog="For more information, see the comment at the top of this script."
    )
    _ = argument_parser.add_argument("labels_path",
        help="The output CSV of process_liqe_labels.py for the KonIQ-10k dataset.")
    _ = argument_parser.add_argument("official_splits_path",
        help="The CSV downloaded from https://github.com/subpic/koniq/blob/master/metadata/koniq10k_distributions_sets.csv that contains the official splits information.")
    _ = argument_parser.add_argument("output_path",
        help="The path to store the output CSV with the \"set\" column added.")
    args = argument_parser.parse_args()
    main(args)
