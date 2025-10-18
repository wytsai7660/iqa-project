# This script takes the output CSV of process_liqe_labels.py for the KADID-10k
# dataset and adds a column named "reference" that contains filenames inside
# datasets/kadid-10k/images/reference-images that identify the undistorted
# reference image of each row. The filenames of the reference images are
# obtained by parsing the filenames of the distorted images.

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pandas import options, read_csv # pyright: ignore[reportUnknownVariableType]

options.mode.copy_on_write = True

def main(args: Namespace):
    labels_data_frame = read_csv(args.labels_path, keep_default_na=False, index_col="filename") # keep_default_na=False makes read_csv treat empty scene types as empty strings
    labels_data_frame["reference"] = labels_data_frame.index.str.split("_").str[0] + ".png"
    labels_data_frame.to_csv(args.output_path)

if __name__ == "__main__":
    argument_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="This script takes the output CSV of process_liqe_labels.py for the KADID-10k dataset and adds a column named \"reference\".",
        epilog="For more information, see the comment at the top of this script."
    )
    _ = argument_parser.add_argument("labels_path",
        help="The output CSV of process_liqe_labels.py for the KADID-10k dataset.")
    _ = argument_parser.add_argument("output_path",
        help="The path to store the output CSV with the \"reference\" column added.")
    args = argument_parser.parse_args()
    main(args)
