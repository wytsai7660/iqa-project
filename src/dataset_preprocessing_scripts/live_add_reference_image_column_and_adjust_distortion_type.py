# This script takes the output CSV of process_liqe_labels.py for the LIVE
# dataset and adds a column named "reference" that contains filenames inside
# datasets/live/images/reference-images that identify the undistorted reference
# image of each row. The filenames of the reference images are obtained by
# parsing the info.txt files under each distortion type directory in the LIVE
# dataset, downloaded from
# https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/blob/main/live.tgz.
#
# It then changes the distortion type of all the images under "fastfading" to
# "jpeg2000 compression", according to the mapping provided by LIQE.

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pandas import DataFrame, concat, options, read_csv # pyright: ignore[reportUnknownVariableType]
from pathlib import PurePath
from typing import Callable

options.mode.copy_on_write = True

def main(args: Namespace):
    labels_data_frame = read_csv(args.labels_path, keep_default_na=False, index_col="filename") # keep_default_na=False makes read_csv treat empty scene types as empty strings
    labels_data_frame.loc[labels_data_frame.index.str.startswith("fastfading"), "distortion"] = "jpeg2000 compression"
    distortion_types = ["fastfading", "gblur", "jp2k", "jpeg", "wn"]
    distortion_type_reference_image_data_frames: list[DataFrame] = []
    for distortion_type in distortion_types:
        all_distortion_type_reference_image_data_frame = read_csv(
            PurePath(args.live_dataset_path) / distortion_type / "info.txt",
            sep=r"\s+",
            names=["reference", "filename", "strength"],
            usecols=["filename", "reference"],
            index_col="filename"
        )
        add_subdirectory_segment: Callable[[str], str] = lambda filename: str(PurePath(distortion_type) / filename)
        all_distortion_type_reference_image_data_frame.index = all_distortion_type_reference_image_data_frame.index.map(add_subdirectory_segment) # pyright: ignore[reportUnknownMemberType]
        distortion_type_reference_image_data_frames.append(all_distortion_type_reference_image_data_frame)
    all_distortion_type_reference_image_data_frame = concat(distortion_type_reference_image_data_frames)
    output_data_frame = labels_data_frame.join(all_distortion_type_reference_image_data_frame, sort=True, validate="1:1")
    output_data_frame.to_csv(args.output_path)


if __name__ == "__main__":
    argument_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="""
            This script takes the output CSV of process_liqe_labels.py for the
            LIVE dataset and adds a column named \"reference\". It then changes
            the distortion type of all the images under \"fastfading\" to
            \"jpeg2000 compression\", according to the mapping provided by LIQE.
            """,
        epilog="For more information, see the comment at the top of this script."
    )
    _ = argument_parser.add_argument("labels_path",
        help="The output CSV of process_liqe_labels.py for the LIVE dataset.")
    _ = argument_parser.add_argument("live_dataset_path",
        help="The path to the directory of the downloaded LIVE dataset. This script will work as long as all info.txt files are in their original positions.")
    _ = argument_parser.add_argument("output_path",
        help="The path to store the output CSV with the \"reference\" column added.")
    args = argument_parser.parse_args()
    main(args)
