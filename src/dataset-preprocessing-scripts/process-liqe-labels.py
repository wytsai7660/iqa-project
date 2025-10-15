# This script processes the dataset labels provided by LIQE into a more usable
# CSV. It takes a file that contains the MOS and standard deviation of each
# image sample, downloaded from any of the subdirectories in
# https://github.com/zwx8981/LIQE/tree/main/IQA_Database, named *_all.txt, with
# another one that contains the distortion type and scene types (3 per image) of
# each image sample, downloaded from any of the subdirectories in
# https://github.com/zwx8981/LIQE/tree/main/IQA_Database, named *_clip.txt, into
# a CSV that contains the filename, MOS, standard deviation, distortion type,
# and 3 scene types of each image sample in that dataset.
# 
# This script will strip all but the last path segment of the image sample paths
# from the input files, convert the "invalid" scene type into the empty string,
# and map the distortion types into the 11 ones described in the paper (the code
# provided by the paper actually uses 33 types first, then maps them back into
# 11 types).

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pandas import merge, options, read_csv # pyright: ignore[reportUnknownVariableType]
from pathlib import PurePath
from typing import Callable

options.mode.copy_on_write = True

def main(args: Namespace):
    mos_stddev_dataframe = read_csv(
        args.mos_stddev_path,
        sep="\t",
        names=["filename", "mos", "stddev"],
        index_col="filename"
    )
    distortion_scene_types_dataframe = read_csv(
        args.distortion_scene_types_path,
        sep="\t",
        names=["filename", "mos", "distortion", "scene1", "scene2", "scene3"],
        index_col="filename"
    )
    output_dataframe = merge(
        mos_stddev_dataframe,
        distortion_scene_types_dataframe[["distortion", "scene1", "scene2", "scene3"]],
        left_index=True,
        right_index=True,
        sort=True,
        validate="1:1"
    )
    get_last_path_segment: Callable[[str], str] = lambda filename: PurePath(filename).name
    output_dataframe.index = output_dataframe.index.map(get_last_path_segment) # pyright: ignore[reportUnknownMemberType]
    output_dataframe[["scene1", "scene2", "scene3"]] = output_dataframe[["scene1", "scene2", "scene3"]].replace("invalid", "") # pyright: ignore[reportUnknownMemberType]
    distortion_type_map = {
        "jpeg2000 compression": "jpeg2000 compression",
        "jpeg compression": "jpeg compression",
        "white noise": "noise",
        "gaussian blur": "blur",
        "fastfading": "jpeg2000 compression",
        "fnoise": "noise",
        "contrast": "contrast",
        "lens": "blur",
        "motion": "blur",
        "diffusion": "color",
        "shifting": "blur",
        "color quantization": "quantization",
        "oversaturation": "color",
        "desaturation": "color",
        "white with color": "noise",
        "impulse": "noise",
        "multiplicative": "noise",
        "white noise with denoise": "noise",
        "brighten": "overexposure",
        "darken": "underexposure",
        "shifting the mean": "other",
        "jitter": "spatial",
        "noneccentricity patch": "spatial",
        "pixelate": "spatial",
        "quantization": "quantization",
        "color blocking": "spatial",
        "sharpness": "contrast",
        "realistic blur": "blur",
        "realistic noise": "noise",
        "underexposure": "underexposure",
        "overexposure": "overexposure",
        "realistic contrast change": "contrast",
        "other realistic": "other"
    }
    output_dataframe["distortion"] = output_dataframe["distortion"].map(distortion_type_map)
    output_dataframe.to_csv(args.output_path)

if __name__ == "__main__":
    argument_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="This script processes the dataset labels provided by LIQE into a more usable CSV.",
        epilog="For more information, see the comment at the top of this script."
    )
    _ = argument_parser.add_argument("mos_stddev_path",
        help="The file that contains the MOS and standard deviation of each image sample.")
    _ = argument_parser.add_argument("distortion_scene_types_path",
        help="The file that contains the distortion type and 3 scene types of each image sample.")
    _ = argument_parser.add_argument("output_path",
        help="The path to store the output CSV.",
        nargs="?",
        default="labels.csv")
    main(argument_parser.parse_args())
