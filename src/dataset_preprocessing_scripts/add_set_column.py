# This script is like koniq_10k_add_set_column.py, but used for datasets that
# don't have official splits information. If there is a column named
# "reference", this script uses that to do a groups split, where if two images
# have the same reference image, then they will always belong in the same set.

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from numpy import ndarray
from pandas import options, read_csv # pyright: ignore[reportUnknownVariableType]
from sklearn.model_selection import GroupShuffleSplit

options.mode.copy_on_write = True

def main(args: Namespace):
    labels_data_frame = read_csv(args.labels_path, keep_default_na=False, index_col="filename") # keep_default_na=False makes read_csv treat empty scene types as empty strings
    if "reference" in labels_data_frame.columns:
        groups = labels_data_frame["reference"]
    else:
        groups = labels_data_frame.index.to_series()
    gss_train_val_and_test = GroupShuffleSplit(1, test_size=args.testing_set_ratio, random_state=args.seed)
    train_val_indices: ndarray
    test_indices: ndarray
    train_val_indices, test_indices = next(gss_train_val_and_test.split(labels_data_frame, groups=groups))
    labels_data_frame.loc[labels_data_frame.index[test_indices], "set"] = "testing"
    val_ratio_out_of_train_val = args.validation_set_ratio / (args.training_set_ratio + args.validation_set_ratio)
    train_val_subset_data_frame = labels_data_frame.iloc[train_val_indices]
    train_val_subset_groups = groups.iloc[train_val_indices]
    gss_train_and_val = GroupShuffleSplit(1, test_size=val_ratio_out_of_train_val, random_state=args.seed)
    train_indices: ndarray
    val_indices: ndarray
    train_indices, val_indices = next(gss_train_and_val.split(train_val_subset_data_frame, groups=train_val_subset_groups))
    # train_indices and val_indices index train_val_subset_data_frame, not
    # labels_data_frame, but the index is unchanged between them, so we can
    # align train_indices and val_indices back to labels_data_frame through the
    # index.
    labels_data_frame.loc[train_val_subset_data_frame.index[train_indices], "set"] = "training"
    labels_data_frame.loc[train_val_subset_data_frame.index[val_indices], "set"] = "validation"
    labels_data_frame.to_csv(args.output_path)

if __name__ == "__main__":
    argument_parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="This script takes the output CSV of process_liqe_labels.py for a dataset and adds a column named \"set\".",
        epilog="For more information, see the comment at the top of this script."
    )
    _ = argument_parser.add_argument("labels_path",
        help="The output CSV of process_liqe_labels.py for the KonIQ-10k dataset.")
    _ = argument_parser.add_argument("output_path",
        help="The path to store the output CSV with the \"set\" column added.")
    _ = argument_parser.add_argument("--training-set-ratio",
        help="A number between 0 and 1 that controls how much of the whole dataset to include as the training set.",
        type=float,
        default=0.72)
    _ = argument_parser.add_argument("--validation-set-ratio",
        help="A number between 0 and 1 that controls how much of the whole dataset to include as the validation set.",
        type=float,
        default=0.18)
    _ = argument_parser.add_argument("--testing-set-ratio",
        help="A number between 0 and 1 that controls how much of the whole dataset to include as the testing set.",
        type=float,
        default=0.1)
    _ = argument_parser.add_argument("--seed",
        help="An integer that controls the random seed to use when splitting.",
        type=int,
        default=42)
    args = argument_parser.parse_args()
    main(args)
