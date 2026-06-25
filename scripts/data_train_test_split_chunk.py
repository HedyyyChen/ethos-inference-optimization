import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from joblib import Parallel, delayed

from ethos.utils import get_logger

logger = get_logger()

DEFAULT_FORMAT = ".csv.gz"
PARQUET_SUFFIX = ".parquet"
CHUNKSIZE = 50000  # Tune based on your RAM


def _process_and_write_chunk(
    df,
    split_paths,
    id_col,
    date_col,
    cutoff_date,
    train_ids_set,
    test_ids_set,
    need_id_filter,
    need_date_filter,
    is_first: bool,
    is_parquet: bool,
):
    # Convert date column if needed and not already datetime
    if need_date_filter and date_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    for split_path in split_paths:
        _df = df.copy()
        processed = False

        # ID-based split
        if need_id_filter and id_col in _df.columns:
            valid_ids = (
                train_ids_set if "train" in str(split_path) else test_ids_set
            )
            _df = _df[_df[id_col].isin(valid_ids)]
            processed = True

        # Date-based split
        if need_date_filter and date_col in _df.columns:
            if _df[date_col].isna().mean() >= 0.01:
                logger.warning(
                    f"Skipping date filter for {split_path} due to high NaN ratio in {date_col}"
                )
            else:
                cond = _df[date_col] < cutoff_date
                fold_name = split_path.parts[1] if len(split_path.parts) > 1 else split_path.name
                if fold_name.endswith("prospective") or (
                    len(split_paths) == 2 and fold_name.startswith("test")
                ):
                    cond = ~cond
                _df = _df[cond]
                processed = True

        if _df.empty:
            continue

        split_path.parent.mkdir(exist_ok=True, parents=True)

        if is_parquet:
            # Parquet: write entire file once (handled outside chunk loop)
            _df.to_parquet(split_path, index=False)
        else:
            # CSV: append with header only on first chunk
            mode = "w" if is_first else "a"
            header = is_first
            _df.to_csv(split_path, mode=mode, header=header, index=False)


def dump_splits(
    col: list[str],
    orig_path: Path,
    split_paths: list[Path],
    cutoff_date: Optional[pd.Timestamp],
    subject_id_split: Optional[tuple[Sequence, Sequence]],
):
    if all(p.exists() for p in split_paths):
        logger.warning(f"All output files already exist, skipping: {orig_path}")
        return

    id_col = None if subject_id_split is None else col[0]
    date_col = None if cutoff_date is None else col[-1]

    need_id_filter = id_col is not None
    need_date_filter = date_col is not None

    # Convert ID lists to sets for O(1) lookup
    train_ids_set = set(subject_id_split[0]) if subject_id_split else None
    test_ids_set = set(subject_id_split[1]) if subject_id_split else None

    is_parquet = orig_path.suffix == PARQUET_SUFFIX

    try:
        if is_parquet:
            # Parquet: read all at once (usually memory-efficient)
            df = pd.read_parquet(orig_path)
            _process_and_write_chunk(
                df, split_paths, id_col, date_col, cutoff_date,
                train_ids_set, test_ids_set, need_id_filter, need_date_filter,
                is_first=True, is_parquet=True
            )
            logger.info(f"Saved Parquet split ({df.shape}): {split_paths[0]}")
        else:
            # CSV/GZ: read in chunks
            first_chunk = True
            for chunk in pd.read_csv(orig_path, low_memory=False, chunksize=CHUNKSIZE):
                _process_and_write_chunk(
                    chunk, split_paths, id_col, date_col, cutoff_date,
                    train_ids_set, test_ids_set, need_id_filter, need_date_filter,
                    is_first=first_chunk, is_parquet=False
                )
                first_chunk = False
    except Exception as e:
        logger.error(f"Failed processing {orig_path}: {e}")
        # Clean up any partially written files
        for p in split_paths:
            if p.exists():
                p.unlink()
        raise


def data_train_test_split(
    dataset_dir: str,
    col: str | list[str],
    test_size: float,
    id_data_path: str = None,
    cutoff_date: str = None,
    subset_format: str = DEFAULT_FORMAT,
    seed: int = 42,
    n_jobs: int = 1,
):
    dataset_dir = Path(dataset_dir)
    assert dataset_dir.is_dir(), f"Path is not a directory: {dataset_dir}"

    if not isinstance(col, list):
        col = [col]

    if cutoff_date is not None:
        cutoff_date = pd.Timestamp(cutoff_date)

        if id_data_path is None:
            logger.info(
                f"Performing time-based split using cutoff date: '{cutoff_date}'. "
                f"`test_size` is ignored."
            )
        else:
            logger.info(
                f"Performing combined time + ID split: cutoff='{cutoff_date}', "
                f"test_size={test_size:.2f} → 4 splits."
            )
            if len(col) != 2:
                raise ValueError(
                    "Two column names required for combined split: ('id_col', 'date_col')."
                )
    elif id_data_path is not None:
        logger.info(f"Performing ID-based split with test_size={test_size:.2f}.")
    else:
        raise ValueError("Either `cutoff_date` or `id_data_path` must be provided.")

    subject_id_split = None
    if id_data_path is not None:
        id_data_path = Path(dataset_dir) / id_data_path
        if not id_data_path.is_file():
            raise FileNotFoundError(f"'{id_data_path}' not found.")

        logger.info(f"Loading subject IDs from: '{id_data_path}'")
        if id_data_path.suffix == PARQUET_SUFFIX:
            df = pd.read_parquet(id_data_path)
        else:
            df = pd.read_csv(id_data_path, low_memory=False)

        id_col = col[0]
        if not df[id_col].is_unique:
            raise ValueError(f"Column '{id_col}' is not unique in '{id_data_path}'")

        test_df = df.sample(frac=test_size, random_state=seed)
        train_df = df.drop(test_df.index)

        subject_id_split = (train_df[id_col].tolist(), test_df[id_col].tolist())
        logger.info(
            "Subject count (train/test): {:,}/{:,} (test_size={:.0%})".format(
                len(subject_id_split[0]),
                len(subject_id_split[1]),
                len(subject_id_split[1]) / len(df),
            )
        )

    data_format = subset_format if subset_format.startswith(".") else f".{subset_format}"
    orig_paths = list(dataset_dir.rglob(f"*{data_format}"))
    logger.info(f"Found {len(orig_paths)} data files to split.")

    out_dir = dataset_dir.parent / f"{dataset_dir.name}_Data"
    folds = ["train", "test"]
    if cutoff_date is not None and id_data_path is not None:
        folds.extend(["train_prospective", "test_prospective"])
    out_dirs = [out_dir / suffix for suffix in folds]

    out_paths = []
    for orig_path in orig_paths:
        subset_rel_out = orig_path.relative_to(dataset_dir).parent
        true_stem = orig_path.stem.split(".")[0]  # e.g., "events.csv.gz" → "events"
        split_paths = [
            out_dir / subset_rel_out / f"{true_stem}{data_format}" for out_dir in out_dirs
        ]
        out_paths.append((orig_path, split_paths))

    logger.info(f"Writing splits to: '{out_dir.resolve()}'")
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(dump_splits)(
            col,
            orig_path,
            split_paths,
            cutoff_date,
            subject_id_split,
        )
        for orig_path, split_paths in out_paths
    )
    logger.info("Splitting completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test with optional time + ID criteria. "
        "Output saved as '<dataset>_Data/' beside input dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", type=str, help="Path to the dataset directory.")
    parser.add_argument(
        "--col",
        type=str,
        nargs="+",
        default="subject_id",
        help="Column(s): <id_col> for ID split, or <id_col> <date_col> for combined split.",
    )
    parser.add_argument(
        "--id_data_path",
        type=str,
        default="hosp/patients.csv.gz",
        help="File (relative to dataset dir) containing unique subject IDs for splitting.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of subjects for test set (used only with --id_data_path).",
    )
    parser.add_argument(
        "--cutoff_date",
        type=str,
        help="Cutoff date (YYYY-MM-DD); data before = train, after = test.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default=DEFAULT_FORMAT,
        help="File extension of data subsets (e.g., '.csv.gz', '.parquet').",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42, help="Random seed for ID-based split."
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=1, help="Number of parallel workers."
    )
    args = parser.parse_args()

    # Fix typo: "dateset" → "dataset"
    data_train_test_split(
        args.dataset,
        args.col,
        args.test_size,
        args.id_data_path,
        args.cutoff_date,
        args.data_format,
        args.seed,
        args.n_jobs,
    )