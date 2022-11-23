import os
import types
import gcsfs

from functools import partial
from pathlib import Path, PurePath
from typing import Dict, Union, List, Optional, Tuple
from tqdm.contrib.concurrent import thread_map
from datasets.data_files import contains_wildcards, EmptyDatasetError, _get_data_files_patterns, DataFilesDict
from datasets.packaged_modules.parquet.parquet import Parquet, ParquetConfig
from datasets.utils import logging
from datasets.utils.file_utils import is_relative_path

FILES_TO_IGNORE = ["README.md", "config.json", "dataset_infos.json", "dummy_data.zip", "dataset.json"]

logger = logging.get_logger(__name__)

DATASET_PATH = "[DATASET_PATH]"
DATASET_NAME = "[DATASET_NAME]"
DATASET_DESCRIPTION = "[DATASET_DESCRIPTION]"
DATASETS_PACKAGE_VERSION = "[DATASETS_PACKAGE_VERSION]"


def get_google_token() -> Optional[str]:
    return os.getenv("GCLOUD_TOKEN")


fs = gcsfs.GCSFileSystem(token=get_google_token(), access="read_only")


def resolve_single_pattern_gcloud(
        base_path: str, pattern: str, allowed_extensions: Optional[List[str]] = None
) -> List[PurePath]:
    """
    Return the absolute paths to all the files that match the given patterns.
    """
    if is_relative_path(pattern):
        pattern = os.path.join(base_path, pattern)
    else:
        base_path = os.path.splitdrive(pattern)[0] + os.sep
    matched_paths = [PurePath(filepath) for filepath in fs.glob(pattern) if
                     fs.isfile(filepath) and filepath not in FILES_TO_IGNORE]
    if allowed_extensions is not None:
        out = [
            filepath
            for filepath in matched_paths
            if any(suffix[1:] in allowed_extensions for suffix in filepath.suffixes)
        ]
        if len(out) < len(matched_paths):
            invalid_matched_files = list(set(matched_paths) - set(out))
            print(
                f"Some files matched the pattern '{pattern}' at {Path(base_path).resolve()}"
                f" but don't have valid data file extensions: {invalid_matched_files}"
            )
    else:
        out = matched_paths
    if not out and not contains_wildcards(pattern):
        error_msg = f"Unable to find '{pattern}' at {Path(base_path).resolve()}"
        if allowed_extensions is not None:
            error_msg += f" with any supported extension {list(allowed_extensions)}"
        raise FileNotFoundError(error_msg)
    return sorted(out)


def resolve_patterns_in_gcloud(
        base_path: str, patterns: List[str], allowed_extensions: Optional[List[str]] = None
) -> List[PurePath]:
    data_files = []
    for pattern in patterns:
        for path in resolve_single_pattern_gcloud(base_path, pattern, allowed_extensions):
            data_files.append(path)

    if not data_files:
        error_msg = f"Unable to resolve any data file that matches '{patterns}' at {Path(base_path).resolve()}"
        if allowed_extensions is not None:
            error_msg += f" with any supported extension {list(allowed_extensions)}"
        raise FileNotFoundError(error_msg)
    return data_files


def get_single_origin_metadata_from_gcloud(data_file: str) -> Tuple[str]:
    return fs.info(data_file)["crc32c"],


def get_origin_metadata_from_gcloud(
        data_files: List[PurePath], max_workers=64) -> List[Tuple[str]]:
    return thread_map(
        partial(get_single_origin_metadata_from_gcloud),
        data_files,
        max_workers=max_workers,
        tqdm_class=logging.tqdm,
        desc="Resolving data files",
        disable=len(data_files) <= 16 or not logging.is_progress_bar_enabled(),
    )


class CustomDataFilesList(List[str]):

    def __init__(self, data_files: List[str], origin_metadata: List[Tuple[str]]):
        super().__init__(data_files)
        self.origin_metadata = origin_metadata

    @classmethod
    def from_gcloud(
            cls,
            patterns: List[str],
            base_path: Optional[str] = None,
            allowed_extensions: Optional[List[str]] = None,
    ) -> "CustomDataFilesList":
        base_path = base_path if base_path is not None else str(Path().resolve())
        data_files = resolve_patterns_in_gcloud(base_path, patterns, allowed_extensions)
        origin_metadata = get_origin_metadata_from_gcloud(data_files)
        return cls([str(data_file) for data_file in data_files], origin_metadata)


class CustomDataFilesDict(DataFilesDict):
    @classmethod
    def from_gcloud(cls,
                    patterns: Dict[str, Union[List[str], CustomDataFilesList]],
                    base_path: Optional[str] = None,
                    allowed_extensions: Optional[List[str]] = None):
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                CustomDataFilesList.from_gcloud(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                )
            )
        return out


def get_data_patterns_in_gcloud(base_path: str) -> Dict[str, List[str]]:
    """
    Get the default pattern from a repository by testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    Some examples of supported patterns:

    Input:

        my_dataset_repository/
        ├── README.md
        └── dataset.csv

    Output:

        {"train": ["**"]}

    Input:

        my_dataset_repository/
        ├── README.md
        ├── train.csv
        └── test.csv

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train.csv
            └── test.csv

        my_dataset_repository/
        ├── README.md
        ├── train_0.csv
        ├── train_1.csv
        ├── train_2.csv
        ├── train_3.csv
        ├── test_0.csv
        └── test_1.csv

    Output:

        {"train": ["**train*"], "test": ["**test*"]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train/
            │   ├── shard_0.csv
            │   ├── shard_1.csv
            │   ├── shard_2.csv
            │   └── shard_3.csv
            └── test/
                ├── shard_0.csv
                └── shard_1.csv

    Output:

        {"train": ["**train*/**"], "test": ["**test*/**"]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train-00000-of-00003.csv
            ├── train-00001-of-00003.csv
            ├── train-00002-of-00003.csv
            ├── test-00000-of-00001.csv
            ├── random-00000-of-00003.csv
            ├── random-00001-of-00003.csv
            └── random-00002-of-00003.csv

    Output:

        {
            "train": ["data/train-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].*"],
            "test": ["data/test-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].*"],
            "random": ["data/random-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].*"],
        }

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    resolver = partial(resolve_single_pattern_gcloud, PurePath(base_path))
    try:
        return _get_data_files_patterns(resolver)
    except FileNotFoundError:
        raise EmptyDatasetError(f"The directory at {base_path} doesn't contain any data files") from None


def custom_download_and_extract(self, url_or_urls):
    def my_custom_download(src_url: str, dst_path: str):
        fs.download(src_url, dst_path)

    return self.extract(self.download_custom(url_or_urls, my_custom_download))


class CustomDatasetBuilder(Parquet):
    BUILDER_CONFIGS = [
        ParquetConfig(
            name=DATASET_NAME,
            version=DATASETS_PACKAGE_VERSION,
            description=DATASET_DESCRIPTION,
            data_files=CustomDataFilesDict.from_gcloud(base_path=DATASET_PATH,
                                                       patterns=get_data_patterns_in_gcloud(DATASET_PATH))
        ),
    ]

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        dl_manager.download_and_extract = types.MethodType(custom_download_and_extract, dl_manager)
        return super()._split_generators(dl_manager)
