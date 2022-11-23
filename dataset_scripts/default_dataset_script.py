import os
import types
import urllib
from typing import Optional

import gcsfs

from datasets.data_files import DataFilesDict
from datasets.packaged_modules.parquet.parquet import Parquet, ParquetConfig
from datasets.utils import logging

logger = logging.get_logger(__name__)

DATASET_PATH = "[DATASET_PATH]"
DATASET_NAME = "[DATASET_NAME]"
DATASET_DESCRIPTION = "[DATASET_DESCRIPTION]"
DATASETS_PACKAGE_VERSION = "[DATASETS_PACKAGE_VERSION]"
DATA_FILES_DICT = DataFilesDict([DATA_FILES_DICT])


def get_google_token() -> Optional[str]:
    return os.getenv("GCS_TOKEN")


def datafilesdict_to_url(self) -> DataFilesDict:
    new_dict = {}
    for split, files_list in self.items():
        new_dict[split] = []
        for file in files_list:
            full_path = file.split("/")
            bucket = full_path[0]
            file_path = urllib.parse.quote(str("/".join(full_path[1:])).encode('utf8'), safe='')
            new_dict[split].append(os.path.join("https://storage.googleapis.com", os.path.join(bucket, file_path)))
    return DataFilesDict(new_dict)


def custom_download_and_extract(self, url_or_urls, fs):
    def my_custom_download(src_url: str, dst_path: str):
        fs.download(src_url, dst_path)

    return self.extract(self.download_custom(url_or_urls, my_custom_download))


class CustomDatasetBuilder(Parquet):
    BUILDER_CONFIGS = [
        ParquetConfig(
            name=DATASET_NAME,
            version=DATASETS_PACKAGE_VERSION,
            description=DATASET_DESCRIPTION,
            data_files=DATA_FILES_DICT
        ),
    ]

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if dl_manager.is_streaming:
            self.config.data_files = datafilesdict_to_url(self.config.data_files)
        else:
            fs = gcsfs.GCSFileSystem(token=get_google_token(), access="read_only")
            dl_manager.download_and_extract = types.MethodType(
                lambda me, url: custom_download_and_extract(self=me, url_or_urls=url, fs=fs), dl_manager)
        return super()._split_generators(dl_manager)
