import itertools
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, Union, Tuple, List

import gcsfs.core
from gcsfs import GCSFileSystem

from datasets.utils.hub import hf_hub_url
from dataset_hub.dataset_pusher.utils import Readme, get_dataset_script
from dataset_hub.dataset_pusher.utils.gcloud_commands import gcloud_list_repo_files, gcloud_upload_filebytes,\
    gcloud_file_size, gcloud_delete_file
from datasets.info import DatasetInfosDict
from datasets.utils.metadata import DatasetMetadata
from huggingface_hub import HfApi, HfFolder, create_repo

from datasets import Dataset, DatasetDict, DatasetInfo, SplitDict, SplitInfo, config, Audio, Image, DownloadConfig
from datasets.download.streaming_download_manager import xgetsize
from datasets.features.features import require_decoding
from datasets.table import table_visitor, embed_table_storage
from datasets.utils import logging
from datasets.utils.file_utils import _retry, cached_path
from datasets.utils.py_utils import convert_file_size_to_int

logger = logging.get_logger(__name__)
_split_re = r"^\w+(\.\w+)*$"


def push_to_hub_dataset(
        dataset: Union[Dataset, DatasetDict],
        repo_id,
        gcloud_path: str,
        project: Optional[str] = None,
        gcloud_token: Optional[str] = None,
        private: Optional[bool] = False,
        token: Optional[str] = None,
        branch: Optional[None] = None,
        max_shard_size: Optional[Union[int, str]] = None,
        embed_external_files: bool = True,
):
    """Pushes the dataset privately to gcloud as a Parquet dataset and uploads the loading script to huggingface hub

    Each dataset split will be pushed independently. The pushed dataset will keep the original split names.

    The resulting Parquet files are dataset_dict-contained by default: if your dataset contains :class:`Image` or :class:`Audio`
    data, the Parquet files will store the bytes of your images or audio files.
    You can disable this by setting `embed_external_files` to False.

    Args:
        dataset: the dataset dict that will be uploaded
        repo_id (:obj:`str`):
            The ID of the repository to push to in the following format: ``<user>/<dataset_name>`` or
            ``<org>/<dataset_name>``. Also accepts ``<dataset_name>``, which will default to the namespace
            of the logged-in user.
        gcloud_path (:obj:`str`):
            The Path in gcloud where the files will be uploaded
        private (Optional :obj:`bool`):
            Whether the huggingface dataset repository should be set to private or not.
            Only affects repository creation: a repository that already exists will not be affected by that parameter.
            The data files uploaded to gcloud are always private
        project (Optional :obj:`str`):
            Name of the gcloud project
        gcloud_token (Optional :obj:`str`):
            Auth token for gcloud, if left to None, gcsfs will attempt to find it.
            https://gcsfs.readthedocs.io/en/latest/api.html
        token (Optional :obj:`str`):
            An optional authentication token for the Hugging Face Hub. If no token is passed, will default
            to the token saved locally when logging in with ``huggingface-cli login``. Will raise an error
            if no token is passed and the user is not logged-in.
        branch (Optional :obj:`str`):
            The git branch on which to push the dataset.
        max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
            The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
            (like `"500MB"` or `"1GB"`).
        embed_external_files (:obj:`bool`, default ``True``):
            Whether to embed file bytes in the shards.
            In particular, this will do the following before the push for the fields of type:

            - :class:`Audio` and class:`Image`: remove local path information and embed file content in the Parquet files.

    Example:

    ```python
    >>> push_to_hub_dataset(dataset, repo_id="<organization>/<dataset_id>", gcloud_path="<bucket>/<dataset_id>")
    ```
    """
    fs = gcsfs.GCSFileSystem(project=project, token=gcloud_token)
    push_to_hub_dataset_fs(dataset=dataset, repo_id=repo_id, gcloud_path=gcloud_path, fs=fs, private=private,
                           token=token, branch=branch, max_shard_size=max_shard_size,
                           embed_external_files=embed_external_files)


def push_to_hub_dataset_fs(
        dataset: Union[Dataset, DatasetDict],
        repo_id: str,
        gcloud_path: str,
        fs: GCSFileSystem,
        private: Optional[bool] = False,
        token: Optional[str] = None,
        branch: Optional[None] = None,
        max_shard_size: Optional[Union[int, str]] = None,
        embed_external_files: bool = True,
):
    """Pushes the dataset privately to gcloud as a Parquet dataset and uploads the loading script to huggingface hub

    Each dataset split will be pushed independently. The pushed dataset will keep the original split names.

    The resulting Parquet files are dataset_dict-contained by default: if your dataset contains :class:`Image` or :class:`Audio`
    data, the Parquet files will store the bytes of your images or audio files.
    You can disable this by setting `embed_external_files` to False.

    Args:
        dataset: the dataset dict that will be uploaded
        repo_id (:obj:`str`):
            The ID of the repository to push to in the following format: ``<user>/<dataset_name>`` or
            ``<org>/<dataset_name>``. Also accepts ``<dataset_name>``, which will default to the namespace
            of the logged-in user.
        gcloud_path (:obj:`str`):
            The Path in gcloud where the files will be uploaded
        fs: (:obj:`GCSFileSystem`):
            The gcsfs fiel system to upload to gcloud
        private (Optional :obj:`bool`):
            Whether the huggingface dataset repository should be set to private or not.
            Only affects repository creation: a repository that already exists will not be affected by that parameter.
            The data files uploaded to gcloud are always private
        token (Optional :obj:`str`):
            An optional authentication token for the Hugging Face Hub. If no token is passed, will default
            to the token saved locally when logging in with ``huggingface-cli login``. Will raise an error
            if no token is passed and the user is not logged-in.
        branch (Optional :obj:`str`):
            The git branch on which to push the dataset.
        max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
            The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
            (like `"500MB"` or `"1GB"`).
        embed_external_files (:obj:`bool`, default ``True``):
            Whether to embed file bytes in the shards.
            In particular, this will do the following before the push for the fields of type:

            - :class:`Audio` and class:`Image`: remove local path information and embed file content in the Parquet files.

    Example:

    ```python
    >>> fs = gcsfs.GCSFileSystem()
    >>> push_to_hub_dataset(dataset, fs=fs, repo_id="<organization>/<dataset_id>", gcloud_path="<bucket>/<dataset_id>")
    ```
    """

    if isinstance(dataset, Dataset):
        dataset = DatasetDict({dataset.split: dataset})
    if "gs://" in gcloud_path:
        logger.warning("gsutil URI not implemented. Removed the gs:// in gcloud path")
        gcloud_path = gcloud_path.split("gs://")[-1]
    if "https://" in gcloud_path:
        raise ValueError("Provide the path in gcloud not the URL, i.e bucket/datasets/mydataset")
    dataset_name = repo_id.split("/")[-1]
    _push_to_hub_dataset_dict(dataset=dataset,
                              dataset_name=dataset_name,
                              repo_id=repo_id,
                              gcloud_path=gcloud_path,
                              fs=fs,
                              private=private,
                              token=token,
                              branch=branch,
                              max_shard_size=max_shard_size,
                              embed_external_files=embed_external_files,
                              )


def _push_to_hub_dataset_dict(
        dataset: DatasetDict,
        dataset_name: str,
        repo_id,
        gcloud_path: str,
        fs: GCSFileSystem,
        private: Optional[bool] = False,
        token: Optional[str] = None,
        branch: Optional[None] = None,
        max_shard_size: Optional[Union[int, str]] = None,
        embed_external_files: bool = True,
):
    dataset._check_values_type()
    dataset._check_values_features()
    data_files_path = {}
    total_uploaded_size = 0
    total_dataset_nbytes = 0
    info_to_dump: DatasetInfo = next(iter(dataset.values())).info.copy()
    info_to_dump.splits = SplitDict()

    for split in dataset.keys():
        if not re.match(_split_re, split):
            raise ValueError(f"Split name should match '{_split_re}' but got '{split}'.")

    for split in dataset.keys():
        logger.warning(f"Pushing split {split} to the Hub.")
        # The split=key needs to be removed before merging
        repo_id, split, uploaded_size, dataset_nbytes, _, _, shards_path_in_repo = _push_parquet_shards_to_gcloud(
            dataset=dataset[split],
            repo_id=repo_id,
            gcloud_path=gcloud_path,
            fs=fs,
            split=split,
            private=private,
            token=token,
            max_shard_size=max_shard_size,
            embed_external_files=embed_external_files,
        )
        total_uploaded_size += uploaded_size
        total_dataset_nbytes += dataset_nbytes
        info_to_dump.splits[split] = SplitInfo(str(split), num_bytes=dataset_nbytes, num_examples=len(dataset[split]))
        data_files_path[split] = [os.path.join(gcloud_path, shard_path_in_repo)
                                  for shard_path_in_repo in shards_path_in_repo]
    info_to_dump.download_checksums = None
    info_to_dump.download_size = total_uploaded_size
    info_to_dump.dataset_size = total_dataset_nbytes
    info_to_dump.size_in_bytes = total_uploaded_size + total_dataset_nbytes

    repo_files = gcloud_list_repo_files(path=gcloud_path, fs=fs)

    # push to the deprecated dataset_infos.json
    if config.DATASETDICT_INFOS_FILENAME in repo_files:
        buffer = BytesIO()
        buffer.write(b'{"default": ')
        info_to_dump._dump_info(buffer, pretty_print=True)
        buffer.write(b"}")
        gcloud_upload_filebytes(filebytes=buffer.getvalue(), path_in_folder=config.DATASETDICT_INFOS_FILENAME,
                                gcloud_path=gcloud_path, fs=fs)
    # push to README
    if "README.md" in repo_files:
        download_config = DownloadConfig()
        download_config.download_desc = "Downloading metadata"
        dataset_readme_path = cached_path(
            hf_hub_url(repo_id, "README.md"),
            download_config=download_config,
        )
        dataset_metadata = DatasetMetadata.from_readme(Path(dataset_readme_path))
        with open(dataset_readme_path, encoding="utf-8") as readme_file:
            readme_content = readme_file.read()
    else:
        dataset_metadata = DatasetMetadata()
        readme_content = Readme.from_dataset(name=dataset_name, dataset=dataset, url=gcloud_path).render()
    DatasetInfosDict({"default": info_to_dump}).to_metadata(dataset_metadata)
    HfApi(endpoint=config.HF_ENDPOINT).upload_file(
        path_or_fileobj=dataset_metadata._to_readme(readme_content).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        revision=branch,
    )
    # Push the dataset script
    HfApi(endpoint=config.HF_ENDPOINT).upload_file(
        path_or_fileobj=get_dataset_script(dataset_description=info_to_dump.description,
                                           dataset_name=dataset_name,
                                           data_files_path=data_files_path,
                                           gcloud_path=gcloud_path).encode(),
        path_in_repo=f"{dataset_name}.py",
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        revision=branch,
    )


def _push_parquet_shards_to_gcloud(
        dataset: Dataset,
        repo_id: str,
        gcloud_path: str,
        fs: GCSFileSystem,
        split: Optional[str] = None,
        private: Optional[bool] = False,
        token: Optional[str] = None,
        max_shard_size: Optional[Union[int, str]] = None,
        embed_external_files: bool = True,
) -> Tuple[str, str, int, int, List[str], int, List[str]]:
    """Pushes the dataset to the hub.
    The dataset is pushed using HTTP requests and does not need to have neither git or git-lfs installed.

    Args:
        dataset (Dataset): the dataset
        split (Optional, :obj:`str`):
            The name of the split that will be given to that dataset. Defaults to `dataset.split`.
        max_shard_size (`int` or `str`, *optional*, defaults to `"500MB"`):
            The maximum size of the dataset shards to be uploaded to the hub. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
        embed_external_files (:obj:`bool`, default ``True``):
            Whether to embed file bytes in the shards.
            In particular, this will do the following before the push for the fields of type:

            - :class:`Audio` and class:`Image`: remove local path information and embed file content in the Parquet files.

    Returns:
        repo_id (:obj:`str`): ID of the repository in <user>/<dataset_name>` or `<org>/<dataset_name>` format
        split (:obj:`str`): name of the uploaded split
        uploaded_size (:obj:`int`): number of uploaded bytes to gcloud
        dataset_nbytes (:obj:`int`): approximate size in bytes of the uploaded dataset afer uncompression
        repo_files (:obj:`str`): list of files in the repository
        deleted_size (:obj:`int`): number of deleted bytes in gcloud

    Example:

    ```python
    >>> dataset.push_to_hub("<organization>/<dataset_id>", split="evaluation")
    ```
    """
    max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)

    api = HfApi(endpoint=config.HF_ENDPOINT)
    token = token if token is not None else HfFolder.get_token()

    if token is None:
        raise EnvironmentError(
            "You need to provide a `token` or be logged in to Hugging Face with `huggingface-cli login`."
        )

    if split is None:
        split = str(dataset.split) if dataset.split is not None else "train"

    if not re.match(_split_re, split):
        raise ValueError(f"Split name should match '{_split_re}' but got '{split}'.")

    identifier = repo_id.split("/")

    if len(identifier) > 2:
        raise ValueError(
            f"The identifier should be in the format <repo_id> or <namespace>/<repo_id>. It is {identifier}, "
            "which doesn't conform to either format."
        )
    elif len(identifier) == 1:
        dataset_name = identifier[0]
        organization_or_username = api.whoami(token)["name"]
        repo_id = f"{organization_or_username}/{dataset_name}"

    create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    # Find decodable columns, because if there are any, we need to:
    # (1) adjust the dataset size computation (needed for sharding) to account for possible external files
    # (2) embed the bytes from the files in the shards
    decodable_columns = (
        [k for k, v in dataset.features.items() if require_decoding(v, ignore_decode_attribute=True)]
        if embed_external_files
        else []
    )

    dataset_nbytes = dataset.data.nbytes

    if decodable_columns:
        # Approximate the space needed to store the bytes from the external files by analyzing the first 1000 examples
        extra_nbytes = 0

        def extra_nbytes_visitor(array, feature):
            nonlocal extra_nbytes
            if isinstance(feature, (Audio, Image)):
                for x in array.to_pylist():
                    if x is not None and x["bytes"] is None and x["path"] is not None:
                        size = xgetsize(x["path"])
                        extra_nbytes += size
                extra_nbytes -= array.field("path").nbytes

        table = dataset.with_format("arrow")[:1000]
        table_visitor(table, extra_nbytes_visitor)

        extra_nbytes = extra_nbytes * len(dataset.data) / len(table)
        dataset_nbytes = dataset_nbytes + extra_nbytes

    if dataset._indices is not None:
        dataset_nbytes = dataset_nbytes * len(dataset._indices) / len(dataset.data)

    num_shards = int(dataset_nbytes / max_shard_size) + 1
    num_shards = max(num_shards, 1)
    shards = (dataset.shard(num_shards=num_shards, index=i, contiguous=True) for i in range(num_shards))

    if decodable_columns:

        def shards_with_embedded_external_files(shards):
            for shard in shards:
                format = shard.format
                shard = shard.with_format("arrow")
                shard = shard.map(
                    embed_table_storage,
                    batched=True,
                    batch_size=1000,
                    keep_in_memory=True,
                )
                shard = shard.with_format(**format)
                yield shard

        shards = shards_with_embedded_external_files(shards)

    files = gcloud_list_repo_files(gcloud_path, fs)
    data_files = [file for file in files if file.startswith("data/")]

    def path_in_repo(_index, shard):
        return f"data/{split}-{_index:05d}-of-{num_shards:05d}-{shard._fingerprint}.parquet"

    shards_iter = iter(shards)
    first_shard = next(shards_iter)
    first_shard_path_in_repo = path_in_repo(0, first_shard)
    if first_shard_path_in_repo in data_files and num_shards < len(data_files):
        logger.warning("Resuming upload of the dataset shards.")

    uploaded_size = 0
    shards_path_in_repo = []
    for index, shard in logging.tqdm(
            enumerate(itertools.chain([first_shard], shards_iter)),
            desc="Pushing dataset shards to the dataset hub",
            total=num_shards,
            disable=not logging.is_progress_bar_enabled(),
    ):
        shard_path_in_repo = path_in_repo(index, shard)
        # Upload a shard only if it doesn't already exist in the repository
        if shard_path_in_repo not in data_files:
            buffer = BytesIO()
            shard.to_parquet(buffer)
            uploaded_size += buffer.tell()
            _retry(
                gcloud_upload_filebytes,
                func_kwargs=dict(
                    filebytes=buffer.getvalue(),
                    path_in_folder=shard_path_in_repo,
                    gcloud_path=gcloud_path,
                    fs=fs
                ),
                status_codes=[504],
                base_wait_time=2.0,
                max_retries=5,
                max_wait_time=20.0,
            )
        shards_path_in_repo.append(shard_path_in_repo)

    # Cleanup to remove unused files
    data_files_to_delete = [
        data_file
        for data_file in data_files
        if data_file.startswith(f"data/{split}-") and data_file not in shards_path_in_repo
    ]
    deleted_size = sum(
        gcloud_file_size(data_file, gcloud_path, fs) for data_file in data_files_to_delete
    )

    if len(data_files_to_delete):
        for data_file in logging.tqdm(
                data_files_to_delete,
                desc="Deleting unused files from dataset repository",
                total=len(data_files_to_delete),
                disable=not logging.is_progress_bar_enabled(),
        ):
            gcloud_delete_file(data_file, gcloud_path, fs)

    repo_files = list(set(files) - set(data_files_to_delete))

    return repo_id, split, uploaded_size, dataset_nbytes, repo_files, deleted_size, shards_path_in_repo
