import os
from typing import List
from gcsfs import GCSFileSystem


def gcloud_list_repo_files(path: str, fs: GCSFileSystem) -> List[str]:
    end_files = []
    for root, subdirs, files in fs.walk(path):
        for file in files:
            end_files.append(os.path.relpath(os.path.join(root, file), path))
    return end_files


def gcloud_upload_filebytes(filebytes: bytes, path_in_folder: str, gcloud_path: str, fs: GCSFileSystem):
    with fs.open(os.path.join(gcloud_path, path_in_folder), "wb") as f:
        f.write(filebytes)


def gcloud_file_size(path_in_folder: str, gcloud_path: str, fs: GCSFileSystem):
    return fs.size(os.path.join(gcloud_path, path_in_folder))


def gcloud_delete_file(path_in_folder: str, gcloud_path: str, fs: GCSFileSystem):
    fs.delete(os.path.join(gcloud_path, path_in_folder))
