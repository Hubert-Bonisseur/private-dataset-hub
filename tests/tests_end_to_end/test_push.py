import unittest

from huggingface_hub.utils import RepositoryNotFoundError
from dataset_hub import push_to_hub_dataset
from datasets import load_dataset
import gcsfs
from huggingface_hub import HfApi
from dataset_hub.config import GCS_DEFAULT_PROJECT, GCS_TOKEN


class TestPush(unittest.TestCase):
    def setUp(self) -> None:
        self.gloud_path = "data-science-e2e/test-ci"
        self.repo_id = "test-ci"
        fs = gcsfs.GCSFileSystem(project=GCS_DEFAULT_PROJECT, token=GCS_TOKEN)
        try:
            HfApi().delete_repo(self.repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            pass
        try:
            fs.rm(self.gloud_path, recursive=True)
        except FileNotFoundError:
            pass

    def test_push_for_first_time_and_on_existing(self):
        dataset = load_dataset("lhoestq/demo1")
        push_to_hub_dataset(dataset, repo_id=self.repo_id,
                            project=GCS_DEFAULT_PROJECT, gcloud_token=GCS_TOKEN,
                            gcloud_path=self.gloud_path,
                            private=True)
        push_to_hub_dataset(dataset, repo_id=self.repo_id,
                            project=GCS_DEFAULT_PROJECT, gcloud_token=GCS_TOKEN,
                            gcloud_path=self.gloud_path,
                            private=True)

    def tearDown(self) -> None:
        fs = gcsfs.GCSFileSystem(token=GCS_TOKEN)
        HfApi().delete_repo(self.repo_id, repo_type="dataset")
        fs.rm(self.gloud_path, recursive=True)
