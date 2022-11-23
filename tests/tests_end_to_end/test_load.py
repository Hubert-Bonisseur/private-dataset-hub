import os
import shutil
import unittest

from dataset_hub import custom_load_dataset, push_to_hub_dataset
from datasets import load_dataset
import gcsfs

from dataset_hub.config import GCS_DEFAULT_PROJECT, GCS_TOKEN


class TestLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_id = "TestingEnvironment/test-ci-load"
        self.cache_dir = "tests_cache"
        fs = gcsfs.GCSFileSystem(project=GCS_DEFAULT_PROJECT, token=GCS_TOKEN)
        if not fs.exists("data-science-e2e/test-ci-load"):
            dataset = load_dataset("lhoestq/demo1", cache_dir=self.cache_dir)
            push_to_hub_dataset(dataset, repo_id=self.repo_id,
                                       gcloud_path="data-science-e2e/test-ci-load",
                                       gcloud_token=GCS_TOKEN,
                                       private=True)
        if os.path.exists(self.cache_dir) and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.cache_dir)

    def test_load_dataset(self):
        dataset = custom_load_dataset(self.repo_id, use_auth_token=True,
                               cache_dir=self.cache_dir)
        for _ in iter(dataset["train"]):
            pass

    def test_custom_load_dataset(self):
        dataset = custom_load_dataset(self.repo_id, use_auth_token=True,
                                      cache_dir=self.cache_dir)
        for _ in iter(dataset["train"]):
            pass

    def test_custom_load_dataset_streaming(self):
        dataset = custom_load_dataset(self.repo_id, use_auth_token=True, streaming=True,
                                      cache_dir=self.cache_dir)
        for _ in iter(dataset["train"]):
            pass

    def test_reuse_cached_dataset_custom_custom(self):
        with self.assertLogs(level='WARNING') as logs:
            dataset = custom_load_dataset(self.repo_id, use_auth_token=True,
                                          cache_dir=self.cache_dir)
            dataset = custom_load_dataset(self.repo_id, use_auth_token=True,
                                          cache_dir=self.cache_dir)
            self.assertIn("Found cached dataset", ''.join(logs.output))

    def test_reuse_cached_dataset_custom_hf(self):
        with self.assertLogs(level='WARNING') as logs:
            dataset = custom_load_dataset(self.repo_id, use_auth_token=True,
                                          cache_dir=self.cache_dir)
            dataset = load_dataset(self.repo_id, use_auth_token=True,
                                   cache_dir=self.cache_dir)
            self.assertIn("Found cached dataset", ''.join(logs.output))

    def test_reuse_cached_dataset_hf_custom(self):
        with self.assertLogs(level='WARNING') as logs:
            dataset = load_dataset(self.repo_id, use_auth_token=True,
                                   cache_dir=self.cache_dir)
            dataset = custom_load_dataset(self.repo_id, use_auth_token=True,
                                          cache_dir=self.cache_dir)
            self.assertIn("Found cached dataset", ''.join(logs.output))
