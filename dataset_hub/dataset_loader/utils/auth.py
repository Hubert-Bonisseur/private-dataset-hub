from typing import Optional, Union

from datasets import config
from huggingface_hub import HfFolder
from gcsfs.credentials import GoogleCredentials

from dataset_hub.config import GCS_DEFAULT_PROJECT, GCS_TOKEN


def get_authentication_headers_for_url(url: str, use_auth_token: Optional[Union[str, bool]] = None) -> dict:
    """Handle the HF authentication"""
    headers = {}
    if url.startswith(config.HF_ENDPOINT):
        if use_auth_token is False:
            token = None
        elif isinstance(use_auth_token, str):
            token = use_auth_token
        else:
            token = HfFolder.get_token()
    elif url.startswith("https://storage.googleapis.com"):
        credentials = GoogleCredentials(GCS_DEFAULT_PROJECT, "read_only", GCS_TOKEN)
        credentials.maybe_refresh()
        token = credentials.credentials.token
    else:
        token = None
    if token:
        headers["authorization"] = f"Bearer {token}"
    return headers
