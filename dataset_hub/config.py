import os

GCS_TOKEN = os.environ.get("GCS_TOKEN")
GCS_DEFAULT_PROJECT = os.environ.get("GCSFS_DEFAULT_PROJECT", "")