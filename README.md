# private-dataset-hub

Helper library to push and load your datasets to gcloud.

When you push a dataset using this library you will create a dataset on your **_huggingface hub_** space and you will securely
upload your data to the specified path in **_gcloud_**.

The data files are uploaded as parquet shard to minimize disk space and allow data streaming.

The link between the two is made by a dataset script that will be automatically uploaded to your huggingface space.
No sensitive data will be hosted on Huggingface hub.

You can now load this dataset as any other dataset using the `load_dataset` command from datasets.

Unfortunately, the dataset streaming option of load_dataset does not work when authorization is needed
to upload a dataset. Fortunately, the function `custom_load_dataset` provided fixes this issue.

## How to use

### uploading a dataset

First make sure you are authorized in gcloud and huggingface
``` bash
gcloud auth login
huggingface-cli login
```

``` python
from dataset_hub import push_to_hub_dataset
from datasets import load_dataset
dataset = load_dataset("lhoestq/demo1")
push_to_hub_dataset(dataset, repo_id="<organization>/<dataset_id>", gcloud_path="<bucket>/<dataset_id>", private=True)
```

### loading a dataset

First make sure you are authorized in gcloud and huggingface
``` bash
gcloud auth login
huggingface-cli login
```

``` python
from datasets import load_dataset
load_dataset("<organization>/<dataset_id>", use_auth_token=True)
```

If you need dataset streaming

``` python
from dataset_hub import custom_load_dataset
custom_load_dataset("<organization>/<dataset_id>", use_auth_token=True)
```
