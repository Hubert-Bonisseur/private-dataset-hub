import re
from typing import Optional, Dict, List
import pkg_resources
import pprint
import os


def get_dataset_script(gcloud_path: str, dataset_name: str, data_files_path: Dict[str, List[str]],
                       dataset_description: Optional[str]):
    this_dir, _ = os.path.split(__file__)
    dataset_script_path = os.path.join(this_dir, "../../../dataset_scripts", "default_dataset_script.py")
    with open(dataset_script_path, "r") as f:
        dataset_script = f.read()
    gcloud_path = gcloud_path.split("//")[-1]
    dataset_script = dataset_script.replace("[DATASET_PATH]", gcloud_path)
    dataset_script = dataset_script.replace("[DATASET_NAME]", dataset_name)
    dataset_script = dataset_script.replace("[DATA_FILES_DICT]", "\n" + pprint.pformat(data_files_path))
    if dataset_description:
        dataset_script = dataset_script.replace("[DATASET_DESCRIPTION]", dataset_description)
    # remove dev0 from version (causes crash when loading the dataset)
    version = re.findall(r"\d\.\d\.\d", pkg_resources.get_distribution("datasets").version)[0]
    dataset_script = dataset_script.replace("[DATASETS_PACKAGE_VERSION]", version)

    return dataset_script
