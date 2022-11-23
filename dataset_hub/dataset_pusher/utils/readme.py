import pprint
from dataclasses import dataclass
from typing import Optional, Union

from datasets import Dataset, DatasetDict


@dataclass
class MardownContent:
    level: int
    title: str
    text_content: Optional[str] = ""


@dataclass
class Readme:
    dataset_name: MardownContent = MardownContent(level=1, title="DATASET CARD")
    dataset_description: MardownContent = MardownContent(level=2, title="Dataset Description")
    dataset_summary: MardownContent = MardownContent(level=3, title="Dataset Summary",
                                                     text_content="[More Information Needed]")
    supported_tasks: MardownContent = MardownContent(level=3, title="Supported Tasks and Leaderboards",
                                                     text_content="[More Information Needed]")
    languages: MardownContent = MardownContent(level=3, title="Languages",
                                               text_content="[More Information Needed]")
    dataset_structure: MardownContent = MardownContent(level=2, title="Dataset Structure")
    data_instances: MardownContent = MardownContent(level=3, title="Data Instances",
                                                    text_content="[More Information Needed]")
    data_fields: MardownContent = MardownContent(level=3, title="Data Fields",
                                                 text_content="[More Information Needed]")
    columns: MardownContent = MardownContent(level=4, title="Columns",
                                             text_content="[More Information Needed]")
    sample: MardownContent = MardownContent(level=4, title="Sample",
                                            text_content="[More Information Needed]")
    data_splits: MardownContent = MardownContent(level=3, title="Data Splits",
                                                 text_content="[More Information Needed]")
    dataset_creation: MardownContent = MardownContent(level=2, title="Dataset Creation")
    curation_rationale: MardownContent = MardownContent(level=3, title="Curation Rationale",
                                                        text_content="[More Information Needed]")
    source_data: MardownContent = MardownContent(level=3, title="Source Data",
                                                 text_content="[More Information Needed]")
    annotations: MardownContent = MardownContent(level=3, title="Annotations",
                                                 text_content="[More Information Needed]")
    personal_info: MardownContent = MardownContent(level=3, title="Personal and Sensitive Information",
                                                   text_content="[More Information Needed]")
    considerations: MardownContent = MardownContent(level=2, title="Considerations for Using the dataset")
    biases: MardownContent = MardownContent(level=3, title="Discussion of Biases",
                                            text_content="[More Information Needed]")
    limitations: MardownContent = MardownContent(level=3, title="Other Known Limitations",
                                                 text_content="[More Information Needed]")
    additional_info: MardownContent = MardownContent(level=2, title="Additional Information")
    curators: MardownContent = MardownContent(level=3, title="Dataset Curators",
                                              text_content="[More Information Needed]")
    licensing: MardownContent = MardownContent(level=3, title="Licensing Information",
                                               text_content="[More Information Needed]")
    contributions: MardownContent = MardownContent(level=3, title="Contributions",
                                                   text_content="[More Information Needed]")

    def render(self) -> str:
        readme_text = ""
        for block in self.__dataclass_fields__:
            content: MardownContent = getattr(self, block)
            readme_text += f"{'#'*content.level} {content.title}"
            readme_text += "\n\n"
            readme_text += content.text_content
            readme_text += "\n\n"
        return readme_text

    @classmethod
    def from_dataset(cls, name: str, url: str,  dataset: Union[Dataset, DatasetDict]) -> 'Readme':
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
        readme = Readme()
        readme.dataset_name.title = name
        readme.dataset_summary.text_content = f"The data files can be found on the gcloud instance at this " \
                                              f"adress: {url} "
        first_split_name = list(dataset.data.keys())[0]
        readme.columns.text_content = " ".join([f"``{column}``" for
                                                column in dataset[first_split_name].column_names])
        readme.sample.text_content = "```\n" + pprint.pformat(dataset[first_split_name][0], indent=4, width=140)\
                                     + "\n```"

        data_splits = ["|split|number_of_rows|", "|:---:|:---:"]
        for split in dataset:
            data_splits.append(f"|{split}|{dataset[split].num_rows}|")
        readme.data_splits.text_content = "\n".join(data_splits)
        return readme

