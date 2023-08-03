"""library for creating the system prompt dataset"""
import os
import json
import random

from framework.prompts import SYSTEM_PROMPTS

class PromptDataset():
    """Implementation of a dataset for system prompts"""

    def __init__(self) -> None:
        self.data_path: str = "./datasets/system_prompts.json"
        if not os.path.exists("./datasets"):
            os.mkdir("./datasets")
        if not os.path.isfile(self.data_path):
            with open(self.data_path, "w", encoding="utf-8"):
                pass
            self.__initialize_dataset()
        # the actual dataset hold in memory
        self.data = self.__load_dataset()

    def __len__(self) -> int:
        return len(self.data)


    def __initialize_dataset(self) -> None:
        """Initializes the dataset
        This method is meant to be called only once when the dataset is created to
        push the existing SYSTEM_PROMPTS to the dataset"""
        with open(self.data_path, "a", encoding="utf-8") as file:
            json.dump(SYSTEM_PROMPTS, file, ensure_ascii=False, indent=4)


    def __load_dataset(self) -> dict:
        """Loads the dataset from the file system"""
        with open(self.data_path, "r", encoding="utf-8") as file:
            return json.load(file)


    def __save_dataset(self) -> None:
        """Saves the dataset to the file system"""
        with open(self.data_path, "w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)


    def add_prompt(self, prompt: str) -> None:
        """Adds a prompt to the dataset"""
        self.data.update({str(len(self.data)): str(prompt)})
        self.__save_dataset()


    def get_first_prompt(self) -> str:
        """Returns the first prompt in the dataset"""
        return self.data["0"]


    def get_last_prompt(self) -> str:
        """Returns the last prompt in the dataset"""
        return self.data[str(len(self.data) - 1)]


    def get_random_prompt(self) -> str:
        """Returns a random prompt from the dataset"""
        return self.data[str(random.randint(0, len(self.data) - 1))]


    def get_prompt_at_idx(self, idx: int) -> str:
        """Returns the prompt at the given index"""
        return self.data[str(idx)]
