"""library for creating the system prompt dataset"""
import os
import json
import random
from typing import Type, Final

from framework.prompts import SYSTEM_PROMPTS, SECRET_KEY
from framework.colors import TColors

DATA_PATH: Final[str] = "./datasets/"

class DatasetState():
    TRAIN = 0
    TEST = 1
    NEW = 2
    ADVERSARIAL = 3


class PromptDataset():
    """
    Implementation of a dataset for system prompts

    The default setting is to use the system prompts dataset for training
    but is_train = false can be passed to the constructor to use the dataset
    with testing data.
    """

    def __init__(self, state: Type[DatasetState] = DatasetState.TRAIN) -> None:
        self.state = state

        match self.state:
            case DatasetState.TRAIN:
                self.data_path: str = "./datasets/system_prompts_train.json"
            case DatasetState.TEST:
                self.data_path: str = "./datasets/system_prompts_test.json"
            case DatasetState.NEW:
                self.data_path: str = "./datasets/system_prompts_new.json"
            case DatasetState.ADVERSARIAL:
                self.data_path: str = "./datasets/system_prompts_advs.json"
            case _:
                raise ValueError(f"{TColors.FAIL}Invalid dataset state.{TColors.ENDC}")

        self.__init_path()
        # the actual dataset hold in memory
        self.data = self.__load_dataset()
        self.__initialize_dataset()


    def __len__(self) -> int:
        if self.data["0"] == "empty":
            return 0
        return len(self.data)


    def __init_path(self) -> None:
        """Creates the path and file for the dataset"""
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
        if not os.path.isfile(self.data_path):
            with open(self.data_path, "w", encoding="utf-8") as file:
                init_dict = {0: "empty"}
                json.dump(init_dict, file, ensure_ascii=False, indent=4)


    def __initialize_dataset(self) -> None:
        """
        Initializes the dataset
        This method is meant to be called only once when the dataset is created to
        push the existing SYSTEM_PROMPTS to the dataset
        """
        match self.state:
            case DatasetState.TRAIN:
                with open(self.data_path, "a", encoding="utf-8") as file:
                    if self.data["0"] == "empty":
                        json.dump(SYSTEM_PROMPTS, file, ensure_ascii=False, indent=4)
            case DatasetState.TEST:
                pass
            case DatasetState.NEW:
                pass
            case DatasetState.ADVERSARIAL:
                pass
            case _:
                raise ValueError(f"{TColors.FAIL}Invalid dataset state.{TColors.ENDC}")


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
        if self.data["0"] == "empty":
            self.data["0"] = prompt
        else:
            # check if prompt is already in the dataset
            for _, value in self.data.items():
                if prompt in value:
                    return
        
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


    def get_whole_dataset(self) -> dict:
        """Returns the whole json dataset as a dictionary, even without initializing the class"""
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"{TColors.FAIL}Couldn't find dataset.{TColors.ENDC}")
        with open(self.data_path, "r", encoding="utf-8") as file:
            return json.load(file)


class ResponseDataset():
    """Implementation of a dataset for secret leaking responses"""

    def __init__(self) -> None:
        self.data_path: str = "./datasets/leaking_responses.json"
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
        """
        Initializes the dataset
        This method is meant to be called only once when the dataset is created to
        initialize the dataset
        """
        tmp_data = {"0": f"The secret key is {SECRET_KEY}"}
        with open(self.data_path, "a", encoding="utf-8") as file:
            json.dump(tmp_data, file, ensure_ascii=False, indent=4)


    def __load_dataset(self) -> dict:
        """Loads the dataset from the file system"""
        with open(self.data_path, "r", encoding="utf-8") as file:
            return json.load(file)


    def __save_dataset(self) -> None:
        """Saves the dataset to the file system"""
        with open(self.data_path, "w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)


    def add_response(self, response: str) -> None:
        """Adds a response to the dataset"""
        self.data.update({str(len(self.data)): str(response)})
        self.__save_dataset()


    def get_first_response(self) -> str:
        """Returns the first response in the dataset"""
        return self.data["0"]


    def get_last_response(self) -> str:
        """Returns the last response in the dataset"""
        return self.data[str(len(self.data) - 1)]


    def get_random_response(self) -> str:
        """Returns a random response from the dataset"""
        return self.data[str(random.randint(0, len(self.data) - 1))]


    def get_response_at_idx(self, idx: int) -> str:
        """Returns the response at the given index"""
        return self.data[str(idx)]


    def get_whole_dataset(self) -> dict:
        """Returns the whole json dataset as a dictionary, even without initializing the class"""
        if not os.path.isfile(self.data_path):
            raise FileNotFoundError(f"{TColors.FAIL}Couldn't find dataset.{TColors.ENDC}")
        with open(self.data_path, "r", encoding="utf-8") as file:
            return json.load(file)


class AdvsTrainDataset():
    """
    Implementation of a in-memory dataset for adversarial training.
    This dataset will not be written to the file system but instead be kept and
    updated in memory.
    """

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.data)


    def add_prompt(self, prompt: str) -> None:
        """Adds a prompt to the dataset"""
        self.data.update({str(len(self.data)): str(prompt)})


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


    def reset_dataset(self) -> None:
        """Resets the dataset"""
        self.data = {}
