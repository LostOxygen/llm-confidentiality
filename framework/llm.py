"""library for LLM models, functions and helper stuff"""
import os
import torch
from openai import ChatCompletion
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LLM():
    """abstract implementation of a genereric LLM model"""
    def __init__(self, llm_type: str, temperature: float = 1.0):
        self.llm_type: str = llm_type
        self.temperature: float = temperature
        self.model: AutoModelForCausalLM = None
        self.tokenizer: AutoTokenizer = None

        # pre load the models and tokenizer and adjust the temperature
        match self.llm_type:
            case ("gpt-3.5-turbo" | "gpt-3.5-turbo-0301" |
                  "gpt-3.5-turbo-0613" | "gpt-4" | "gpt-4-0613"):
                self.temperature = max(0.0, min(self.temperature, 2.0))

            case ("llama2" | "llama2-7b" | "llama2-13b" | "llama2-70b"):
                self.temperature = max(0.01, min(self.temperature, 5.0))
                # create quantization config
                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )

                model_name = "meta-llama/"
                if self.llm_type.split("-")[1] == "7b":
                    model_name += "Llama-2-7b-chat-hf"
                elif self.llm_type.split("-")[1] == "13b":
                    model_name += "Llama-2-13b-chat-hf"
                elif self.llm_type.split("-")[1] == "70b":
                    model_name += "Llama-2-70b-chat-hf"
                else:
                    model_name += "Llama-2-70b-chat-hf"

                self.tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                token=os.environ["HF_TOKEN"],
                            )

                self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="auto",
                            quantization_config=config,
                            low_cpu_mem_usage=True,
                            token=os.environ["HF_TOKEN"],
                        )

            case "beluga2":
                self.temperature = max(0.01, min(self.temperature, 2.0))
                model_name = "stabilityai/StableBeluga2"
                # create quantization config
                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )

                self.tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                token=os.environ["HF_TOKEN"],
                            )

                self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="auto",
                            quantization_config=config,
                            low_cpu_mem_usage=True,
                            token=os.environ["HF_TOKEN"],
                        )


            case ("vicuna" | "vicuna-7b" | "vicuna-13b" | "vicuna-33b"):
                self.temperature = max(0.01, min(self.temperature, 2.0))

                model_name = "lmsys/"
                if self.llm_type.split("-")[1] == "7b":
                    model_name += "vicuna-7b-v1.3"
                elif self.llm_type.split("-")[1] == "13b":
                    model_name += "vicuna-13b-v1.3"
                elif self.llm_type.split("-")[1] == "33b":
                    model_name += "vicuna-33b-v1.3"
                else:
                    model_name += "lmsys/vicuna-33b-v1.3"

                # create quantization config
                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )

                self.tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                token=os.environ["HF_TOKEN"],
                            )

                self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="auto",
                            quantization_config=config,
                            low_cpu_mem_usage=True,
                            token=os.environ["HF_TOKEN"],
                        )
            case _:
                raise NotImplementedError(f"LLM type {self.llm_type} not implemented")


    def __del__(self):
        """Deconstructor for the LLM class"""
        del self.llm_type
        del self.temperature
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer


    @torch.inference_mode()
    def predict(self, system_prompt: str, user_prompt: str) -> str:
        """
        predicts a response for a given prompt input

        Parameters:
            system_prompt: str - the system prompt to initialize the LLM
            user_prompt: str - the user prompt for the LLM to respond on

        Returns:
            response: str - the LLMs' response
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        match self.llm_type:
            case ("gpt-3.5-turbo" | "gpt-3.5-turbo-0301" |
                  "gpt-3.5-turbo-0613" | "gpt-4" | "gpt-4-0613"):
                completion = ChatCompletion.create(model=self.llm_type,
                                                   messages=messages,
                                                   temperature=self.temperature)
                response = completion.choices[0].message.content

            case ("llama2" | "llama2-7b" | "llama2-13b" | "llama2-70b"):
                formatted_messages = f"""<s>[INST] <<SYS>>
                    {system_prompt}
                    <</SYS>>
                    {user_prompt}
                    <</INST>>
                """

                with torch.no_grad():
                    inputs = self.tokenizer(formatted_messages, return_tensors="pt").input_ids
                    inputs.to("cuda")
                    outputs = self.model.generate(inputs, do_sample=True,
                                                temperature=self.temperature,
                                                max_length=4096)
                    response = self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

                # remove the previous chat history from the response
                # so only the models' actual response remains
                response = response[0].replace(formatted_messages, "")


            case "beluga2":
                formatted_messages = f"""
                ### System:
                {system_prompt}

                ### User:
                {user_prompt}

                ### Assistant:\n
                """
                with torch.no_grad():
                    inputs = self.tokenizer(formatted_messages, return_tensors="pt").input_ids
                    inputs.to("cuda")
                    outputs = self.model.generate(inputs, do_sample=True,
                                                temperature=self.temperature,
                                                max_length=4096)
                    response = self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

            case ("vicuna" | "vicuna-7b" | "vicuna-13b" | "vicuna-33b"):
                formatted_messages = f"""
                {system_prompt}

                USER: {user_prompt}
                """

                with torch.no_grad():
                    inputs = self.tokenizer(formatted_messages, return_tensors="pt").input_ids
                    inputs.to("cuda")
                    outputs = self.model.generate(inputs, do_sample=True,
                                                temperature=self.temperature,
                                                max_length=4096)
                    response = self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)

                 # remove the previous chat history from the response
                # so only the models' actual response remains
                response = response[0].replace(formatted_messages, "")

            case _:
                raise NotImplementedError(f"LLM type {self.llm_type} not implemented")

        return response
