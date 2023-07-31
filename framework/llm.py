"""library for LLM models, functions and helper stuff"""
import os
import torch
from openai import ChatCompletion
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM():
    """abstract implementation of a genereric LLM model"""
    def __init__(self, llm_type: str, temperature: float = 1.0):
        self.llm_type: str = llm_type
        self.temperature: float = temperature

        # pre load the models and tokenizer and adjust the temperature
        match self.llm_type:
            case ("gpt-3.5-turbo" | "gpt-3.5-turbo-0301" |
                  "gpt-3.5-turbo-0613" | "gpt-4" | "gpt-4-0613"):
                self.temperature = max(0.0, min(self.temperature, 2.0))

            case "llama2":
                self.temperature = max(0.1, min(self.temperature, 5.0))
                self.tokenizer = AutoTokenizer.from_pretrained(
                                "meta-llama/Llama-2-7b-chat-hf",
                                token=os.environ["HF_TOKEN"],
                            )

                self.model = AutoModelForCausalLM.from_pretrained(
                            "meta-llama/Llama-2-7b-chat-hf",
                            device_map="auto",
                            torch_dtype=torch.bfloat16,
                            load_in_8bit=True,
                            low_cpu_mem_usage=True,
                            token=os.environ["HF_TOKEN"],
                            max_memory=f"{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB"
                        )

            case "llama":
                raise NotImplementedError(f"LLM type {self.llm_type} not implemented")
            case _:
                raise NotImplementedError(f"LLM type {self.llm_type} not implemented")


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

            case "llama2":
                formatted_messages = f"""<s>[INST] <<SYS>>
                    {system_prompt}
                    <</SYS>>
                    {user_prompt}
                    <</INST>>
                """

                inputs = self.tokenizer.encode(formatted_messages, return_tensors="pt").to("cuda:0")
                outputs = self.model.generate(inputs, do_sample=True, temperature=self.temperature)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            case "llama":
                raise NotImplementedError(f"LLM type {self.llm_type} not implemented")
            case _:
                raise NotImplementedError(f"LLM type {self.llm_type} not implemented")

        return response
