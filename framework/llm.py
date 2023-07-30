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
            case ("gpt-3.5-turbo" | "gpt-3.5-turbo-0301" | "gpt-4"):
                completion = ChatCompletion.create(model=self.llm_type,
                                                   messages=messages,
                                                   temperature=self.temperature)
                response = completion.choices[0].message.content

            case "llama2":
                tokenizer = AutoTokenizer.from_pretrained(
                                "meta-llama/Llama-2-7b-chat-hf",
                                token=os.environ["HF_TOKEN"],
                            )

                model = AutoModelForCausalLM.from_pretrained(
                            "meta-llama/Llama-2-7b-chat-hf",
                            torch_dtype=torch.bfloat16,
                            load_in_8bit=True,
                            low_cpu_mem_usage=True,
                            token=os.environ["HF_TOKEN"],
                        )

                formatted_messages = f"""<s>[INST] <<SYS>>
                    {system_prompt}
                    <</SYS>>
                    {user_prompt}
                    <</INST>>
                """

                inputs = tokenizer.encode(formatted_messages, return_tensors="pt")
                outputs = model.generate(inputs, do_sample=True, temperature=self.temperature)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            case "llama":
                raise NotImplementedError(f"LLM type {self.llm_type} not implemented")
            case _:
                raise NotImplementedError(f"LLM type {self.llm_type} not implemented")

        return response
