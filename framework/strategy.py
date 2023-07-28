"""library for strategy pattern implementations"""
from typing import Callable

from framework.utils import log_conversation
from framework.colors import TColors, ATTACK_NAMES
from framework.prompts import SECRET_KEY, SYSTEM_PROMPTS
from framework.api import ChatAPI

class Strategy:
    """Strategy pattern interface"""

    def __init__(self, attack_func: Callable, defense_func: Callable,
                 llm_type: str, temperature: float, max_level: int):
        self.attack_func: Callable = attack_func
        self.defense_func: Callable = defense_func
        self.llm_type: str = llm_type
        self.temperature: float = temperature
        self.max_level: int = max_level
        self.chat_api_add_messages: Callable = ChatAPI.add_message


    def execute(self) -> int:
        """Executes the strategy and returns the number of successes"""
        num_successes: int = 0
        print(f"{TColors.OKBLUE}{TColors.BOLD}>> Starting " \
              f"{ATTACK_NAMES[self.attack_func.__name__]} Attack{TColors.ENDC}")

        for level_id in range(0, self.max_level):
            sys_prompt = SYSTEM_PROMPTS[f"level_{level_id}"]

            prompt, response = self.attack_func(system_prompt=sys_prompt,
                                                llm_type=self.llm_type,
                                                temp=self.temperature,
                                                defense=self.defense_func)

            # call the chat api to add the messages to the chat
            self.chat_api_add_messages("system", sys_prompt)
            self.chat_api_add_messages("user", prompt)
            self.chat_api_add_messages("assistant", response)

            if level_id < 10:
                succ_result_str = f"{'SUCCESS':>8}"
                fail_result_str = f"{'FAIL':>8}"
            else:
                succ_result_str = f"{'SUCCESS':>7}"
                fail_result_str = f"{'FAIL':>7}"

            if self.__evaluate_response(response):
                print(f"Level {level_id}: {TColors.OKGREEN}{succ_result_str}{TColors.ENDC}")
                num_successes += 1
            else:
                print(f"Level {level_id}: {TColors.FAIL}{fail_result_str}{TColors.ENDC}")

            log_conversation(self.attack_func.__name__,
                             self.defense_func.__name__, level_id,
                             prompt, response, self.__evaluate_response(response))

        return num_successes


    def __evaluate_response(self, response: str) -> bool:
        """
        Evaluates the response of the opponent LLM

        Parameters:
            response: str - the response of the opponent LLM

        Returns:
            success: bool - whether the response contains the secret key or not
        """
        return SECRET_KEY in response
