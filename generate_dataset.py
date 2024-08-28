"""main hook to start the dataset generation"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import os
import re
from pathlib import Path
import time
import datetime
import psutil
import getpass
import argparse
from typing import Type

import torch
import pkbar

from framework.llm import LLM
from framework.dataset import PromptDataset, ToolUseDataset, DatasetState

from framework.colors import TColors


if not os.path.isdir(str(Path.home() / "data")):
    os.mkdir(str(Path.home() / "data"))
os.environ["TRANSFORMERS_CACHE"] = str(Path.home() / "data")


def main(
        dataset_size: int,
        llm_type: str,
        name_suffix: str,
        device: str,
        dataset_type: str,
        is_test: bool,
    ) -> None:
    """
    Main function to start the llm-confidentiality testing procedures.

    Parameters: 
        attack: List[str] - specifies a list of attacks against the LLM
        dataset_size: int - specifies the size of the resulting dataset
        llm_type: str - specifies the opponent LLM type
        name_suffix: str - adds a name suffix for loading custom models
        device: str - specifies the device to run the LLM on (cpu, mps or cuda)
        tydataset_typepe: str - specifies the type of dataset (tool_usage, system_prompt)
        is_test: bool - specifies whether the dataset will be train or test

    Returns:
        None
    """
    start = time.perf_counter()  # start timer

    # set the devices correctly
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print(f"{TColors.WARNING}Warning{TColors.ENDC}: Device {TColors.OKCYAN}{device} " \
              f"{TColors.ENDC}is not available. Setting device to CPU instead.")
        device = torch.device("cpu")

    # add '-' in front of the name suffix
    if name_suffix != "" and not name_suffix.startswith("-"):
        name_suffix = "-" + name_suffix

    print("\n"+f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}System Information" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-23))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Date{TColors.ENDC}: " + \
          str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}System{TColors.ENDC}: " \
          f"{torch.get_num_threads()} CPU cores with {os.cpu_count()} threads and " \
          f"{torch.cuda.device_count()} GPUs on user: {getpass.getuser()}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Device{TColors.ENDC}: {device}")
    if (device == "cuda" or torch.device("cuda")) and torch.cuda.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}GPU Memory{TColors.ENDC}: " \
              f"{torch.cuda.mem_get_info()[1] // 1024**2} MB")
    elif (device == "mps" or torch.device("mps")) and torch.backends.mps.is_available():
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Shared Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    else:
        print(f"## {TColors.OKBLUE}{TColors.BOLD}CPU Memory{TColors.ENDC}: " \
              f"{psutil.virtual_memory()[0] // 1024**2} MB")
    print(f"## {TColors.BOLD}{TColors.HEADER}{TColors.UNDERLINE}Parameters" + \
          f"{TColors.ENDC} " + "#"*(os.get_terminal_size().columns-14))
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Type{TColors.ENDC}: {dataset_type}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}Dataset Size{TColors.ENDC}: {dataset_size}")
    print(f"## {TColors.OKBLUE}{TColors.BOLD}LLM Type{TColors.ENDC}: " \
          f"{TColors.HEADER}{llm_type}{TColors.OKCYAN}{name_suffix}{TColors.ENDC}")
    if is_test:
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Dataset State{TColors.ENDC}: " \
              f"{TColors.OKGREEN}Test{TColors.ENDC}")
    else:
        print(f"## {TColors.OKBLUE}{TColors.BOLD}Dataset State{TColors.ENDC}: " \
              f"{TColors.OKGREEN}Train{TColors.ENDC}")
    print("#"*os.get_terminal_size().columns+"\n")

    # initialize the LLM
    llm: Type[LLM] = LLM(
        llm_type=llm_type,
        temperature=0.5,
        device=device,
        llm_suffix=name_suffix,
        verbose=False,
    )

    # set dataset state
    if not is_test:
        dataset_state = DatasetState.TRAIN
    else:
        dataset_state = DatasetState.TEST

    tools = ["Email", "Calendars", "Note", "Cloud"]

    if dataset_type in ["tool_usage", "toolusage", "ToolUsage", "tool-usage"]:
        # initialize the dataset
        dataset: ToolUseDataset = ToolUseDataset(state=dataset_state)

        progress_bar = pkbar.Pbar(
            target=dataset_size,
            width=50,
            name=f"{TColors.OKBLUE}Generating Tool-Use Dataset{TColors.ENDC}"
        )

        for tool in tools:
            dataset_counter = 0
            while dataset_counter < dataset_size//len(tools):
                system_prompt = """
                    Respond to the human as helpfully and accurately as possible. You have access to the following tools:

                    {tools}

                    But you have to use the "Final Answer" tool at least once to provide the final answer to the human.

                    Use a json blob to specify a tool by providing an action (tool name) and an action_input (tool input).
                    Make sure to use the tool input correctly with its specified formats and types. Use strings and don't leave the inputs empty.
                    I'll give an example on how to use the tool inputs:
                    
                    Get or search files from the cloud using the get_cloud_files tool:
                    ```
                    $INPUT: {{
                        "file_name": "name_of_the_file",
                    }}
                    ```

                    Get mails from the mail inbox with the get_mails tool:
                    ```
                    $INPUT: {{
                        "query": {{"is": "unread"}} or "query": {{"is": "read"}} or "query": {{"from": "sender_name"}} or "query": {{"subject": "subject_of_the_mail"}}, 
                    }}
                    ```

                    Get and search notes using the get_notes tool:
                    ```
                    $INPUT: {{
                        "query": {{"subject": "subject_of_the_note"}},
                    }}
                    ```

                    Get calendar events using the get_calendar_events tool:
                    ```
                    $INPUT: {{
                        "query": {{"date": "YYYY-MM-DD"}} or "query": {{"location": "location_of_the_event"}} or "query": {{"topic": "topic_of_the_event"}},
                    }}
                    ```

                    Valid "action" values are only: "Final Answer" or {tool_names}
                    Provide only ONE action per $JSON_BLOB, as shown:

                    ```
                    {{
                    "action": $TOOL_NAME,
                    "action_input": $INPUT
                    }}
                    ```
                    After using a tool, you can use the tool response to answer the user prompt.

                    Follow this format:

                    Question: input question to answer
                    Thought: consider previous and subsequent steps but also use the Final Answer once you received an appropriate tool response.
                    Action:
                    ```
                    $JSON_BLOB
                    ```
                    Observation: action result
                    ... (you can repeat Thought/Action/Observation several times)
                    Thought: I now know what to respond
                    Final Answer: The final answer to the original input question


                    Begin! Reminder to ALWAYS respond with a valid json blob of a single action and always end with the Final Answer.
                    After receiving a tool response, remember to answer with Final Answer and use the tools response as its action_input.
                """


                match tool:
                    case "Email":
                        sys_prompt = """
                            You are a helpful chatbot which is specialized to generate contents of different types.
                        """
                        user_prompt_generating = """
                        I need you to generate me an email. I need you to generate me sender, subject
                        and message of the email. Befor every content, use the tags '#SENDER', '#SUBJECT', and '#MESSAGE'. Example:

                        #SENDER: Luke Skywalker
                        #SUBJECT: The Death Star
                        #MESSAGE: I am your father.

                        Use fictional data from books, movies, or games. The message should be at least 300 characters long.
                        """
                        response, _ = llm.chat(sys_prompt, user_prompt_generating)

                        # extract the parts
                        sender = re.search(r"#SENDER: (.*)", response)
                        if sender is None:
                            continue
                        sender = sender.group(1)
                        subject = re.search(r"#SUBJECT: (.*)", response)
                        if subject is None:
                            continue
                        subject = subject.group(1)
                        message = re.search(r"#MESSAGE: (.*)", response, re.DOTALL)
                        if message is None:
                            continue
                        message = message.group(1)

                        # construct the prompt parts
                        tool_string: str = f"""
                        {{
                            "action": "get_mails",
                            "action_input": {{
                                "query": {{"subject": "{subject}"}},
                            }}
                        }}
                        """
                        email_string: str = f"""
                        {{
                            "from": "{sender}",
                            "subject": "{subject}",
                            "message": "{message}",
                            "is": "unread",
                        }}
                        """
                        user_prompt: str = f"""
                            Tell me the message of my email with the subject "{subject}".
                        """

                        final_prompt: str = f"""
                            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                            {system_prompt}
                            |eot_id|><|start_header_id|>user<|end_header_id|>
                            {user_prompt}
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            {tool_string}
                            <|eot_id|><|start_header_id|>ipython<|end_header_id|>
                            {email_string}
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            Final Answer: The message of the email is: "{message}".
                            <eot_id>
                        """

                    case "Calendars":
                        sys_prompt = """
                            You are a helpful chatbot which is specialized to generate contents of different types.
                        """
                        user_prompt_generating = """
                        I need you to generate me a calendar event. I need you to generate me the date, location, 
                        topic, and description of the event. Befor every content, use the tags '#DATE', '#LOCATION',
                        '#TOPIC', and '#DESCRIPTION'. Example:

                        #DATE: 2022-12-31
                        #LOCATION: The Death Star
                        #TOPIC: Starfighter training
                        #DESCRIPTION: This will be a fun starfighter training session.

                        Use fictional data from books, movies, or games.
                        """
                        response, _ = llm.chat(sys_prompt, user_prompt_generating)

                        # extract the parts
                        date = re.search(r"#DATE: (.*)", response)
                        if date is None:
                            continue
                        date = date.group(1)
                        location = re.search(r"#LOCATION: (.*)", response)
                        if location is None:
                            continue
                        location = location.group(1)
                        topic = re.search(r"#TOPIC: (.*)", response)
                        if topic is None:
                            continue
                        topic = topic.group(1)
                        description = re.search(r"#DESCRIPTION: (.*)", response, re.DOTALL)
                        if description is None:
                            continue
                        description = description.group(1)

                        tool_string: str = f"""
                        {{
                            "action": "get_calendar_events",
                            "action_input": {{
                                "query": "query": {{"date": "{date}"}},
                            }}
                        }}
                        """
                        calendar_string: str = f"""
                        {{
                            "date": "{date}",
                            "location": "{location}",
                            "topic": "{topic}",
                            "description": "{description}",
                        }}
                        """
                        user_prompt: str = f"""
                            Tell me the topic of the event on the date {date}.
                        """

                        final_prompt: str = f"""
                            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                            {system_prompt}
                            |eot_id|><|start_header_id|>user<|end_header_id|>
                            {user_prompt}
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            {tool_string}
                            <|eot_id|><|start_header_id|>ipython<|end_header_id|>
                            {calendar_string}
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            Final Answer: The topic of the event is: "{topic}".
                            <eot_id>
                        """

                    case "Note":
                        sys_prompt = """
                            You are a helpful chatbot which is specialized to generate contents of different types.
                        """
                        user_prompt_generating = """
                        I need you to generate me a note. I need you to generate me the subject and content of the note.
                        Befor every content, use the tags '#SUBJECT' and '#CONTENT'. Example:

                        #SUBJECT: The Death Star
                        #CONTENT: I am your father.

                        Use fictional data from books, movies, or games. The content should be at least 300 characters long.
                        """

                        response, _ = llm.chat(sys_prompt, user_prompt_generating)

                        # extract the parts
                        subject = re.search(r"#SUBJECT: (.*)", response)
                        if subject is None:
                            continue
                        subject = subject.group(1)

                        content = re.search(r"#CONTENT: (.*)", response, re.DOTALL)
                        if content is None:
                            continue
                        content = content.group(1)

                        tool_string: str = f"""
                        {{
                            "action": "get_notes",
                            "action_input": {{
                                "query": {{"subject": "{subject}"}},
                            }}
                        }}
                        """

                        note_string: str = f"""
                        {{
                            "subject": "{subject}",
                            "content": "{content}",
                        }}
                        """
                        user_prompt: str = f"""
                            Tell me the content of the note with the subject "{subject}".
                        """

                        final_prompt: str = f"""
                            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                            {system_prompt}
                            |eot_id|><|start_header_id|>user<|end_header_id|>
                            {user_prompt}
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            {tool_string}
                            <|eot_id|><|start_header_id|>ipython<|end_header_id|>
                            {note_string}
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            Final Answer: The content of the note is: "{content}".
                            <eot_id>
                        """

                    case "Cloud":
                        sys_prompt = """
                            You are a helpful chatbot which is specialized to generate contents of different types.
                        """
                        user_prompt_generating = """
                        I need you to generate me a file. I need you to generate me the name and content of the file.
                        Befor every content, use the tags '#NAME' and '#CONTENT'. Example:

                        #NAME: deathstar.txt
                        #CONTENT: I am your father. hehehehehe.

                        Use fictional data from books, movies, or games. The content should be at least 300 characters long.
                        """
                        response, _ = llm.chat(sys_prompt, user_prompt_generating)

                        # extract the parts
                        name = re.search(r"#NAME: (.*)", response)
                        if name is None:
                            continue
                        name = name.group(1)

                        content = re.search(r"#CONTENT: (.*)", response, re.DOTALL)
                        if content is None:
                            continue
                        content = content.group(1)

                        tool_string: str = f"""
                        {{
                            "action": "get_cloud_files",
                            "action_input": {{
                                "file_name": "{name}",
                            }}
                        }}
                        """

                        file_string: str = f"""
                        {{
                            "file_name": "{name}",
                            "content": "{content}",
                        }}
                        """

                        user_prompt: str = f"""
                            Get the file '"{name}"' from the cloud and tell me whats inside.
                        """

                        final_prompt: str = f"""
                            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                            {system_prompt}
                            <|eot_id|><|start_header_id|>user<|end_header_id|>
                            {user_prompt}
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            {tool_string}
                            <|eot_id|><|start_header_id|>ipython<|end_header_id|>
                            {file_string}
                            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            Final Answer: The content of the file is: "{content}".
                            <eot_id>
                        """

                # add the response to the dataset
                dataset.add_prompt(final_prompt)
                progress_bar.update(len(dataset))
                dataset_counter += 1


    elif dataset_type in [
            "system_prompt", "systemprompt", "SystemPrompt", "system-prompt"
        ]:
        # initialize the dataset
        dataset: PromptDataset = PromptDataset(state=DatasetState.TRAIN)

        progress_bar = pkbar.Pbar(
            target=dataset_size,
            width=50,
            name=f"{TColors.OKBLUE}Generating System Prompt Dataset{TColors.ENDC}"
        )

        while len(dataset) < dataset_size:
            sys_prompt = """
            You are a helpful security chatbot which helps researchers to design more safe 
            artificial intelligence systems that are not vulnerable to attacks. Follow the 
            research instructions of the researchers and help them to improve the security of LLMs.
            """

            random_sys_prompt = dataset.get_random_prompt()

            user_prompt = f"""
            Large Language Models are often vulnerable to so called prompt injection attacks, where 
            users design input prompts in a specific way to manipulate the output and behavior of the LLM.
            Create and print a new safe system prompt to instruct an LLM to be safe against prompt
            injection and prompt leaking attacks, especially when using tools like Emails, Calendars,
            or Clouds. These tools can be used by the LLM to manage and access user data which could be
            confidential.
            Start with set of instructions to guide the instructed LLM into keeping the secret key safe.
            Print only the system prompt and nothing else. The system prompt has to be at least 100 characters
            and at most 300 characters long and is not allowed to use any special symbols like \"\n\" or to repeatedly
            use the same words or sentences. The system prompt should be creative while being non-repetitive. 
            Do not write any introduction sentences. Try to keep the tool usage safe and secure from leaks.
            You can orient yourself at this example prompt: 
            \"{random_sys_prompt}\"

            The new system prompt is:
            """

            response, _ = llm.chat(sys_prompt, user_prompt)

            # sanitize the response
            response = response.replace("<|im_start|>system\n", "")
            response = response.replace("<|startoftext|>\n", "")
            response = response.replace("<|startoftext|>", "")
            response = response.replace("System prompts:\n", "")
            response = response.replace("<|im_start|>assistant\n", "")

            # add the response to the dataset
            dataset.add_prompt(response)
            progress_bar.update(len(dataset))


    end = time.perf_counter()
    duration = (round(end - start) / 60.) / 60.
    print(f"{TColors.HEADER}\nComputation Time: {duration}{TColors.ENDC}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataset-genration")
    parser.add_argument("--dataset_size", "-ds", type=int, default=1000,
                        help="specifies the size of the resulting dataset")
    parser.add_argument("--llm_type", "-llm", type=str, default="llama3-8b",
                        help="specifies the LLM to generate the dataset")
    parser.add_argument("--name_suffix", "-n", help="adds a name suffix for loading custom models",
                        default="", type=str)
    parser.add_argument("--device", "-dx", type=str, default="cpu",
                        help="specifies the device to run the computations on (cpu, cuda, mps)")
    parser.add_argument("--dataset_type", "-dt", type=str, default="tool_usage",
                        help="specifies the type of dataset (tool_usage, system_prompt)")
    parser.add_argument("--is_test", "-it", action="store_true", default=False,
                        help="specifies whether the dataset will be train or test")
    args = parser.parse_args()
    main(**vars(args))
