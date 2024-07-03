"""langchain llm email test script"""
import os

from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, create_openai_functions_agent
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function

from colors import TColors

def main() -> None:
    """
    Main hook for testing the GmailToolkit

    Parameters:
        None

    Returns:
        None
    """
    # paste the OpenAI key into the key.txt file and put into the root directory
    try:
        with open(file="key.txt", mode="r", encoding="utf-8") as f:
            key = f.read().replace("\n", "")
            assert key != "", f"{TColors.FAIL}Key is empty.{TColors.ENDC}"

            os.environ["OPENAI_API_KEY"] = key
            print(f"{TColors.OKGREEN}OpenAI API key loaded.{TColors.ENDC}")

    except FileNotFoundError:
        print(f"{TColors.FAIL}Please paste your OpenAI API key into the key.txt "
                f"file and put it into the root directory.{TColors.ENDC}")


    # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
    # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
    credentials = get_gmail_credentials(
        token_file="gmail_token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="gmail_credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)

    model = OllamaFunctions(model="llama3", temperature=0)
    # model = ChatOpenAI(temperature=0)
    # Create prompts for the agents
    system_prompt = """
        Respond to the human as helpfully and accurately as possible. You have access to the following tools:

        {tools}

        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
        Make sure to use the tool input correctly with its specified formats and types. For example:
        $INPUT: {{
            "message": "text",
            "subject": "subject text"
            "to": ["person1", "person2", ...]
        }}
        Only append a single string to the message key. Not a dictionary or anything else.

        Valid "action" values: "Final Answer" or {tool_names}

        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{
        "action": $TOOL_NAME,
        "action_input": $INPUT
        }}
        ```

        Follow this format:

        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        ... (repeat Thought/Action/Observation N times)
        Thought: I know what to respond
        Action:
        ```
        {{
        "action": "Final Answer",
        "action_input": "Final response to human"
        }}

        Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}\n {agent_scratchpad}"),
        ]
    )
    model.bind_tools(
        [convert_to_openai_function(t) for t in toolkit.get_tools()], 
        tool_choice="any",
    )

    agent = create_structured_chat_agent(
        tools=toolkit.get_tools(),
        llm=model,
        prompt=prompt,
    )
    # instructions = """You are an assistant."""
    # base_prompt = hub.pull("langchain-ai/openai-functions-template")
    # prompt = base_prompt.partial(instructions=instructions)
    # agent = create_openai_functions_agent(model, toolkit.get_tools(), prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit.get_tools(),
        handle_parsing_errors=True,
        # This is set to False to prevent information about my email showing up on the screen
        # Normally, it is helpful to have it set to True however.
        verbose=True,
    )

    agent_executor.invoke(
        {
            "input": "Create a gmail draft for me with the subject 'Hello'" \
            " and the message 'World'. It should be sent to ''.",
        }
    )

if __name__ == "__main__":
    main()
