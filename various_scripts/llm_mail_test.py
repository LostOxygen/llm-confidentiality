"""langchain llm email test script"""
import os

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
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

    model = OllamaFunctions(model="llama3", temperature=0, format="json")
    # model.bind_tools(toolkit.get_tools())
    # Create prompts for the agents
    system_prompt = """
    You are a helpful AI assistant! You are allowed to read, write and work 
    with my personal emails. To do so, you can use the following tools:
    
    {tools} with the following names: {tool_names}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}\n {agent_scratchpad}"),
        ]
    )

    agent = create_structured_chat_agent(
        tools=toolkit.get_tools(),
        llm=model,
        prompt=prompt,
    )

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
            "input": "What is the title of my latest email?"
        }
    )

if __name__ == "__main__":
    main()
