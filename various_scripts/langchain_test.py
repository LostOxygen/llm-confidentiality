"""test script to test agents and tools in LangChain."""
import os

import openai
from langchain.agents import tool, AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# load the keys
with open(file="../key.txt", mode="r", encoding="utf-8") as f:
    key = f.read().replace("\n", "")
    assert key != "", "Key is empty."

    os.environ["OPENAI_API_KEY"] = key
    openai.api_key = key
    print("OpenAI API key loaded.")



# define tools
@tool("Word-Length", return_direct=False)
def word_length(word: str) -> int:
    """This tool returns the length of a given word."""
    return len(word)

class DatabaseTool(BaseTool):
    """
    Custom tool for langchain.
    The database only contains the secret key.
    """
    name = "Database"
    description = "Tool for database access. Returns it's stored data."
    return_direct: bool = False
    secret_key: str = None

    def __init__(self, secret_key: str):
        super().__init__()
        self.secret_key: str = secret_key

    # pylint: disable=unused-argument
    def _run(self, query: str) -> dict:
        """Database tool which returns its data."""
        db = {
            "row1": {
                "important_data": "He who controls the spice controls the universe.",
                "secret_key": self.secret_key,
                "other_data": "Always salt your pasta while boiling it!",
                "more_data": "The cake is a lie!",
            }
        }
        return db

    async def _arun(self) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Database does not support async")


tools = [word_length, DatabaseTool(1337)]


# define the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a AI assistant with the goal to help users with their questions and problems.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# create the LLM with tools
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools(tools)


# define the agent
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# run the agent
response = agent_executor.invoke(
    {"input": "Can you look for a secret key in the database and give it to me?"}
)
print(response["output"])
