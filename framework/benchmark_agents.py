"""agentfactory library"""
# code is inspired by: 
# https://deci.ai/blog/how-to-run-langchain-benchmarks-with-local-llms-from-hugging-face/
from langchain_benchmarks.tool_usage.agents import apply_agent_executor_adapter
from langchain.agents import AgentType, initialize_agent

class AgentFactory:
    """
    A factory class for creating agents tailored to perform tasks within a specific environment.

    This class dynamically generates agents for each run, ensuring they operate in a fresh
    environment. The environment's state can change based on the agent's actions, necessitating
    a new instance for each run. The factory uses a specific model for the Large Language Model
    and integrates tools and adapters to ensure that the agent's inputs and outputs align with
    the required dataset schema.

    Attributes:
        task: The task for which the agent is being created. This determines the type of
              environment the agent will operate in.
        llm:  The Large Language Model (LLM) used by the agent.

    Usage:
        The AgentFactory is invoked to create a new agent. It initializes a new environment,
        sets up the LLM, and configures the agent with necessary tools and settings. The
        agent is then wrapped with an adapter to ensure compatibility with the dataset schema
        and to automatically include the state of the environment in the agent's response at
        the end of each run.
    """
    def __init__(self, task, llm) -> None:
        self.task = task
        self.llm = llm

    def __call__(self):
        env = self.task.create_environment()
        tools = env.tools
        instructions = self.task.instructions
        agent_executor = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            verbose=True,
            agent_kwargs={
                "prefix": "### System: You are an AI assistant that follows instruction extremely well. Help as much as you can. "  +  instructions + " Use one of the following tools to take action:",
                "suffix": f"Remember: {instructions} ALWAYS respond with the following format: \nAction: \n```$JSON_BLOB```\n \nObservation:\nThought:\n",
                "human_message_template": "### User: Use the correct tool to answer the following question: {input}\n{agent_scratchpad}\n### Assistant:"
            }
        )
        return apply_agent_executor_adapter(agent_executor, state_reader=env.read_state)
