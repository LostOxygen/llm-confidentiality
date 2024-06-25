"""agentfactory library"""
# code is inspired by:
# https://deci.ai/blog/how-to-run-langchain-benchmarks-with-local-llms-from-hugging-face/
from langchain_benchmarks.tool_usage.agents import apply_agent_executor_adapter
from langchain.agents import AgentExecutor, create_structured_chat_agent

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
        prompt: The prompt which is used for the LLM

    Usage:
        The AgentFactory is invoked to create a new agent. It initializes a new environment,
        sets up the LLM, and configures the agent with necessary tools and settings. The
        agent is then wrapped with an adapter to ensure compatibility with the dataset schema
        and to automatically include the state of the environment in the agent's response at
        the end of each run.
    """
    def __init__(self, task, llm, prompt) -> None:
        self.task = task
        self.llm = llm
        self.prompt = prompt
        print("AgentFactory hat geklappt und so")

    def __call__(self):
        env = self.task.create_environment()
        tools = env.tools
        agent = create_structured_chat_agent(
            tools=tools,
            llm=self.llm,
            prompt=self.prompt,
        )

        print("Agent erstellt")
        executor = AgentExecutor(
            agent=agent,
            tools=env.tools,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        print("Executor erstellt")
        print("Test Invoke: ", executor.invoke({"input": "hey how are you?"}))
        return apply_agent_executor_adapter(executor, state_reader=env.read_state)
