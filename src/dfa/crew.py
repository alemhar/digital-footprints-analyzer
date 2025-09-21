from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

try:
    # LangChain Azure OpenAI client used directly to avoid routing confusion
    from langchain_openai import AzureChatOpenAI  # type: ignore
except Exception:
    AzureChatOpenAI = None  # Will raise at runtime with a helpful error
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Dfa():
    """Dfa crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    def _azure_llm(self):
        """Return an AzureChatOpenAI client based on environment variables.

        Required envs:
        - AZURE_OPENAI_API_KEY
        - AZURE_OPENAI_ENDPOINT (e.g., https://<resource>.openai.azure.com/)
        - OPENAI_API_VERSION (e.g., 2023-05-15)
        - OPENAI_ENGINE (Azure deployment name, e.g., gpt-4)
        """
        if AzureChatOpenAI is None:
            raise RuntimeError(
                "langchain-openai is not installed. Run: pip install langchain-openai"
            )
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        version = os.getenv("OPENAI_API_VERSION")
        deployment = os.getenv("OPENAI_ENGINE")
        if not all([api_key, endpoint, version, deployment]):
            missing = [
                name for name, val in [
                    ("AZURE_OPENAI_API_KEY", api_key),
                    ("AZURE_OPENAI_ENDPOINT", endpoint),
                    ("OPENAI_API_VERSION", version),
                    ("OPENAI_ENGINE", deployment),
                ] if not val
            ]
            raise RuntimeError(
                f"Missing Azure settings: {', '.join(missing)}. Check your .env."
            )
        # Normalize endpoint without trailing slash
        endpoint = endpoint.rstrip("/")
        return AzureChatOpenAI(
            model=deployment,
            temperature=0.1,
            azure_endpoint=endpoint,
            openai_api_key=api_key,
            max_retries=0,
            request_timeout=60,
        )
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            llm=self._azure_llm(),
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
            verbose=True,
            llm=self._azure_llm(),
        )

    @agent
    def security_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['security_strategist'],  # type: ignore[index]
            verbose=True,
            llm=self._azure_llm(),
        )

    @task
    def strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['strategy_task'],  # type: ignore[index]
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        # Run after strategy_task in sequential process
        return Task(
            config=self.tasks_config['research_task'],  # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Dfa crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
