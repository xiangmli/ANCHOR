import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from loguru import logger
from langchain.prompts import PromptTemplate

from llms import BaseLLM, AnyOpenAILLM, OpenSourceLLM
from tools import TOOL_MAP, Tool
from util import run_once, format_history, read_prompts


class BasePsychologicalAgent(ABC):
    """
    Base class for all psychological agents in the recommendation system.
    Combines essential functionality from Agent and ToolAgent while adding
    recommendation-specific features.
    """

    def __init__(
            self,
            prompt_config: Optional[str] = None,
            web_demo: bool = False,
            system: Optional['System'] = None,
            dataset: Optional[str] = None,
            *args,
            **kwargs
    ):
        """
        Initialize the psychological agent.

        Args:
            prompts: Dictionary of prompts for the agent
            prompt_config: Path to the prompt config file
            web_demo: Whether the agent is used in a web demo
            system: The system that the agent belongs to
            dataset: The dataset that the agent is used on
        """
        # Basic agent properties
        self.json_mode: bool = False
        self.system = system
        self.web_demo = web_demo
        self.dataset = dataset

        # Load prompts
        if prompt_config is not None:
            prompts = read_prompts(prompt_config)
        self.prompts = prompts

        # Partial task_type if system exists
        if self.system is not None:
            for prompt_name, prompt_template in self.prompts.items():
                if isinstance(prompt_template, PromptTemplate) and 'task_type' in prompt_template.input_variables:
                    self.prompts[prompt_name] = prompt_template.partial(task_type=self.system.task_type)

        if self.web_demo:
            assert self.system is not None, 'System not found.'

        # Tool-related properties
        self.tools: Dict[str, Tool] = {}
        self._history: List = []
        self.max_turns: int = 6
        self.finished: bool = False
        self.results: Any = None

        # Recommendation-specific properties
        self.user_history: List[str] = []
        self.user_id: Optional[str] = None
        self.is_initialized: bool = False

    def observation(self, message: str, log_head: str = '') -> None:
        """
        Log the message.

        Args:
            message: The message to log
            log_head: The log head
        """
        if self.web_demo:
            self.system.log(log_head + message, agent=self)
        else:
            logger.debug(f'Observation: {message}')

    def get_LLM(self, config_path: Optional[str] = None, config: Optional[dict] = None) -> BaseLLM:
        """
        Get the base large language model for the agent.

        Args:
            config_path: Path to the config file of the LLM
            config: The config of the LLM

        Returns:
            BaseLLM instance
        """
        if config is None:
            assert config_path is not None
            with open(config_path, 'r') as f:
                config = json.load(f)
        config = config.copy()
        model_type = config['model_type']
        del config['model_type']
        if model_type != 'api':
            return OpenSourceLLM(**config)
        else:
            return AnyOpenAILLM(**config)

    @run_once
    def validate_tools(self) -> None:
        """
        Validate the tools required by the agent.
        """
        required_tools = self.required_tools()
        for tool, tool_type in required_tools.items():
            assert tool in self.tools, f'Tool {tool} not found.'
            assert isinstance(self.tools[tool], tool_type), f'Tool {tool} must be an instance of {tool_type}.'

    @staticmethod
    @abstractmethod
    def required_tools() -> Dict[str, type]:
        """
        The required tools for the agent.

        Returns:
            Dictionary of tool names and their types
        """
        raise NotImplementedError("BasePsychologicalAgent.required_tools() not implemented")

    def get_tools(self, tool_config: Dict[str, dict]) -> None:
        """
        Initialize tools from configuration.

        Args:
            tool_config: Dictionary of tool configurations
        """
        assert isinstance(tool_config, dict), 'Tool config must be a dictionary.'
        for tool_name, tool in tool_config.items():
            assert isinstance(tool, dict), 'Config of each tool must be a dictionary.'
            assert 'type' in tool, 'Tool type not found.'
            assert 'config_path' in tool, 'Tool config path not found.'
            tool_type = tool['type']
            if tool_type not in TOOL_MAP:
                raise NotImplementedError(f'Tool type {tool_type} not implemented.')
            config_path = tool['config_path']
            if self.dataset is not None:
                config_path = config_path.format(dataset=self.dataset)
            self.tools[tool_name] = TOOL_MAP[tool_type](config_path=config_path)

    def reset(self) -> None:
        """
        Reset agent state.
        """
        self._history = []
        self.finished = False
        self.results = None
        for tool in self.tools.values():
            tool.reset()

    @property
    def history(self) -> str:
        """
        Get formatted agent interaction history.
        """
        return format_history(self._history)

    def finish(self, results: Any) -> str:
        """
        Mark the agent as finished with results.

        Args:
            results: The final results

        Returns:
            String representation of results
        """
        self.results = results
        self.finished = True
        return str(self.results)

    def is_finished(self) -> bool:
        """
        Check if the agent has finished processing.

        Returns:
            True if finished or max turns reached
        """
        return self.finished or len(self._history) >= self.max_turns

    def initialize_with_history(self, history_data: Dict[str, Any]) -> None:
        """
        Initialize agent with user's interaction history.
        This method should be called before any ranking operations.

        Args:
            history_data: Dictionary containing user_id and history_list
        """
        self.user_history = history_data.get('history_list', [])
        self.user_id = history_data.get('user_id', None)
        self._process_history()
        self.is_initialized = True

    @abstractmethod
    def _process_history(self) -> None:
        """
        Process the user's interaction history.
        Each subclass implements its own logic for understanding user preferences.
        """
        pass


    def _validate_initialization(self) -> None:
        """
        Check if the agent has been properly initialized with history.
        """
        if not self.is_initialized:
            raise RuntimeError(
                f"Agent must be initialized with history before ranking."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Make the agent callable.
        """
        self.validate_tools()
        self.reset()
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass of the agent.

        Returns:
            Agent output
        """
        raise NotImplementedError("BasePsychologicalAgent.forward() not implemented")

    @abstractmethod
    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Invoke the agent with the argument.

        Args:
            argument: The argument for the agent
            json_mode: Whether the argument is in JSON mode

        Returns:
            The observation of the invoking process
        """
        raise NotImplementedError("BasePsychologicalAgent.invoke() not implemented")
