from typing import Dict, Any, Optional
from pathlib import Path
import json
from agents.adjudicator import Adjudicator
from agents.intuitor import Intuitor
from agents.reasoner import Reasoner


class User:
    """
    User class representing a single user with three psychological agents.

    Each user has three agents that model different aspects of their decision-making:
    - Adjudicator (Ego): Balances intuition and rationality
    - Intuitor (Id): Models impulsive, intuitive preferences
    - Reasoner (Superego): Models rational, deliberate thinking
    """

    def __init__(
            self,
            user_id: str,
            dataset: str,
            init_data: str,
            init_reviews_count: int,
            config_dir: Optional[Path] = None,
            adjudicator_llm_config: Optional[Dict[str, Any]] = None,
            intuitor_llm_config: Optional[Dict[str, Any]] = None,
            reasoner_llm_config: Optional[Dict[str, Any]] = None,
            web_demo: bool = False
    ):
        """
        Initialize a User with three psychological agents.

        Args:
            user_id: Unique identifier for the user
            dataset: Dataset name (e.g., 'amazon', 'yelp')
            init_data: Formatted user interaction history string
            init_reviews_count: Number of reviews used for initialization
            config_dir: Path to configuration directory (default: project_root/config)
            adjudicator_llm_config: LLM configuration for Adjudicator (optional)
            intuitor_llm_config: LLM configuration for Intuitor (optional)
            reasoner_llm_config: LLM configuration for Reasoner (optional)
            web_demo: Whether running in web demo mode
        """
        # Store user information
        self.user_id = user_id
        self.dataset = dataset
        self.init_reviews_count = init_reviews_count

        # Set up configuration paths
        if config_dir is None:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            config_dir = project_root / "config"

        self.config_dir = config_dir
        prompts_dir = config_dir / "prompts"
        agents_config_dir = config_dir / "agents"

        # Initialize Adjudicator
        adjudicator_prompt_config = str(prompts_dir / "adjudicator.json")

        if adjudicator_llm_config is None:
            with open(agents_config_dir / "adjudicator.json", 'r') as f:
                adjudicator_llm_config = json.load(f)

        self.adjudicator = Adjudicator(
            prompt_config=adjudicator_prompt_config,
            dataset=dataset,
            llm_config=adjudicator_llm_config,
            web_demo=web_demo
        )
        self.adjudicator.user_id = user_id
        self.adjudicator._process_history(init_data, init_reviews_count)
        self.adjudicator.is_initialized = True

        # Initialize Intuitor
        intuitor_prompt_config = str(prompts_dir / "intuitor.json")

        if intuitor_llm_config is None:
            with open(agents_config_dir / "intuitor.json", 'r') as f:
                intuitor_llm_config = json.load(f)

        self.intuitor = Intuitor(
            prompt_config=intuitor_prompt_config,
            dataset=dataset,
            llm_config=intuitor_llm_config,
            web_demo=web_demo
        )
        self.intuitor.user_id = user_id
        self.intuitor._process_history(init_data, init_reviews_count)
        self.intuitor.is_initialized = True

        # Initialize Reasoner
        reasoner_prompt_config = str(prompts_dir / "reasoner.json")

        if reasoner_llm_config is None:
            with open(agents_config_dir / "reasoner.json", 'r') as f:
                reasoner_llm_config = json.load(f)

        self.reasoner = Reasoner(
            prompt_config=reasoner_prompt_config,
            dataset=dataset,
            llm_config=reasoner_llm_config,
            web_demo=web_demo
        )
        self.reasoner.user_id = user_id
        self.reasoner._process_history(init_data, init_reviews_count)
        self.reasoner.is_initialized = True
