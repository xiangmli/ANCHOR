from typing import Any, Dict, List, Optional
import json
from agents.base import BasePsychologicalAgent
import os


class Intuitor(BasePsychologicalAgent):
    """
    Intuitor agent representing the Id in Freud's psychological theory.

    The Id embodies immediate desires and instinctive impulses, focusing on
    quick gratification and intuitive appeal without rational deliberation.
    """

    def __init__(
            self,
            prompt_config: Optional[str] = None,
            web_demo: bool = False,
            system: Optional['System'] = None,
            dataset: Optional[str] = None,
            llm_config: Optional[Dict[str, Any]] = None,
            llm_config_path: Optional[str] = None,
            *args,
            **kwargs
    ):
        """
        Initialize the Intuitor agent.

        Args:
            prompt_config: Path to the prompt config file
            web_demo: Whether the agent is used in a web demo
            system: The system that the agent belongs to
            dataset: The dataset that the agent is used on
            llm_config: Configuration dictionary for the LLM
            llm_config_path: Path to LLM configuration file
        """
        super().__init__(
            prompt_config=prompt_config,
            web_demo=web_demo,
            system=system,
            dataset=dataset,
            *args,
            **kwargs
        )

        self.llm = self.get_LLM(config_path=llm_config_path, config=llm_config)

    @staticmethod
    def required_tools() -> Dict[str, type]:
        """
        Intuitor does not require specific tools for basic functionality.
        Can be extended if needed.

        Returns:
            Empty dictionary (no required tools)
        """
        return {}

    def _process_history(self, init_data: str, init_reviews_count: int) -> None:
        """
        Process the user's interaction history with intuitive, fast-thinking approach.

        Uses open-ended summarization - lets the LLM freely identify intuitive patterns
        in a structured list format.

        Args:
            init_data: Formatted user interaction history
            init_reviews_count: Number of reviews used for initialization
        """
        if not init_data:
            raise ValueError("init_data cannot be empty")

        self.observation(f"Initializing Intuitor for user {self.user_id}")
        self.observation(f"Using {init_reviews_count} initialization reviews")

        # Get prompt template
        if 'initialize_prompt' not in self.prompts:
            raise ValueError("initialize_prompt not found in prompts config")

        prompt_template = self.prompts['initialize_prompt']
        final_prompt = prompt_template.format(interaction_history=init_data)

        # Call LLM
        self.observation("Calling LLM to capture intuitive patterns...")
        response = self.llm(final_prompt)

        # Parse JSON response
        try:
            clean_response = response.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('```')[1]
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            intuitive_profile = json.loads(clean_response)

            # No longer enforce specific fields, only require valid JSON dictionary
            if not isinstance(intuitive_profile, dict):
                raise ValueError("Expected a dictionary from LLM response")

            # Store the complete profile
            self.intuitive_profile = intuitive_profile

            # Dynamically count the extracted content
            total_items = sum(
                len(v) if isinstance(v, list) else 1
                for v in intuitive_profile.values()
            )

            self.observation(
                f"Captured {len(intuitive_profile)} insight categories "
                f"with {total_items} total insights"
            )

            # Log which categories were extracted (for debugging)
            categories = ', '.join(intuitive_profile.keys())
            self.observation(f"Insight categories: {categories}")

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse LLM response as JSON: {e}")
            self.observation(f"Raw response: {response}")
            raise
        except Exception as e:
            self.observation(f"Unexpected error during initialization: {e}")
            raise

    def rank_candidates(
            self,
            candidate_set: Dict[str, Any],
            **kwargs
    ) -> Dict[str, Any]:
        """
        Rank candidate items based on intuitive, fast-thinking preferences.

        Intuitor focuses on immediate appeal, gut feelings, and surface-level patterns
        rather than deep analysis. Rankings prioritize items that trigger instant
        recognition and align with recent behavioral patterns.

        Args:
            candidate_set: Dictionary containing:
                - candidate_list: List of item IDs to rank
                - candidate_items_info: Formatted item information
                - ground_truth: The actual positive item (optional, for evaluation)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - ranked_list: List of item_ids in ranked order (most to least immediately appealing)
                - ranking_rationale: Explanation for the intuitive ranking decision
                - ground_truth: The ground truth item (if provided in input)
        """
        # Validate initialization
        if not self.is_initialized:
            raise RuntimeError(
                "Intuitor must be initialized with _process_history() before ranking"
            )

        if not self.intuitive_profile:
            raise ValueError(
                "Intuitive profile not found. Please run _process_history() first"
            )

        # Extract candidate information
        candidate_list = candidate_set.get('candidate_list', [])
        candidate_items_info = candidate_set.get('candidate_items_info', '')
        ground_truth = candidate_set.get('ground_truth', None)

        # Validate inputs
        if not candidate_list:
            raise ValueError("candidate_list is empty")

        if not candidate_items_info:
            raise ValueError("candidate_items_info is missing in candidate_set")

        # Prepare prompt data
        user_preferences_summary = self._format_preferences_for_prompt()
        candidate_count = len(candidate_list)

        # Generate candidate ID list to help LLM copy accurately
        candidate_ids_list = '\n'.join([
            f"{i}. {item_id}"
            for i, item_id in enumerate(candidate_list, 1)
        ])

        # Get ranking prompt template
        if 'ranking_prompt' not in self.prompts:
            raise ValueError("ranking_prompt not found in prompts config")

        prompt_template = self.prompts['ranking_prompt']

        # Format the final prompt with all required information
        try:
            final_prompt = prompt_template.format(
                user_preferences_summary=user_preferences_summary,
                candidate_count=candidate_count,
                candidate_items_info=candidate_items_info,
                candidate_ids_list=candidate_ids_list
            )
        except KeyError as e:
            raise ValueError(f"Missing placeholder in ranking prompt: {e}")

        # Call LLM to perform intuitive ranking
        self.observation(f"Calling LLM for intuitive ranking of {candidate_count} candidate items...")
        response = self.llm(final_prompt)

        # Parse JSON response
        try:
            # Clean response (remove markdown code blocks if present)
            clean_response = response.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('```')[1]
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            ranking_result = json.loads(clean_response)

            # Validate response structure
            if 'ranked_items' not in ranking_result:
                raise ValueError(
                    "LLM response missing 'ranked_items' field. "
                    f"Got keys: {list(ranking_result.keys())}"
                )

            ranked_list = ranking_result['ranked_items']
            ranking_rationale = ranking_result.get(
                'ranking_rationale',
                'No rationale provided'
            )

            # Validate that all items are included and no extras
            ranked_set = set(ranked_list)
            candidate_set_ids = set(candidate_list)

            if ranked_set != candidate_set_ids:
                missing = candidate_set_ids - ranked_set
                extra = ranked_set - candidate_set_ids

                warning_msg = "Warning: Ranked list does not match candidate list exactly"
                if missing:
                    warning_msg += f"\n  Missing items: {missing}"
                if extra:
                    warning_msg += f"\n  Extra items: {extra}"

                self.observation(warning_msg)

            # Validate list length
            if len(ranked_list) != len(candidate_list):
                self.observation(
                    f"Warning: Ranked list length ({len(ranked_list)}) != "
                    f"candidate list length ({len(candidate_list)})"
                )

            self.observation(f"Successfully completed intuitive ranking of {len(ranked_list)} items")

            # Prepare return dictionary
            result = {
                'ranked_list': ranked_list,
                'ranking_rationale': ranking_rationale,
                'ranking_type': 'intuitive'  # Identify as intuitive ranking
            }

            # Include ground truth if provided
            if ground_truth is not None:
                result['ground_truth'] = ground_truth

                # Calculate ground truth position for logging
                try:
                    gt_position = ranked_list.index(ground_truth) + 1
                    self.observation(
                        f"Ground truth item '{ground_truth}' ranked at position {gt_position}/{len(ranked_list)} "
                        f"(intuitive ranking)"
                    )
                except ValueError:
                    self.observation(
                        f"Warning: Ground truth item '{ground_truth}' not found in ranked list"
                    )

            return result

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse ranking response as JSON: {e}")
            self.observation(f"Raw response: {response[:500]}...")
            raise
        except Exception as e:
            self.observation(f"Unexpected error during intuitive ranking: {e}")
            raise

    def _format_preferences_for_prompt(self) -> str:
        """
        Format extracted intuitive profile for use in ranking prompts.

        Returns:
            Formatted string presenting the intuitive preferences
        """
        if not self.intuitive_profile:
            return "No intuitive profile available."

        output = "INTUITIVE USER PROFILE:\n"

        for category, insights in self.intuitive_profile.items():
            output += f"\n{category.upper().replace('_', ' ')}:\n"

            if isinstance(insights, list):
                for i, insight in enumerate(insights, 1):
                    output += f"  {i}. {insight}\n"
            else:
                output += f"  {insights}\n"

        return output

    def debate(
            self,
            own_ranking_result: Dict[str, Any],
            opponent_ranking_result: Dict[str, Any],
            **kwargs
    ) -> Dict[str, Any]:
        """
        Engage in debate with Reasoner agent to defend intuitive ranking.

        Intuitor argues for the superiority of gut-level, instinctive decision-making
        over rational analysis, using evidence from the user's impulsive behavior patterns.

        Args:
            own_ranking_result: Dictionary containing Intuitor's ranking:
                - ranked_list: List of item IDs in intuitive order
                - ranking_rationale: Intuitor's reasoning
                - ground_truth: The actual positive item (optional)
            opponent_ranking_result: Dictionary containing Reasoner's ranking:
                - ranked_list: List of item IDs in rational order
                - ranking_rationale: Reasoner's reasoning
                - ground_truth: The actual positive item (optional)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - defense_of_own_ranking: Argument for intuitive ranking
                - critique_of_opponent_ranking: Argument against rational ranking
                - debate_type: 'intuitor_vs_reasoner'
        """
        # Validate initialization
        if not self.is_initialized:
            raise RuntimeError(
                "Intuitor must be initialized before engaging in debate"
            )

        if not self.intuitive_profile:
            raise ValueError(
                "Intuitive profile not found. Cannot debate without user understanding"
            )

        # Extract ranking information
        intuitor_ranked_list = own_ranking_result.get('ranked_list', [])
        intuitor_rationale = own_ranking_result.get('ranking_rationale', '')

        reasoner_ranked_list = opponent_ranking_result.get('ranked_list', [])
        reasoner_rationale = opponent_ranking_result.get('ranking_rationale', '')

        # Validate inputs
        if not intuitor_ranked_list or not reasoner_ranked_list:
            raise ValueError("Both ranking results must contain non-empty ranked_list")

        if not intuitor_rationale or not reasoner_rationale:
            raise ValueError("Both ranking results must contain ranking_rationale")

        self.observation(f"Initiating debate with Reasoner agent...")
        self.observation(f"Defending intuitive ranking of {len(intuitor_ranked_list)} items")

        # Prepare top 5 items for comparison
        intuitor_top5 = intuitor_ranked_list[:5]
        reasoner_top5 = reasoner_ranked_list[:5]

        # Format intuitive profile for debate
        intuitor_profile_summary = self._format_preferences_for_prompt()

        # Format top 5 lists for prompt
        intuitor_top5_str = '\n'.join([
            f"{i}. {item_id}"
            for i, item_id in enumerate(intuitor_top5, 1)
        ])

        reasoner_top5_str = '\n'.join([
            f"{i}. {item_id}"
            for i, item_id in enumerate(reasoner_top5, 1)
        ])

        # Get debate prompt template
        if 'debate_prompt' not in self.prompts:
            raise ValueError("debate_prompt not found in prompts config")

        prompt_template = self.prompts['debate_prompt']

        # Format the final debate prompt
        try:
            final_prompt = prompt_template.format(
                intuitor_top5=intuitor_top5_str,
                intuitor_rationale=intuitor_rationale,
                reasoner_top5=reasoner_top5_str,
                reasoner_rationale=reasoner_rationale,
                intuitor_profile_summary=intuitor_profile_summary
            )
        except KeyError as e:
            raise ValueError(f"Missing placeholder in debate prompt: {e}")

        # Call LLM to generate debate arguments
        self.observation("Calling LLM to generate intuitive debate arguments...")
        response = self.llm(final_prompt)

        # Parse JSON response
        try:
            # Clean response (remove markdown code blocks if present)
            clean_response = response.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('```')[1]
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            debate_result = json.loads(clean_response)

            # Validate response structure (only check core fields)
            required_fields = [
                'defense_of_own_ranking',
                'critique_of_reasoner_ranking'
            ]

            missing_fields = [
                field for field in required_fields
                if field not in debate_result
            ]

            if missing_fields:
                raise ValueError(
                    f"LLM response missing required fields: {missing_fields}. "
                    f"Got keys: {list(debate_result.keys())}"
                )

            # Log debate summary
            self.observation("Generated intuitive debate arguments successfully")

            # Prepare simplified return dictionary
            result = {
                'defense_of_own_ranking': debate_result['defense_of_own_ranking'],
                'critique_of_opponent_ranking': debate_result['critique_of_reasoner_ranking'],
                'debate_type': 'intuitor_vs_reasoner'
            }

            # Log debate highlights
            self.observation("\n=== INTUITOR DEBATE HIGHLIGHTS ===")
            self.observation(f"Defense: {debate_result['defense_of_own_ranking'][:100]}...")
            self.observation(f"Critique: {debate_result['critique_of_reasoner_ranking'][:100]}...")
            self.observation("=================================\n")

            return result

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse debate response as JSON: {e}")
            self.observation(f"Raw response: {response[:500]}...")
            raise
        except Exception as e:
            self.observation(f"Unexpected error during debate generation: {e}")
            raise

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass of the agent.

        TODO: Implement the main workflow for Intuitor:
        - Process immediate sensory information
        - Generate quick, instinctive responses
        - Prioritize items with immediate appeal
        """
        pass

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Invoke the agent with the argument.

        TODO: Implement how Intuitor processes input arguments
        with a focus on immediate, instinctive responses.
        """
        pass


if __name__ == "__main__":
    """
    Test script for Intuitor ranking.
    For each candidate set:
    1. Initialize with user history
    2. Rank candidates based on intuitive preferences
    3. Display ranking results and metrics
    """
    import sys
    from pathlib import Path
    import json
    from utils import init_openai_api, read_json

    # ==================== 配置参数 ====================
    # 可配置的 Hit@k 值列表
    HIT_K_LIST = [1, 3, 5, 10]

    # Set up paths
    current_dir = Path(__file__).parent  # agents/
    project_root = current_dir.parent  # mycode/

    # Initialize OpenAI API
    api_config_path = project_root / 'config' / 'api-config.json'
    init_openai_api(read_json(str(api_config_path)))

    # Configuration paths
    config_dir = project_root / "config"
    prompts_dir = config_dir / "prompts"
    agents_config_dir = config_dir / "agents"
    data_dir = project_root / "data" / "amazon"

    prompt_config_path = prompts_dir / "intuitor.json"
    llm_config_path = agents_config_dir / "intuitor.json"
    sampled_data_path = data_dir / "sampled_users_data.json"

    print("=" * 80)
    print("INTUITOR RANKING TEST")
    print("=" * 80)
    print(f"\nPrompt Config: {prompt_config_path}")
    print(f"LLM Config: {llm_config_path}")
    print(f"Sampled Data: {sampled_data_path}")
    print(f"Dataset: amazon")
    print(f"Hit@k Configuration: {HIT_K_LIST}")
    print("\n" + "-" * 80 + "\n")

    try:
        # Step 1: Load sampled user data
        print("Step 1: Loading sampled user data...")
        with open(sampled_data_path, 'r', encoding='utf-8') as f:
            all_user_data = json.load(f)
        print(f"✓ Loaded data for {len(all_user_data)} users\n")

        # Step 2: Load LLM configuration
        print("Step 2: Loading LLM configuration...")
        with open(llm_config_path, 'r', encoding='utf-8') as f:
            llm_config = json.load(f)
        print(f"✓ LLM config loaded: {llm_config.get('model_name', 'N/A')}\n")

        # Step 3: Initialize Intuitor
        print("Step 3: Creating Intuitor instance...")
        intuitor = Intuitor(
            prompt_config=str(prompt_config_path),
            dataset="amazon",
            llm_config=llm_config,
            web_demo=False
        )
        print("✓ Intuitor created successfully\n")

        # Step 4: Get first user data
        test_user_id = list(all_user_data.keys())[0]
        test_user_data = all_user_data[test_user_id]

        print(f"Step 4: Testing with user {test_user_id}")
        print(f"  - Init reviews: {test_user_data['init_reviews_count']}")
        print(f"  - Refinement sets: {test_user_data['refinement_count']}")
        print(f"  - Total reviews: {test_user_data['total_reviews']}")
        print("-" * 80 + "\n")

        # Step 5: Process user history to extract intuitive profile
        print("Step 5: Extracting intuitive user profile...")
        intuitor.user_id = test_user_data['user_id']
        intuitor._process_history(
            init_data=test_user_data['init_data'],
            init_reviews_count=test_user_data['init_reviews_count']
        )
        intuitor.is_initialized = True

        # Display intuitive profile
        print(f"✓ Intuitive profile extracted successfully")
        print(f"\nINTUITIVE PROFILE SUMMARY:")
        print(f"{'─' * 80}")
        for category, insights in intuitor.intuitive_profile.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            if isinstance(insights, list):
                for i, insight in enumerate(insights, 1):
                    print(f"  {i}. {insight}")
            else:
                print(f"  {insights}")
        print(f"\n{'─' * 80}\n")

        # Step 6: Ranking for all refinement sets
        print("=" * 80)
        print(f"INTUITIVE RANKING LOOP ({test_user_data['refinement_count']} SETS)")
        print("=" * 80 + "\n")

        ranking_results = []

        for idx, refinement_data in enumerate(test_user_data['refinement_candidate_sets'], 1):
            print(f"{'=' * 80}")
            print(f"CANDIDATE SET {idx}/{test_user_data['refinement_count']}")
            print(f"{'=' * 80}\n")

            # Extract candidate set data
            candidate_set = {
                'candidate_list': refinement_data['candidate_list'],
                'ground_truth': refinement_data['ground_truth'],
                'candidate_items_info': refinement_data['candidate_items_info']
            }

            # Rank candidates
            print(f"[Step 6.{idx}] Ranking candidates intuitively...")
            ranking_result = intuitor.rank_candidates(candidate_set)
            ranking_results.append(ranking_result)

            # Display ranking results
            ranked_list = ranking_result['ranked_list']
            ground_truth = ranking_result['ground_truth']
            ranking_rationale = ranking_result['ranking_rationale']

            try:
                gt_position = ranked_list.index(ground_truth) + 1
            except ValueError:
                gt_position = None

            print(f"  ✓ Ranked {len(ranked_list)} items")
            print(f"  Ground Truth: {ground_truth}")
            if gt_position:
                print(f"  Ground Truth Position: {gt_position}/{len(ranked_list)}")
            else:
                print(f"  Ground Truth Position: NOT FOUND")

            print(f"\n  Top 5 Ranked Items (Intuitive Ranking):")
            for rank, item_id in enumerate(ranked_list[:5], 1):
                marker = " ← GROUND TRUTH" if item_id == ground_truth else ""
                print(f"    {rank}. {item_id}{marker}")

            print(f"\n  Intuitive Ranking Rationale:")
            # Wrap long rationale text
            rationale_lines = ranking_rationale.split('\n')
            for line in rationale_lines:
                if len(line) > 76:
                    # Split long lines
                    words = line.split()
                    current_line = "    "
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 76:
                            current_line += word + " "
                        else:
                            print(current_line.rstrip())
                            current_line = "    " + word + " "
                    print(current_line.rstrip())
                else:
                    print(f"    {line}")

            print()  # Empty line for separation

        # Step 7: Calculate and display summary statistics
        print("=" * 80)
        print("INTUITIVE RANKING SUMMARY STATISTICS")
        print("=" * 80)

        # Calculate ranking metrics
        total_sets = len(ranking_results)
        ground_truth_positions = []

        for result in ranking_results:
            try:
                gt_pos = result['ranked_list'].index(result['ground_truth']) + 1
                ground_truth_positions.append(gt_pos)
            except ValueError:
                ground_truth_positions.append(None)

        valid_positions = [pos for pos in ground_truth_positions if pos is not None]

        print(f"\n{'─' * 80}")
        print("RANKING PERFORMANCE")
        print(f"{'─' * 80}")

        if valid_positions:
            avg_position = sum(valid_positions) / len(valid_positions)

            # Calculate Hit@k for all configured values
            hit_metrics = {}
            for k in HIT_K_LIST:
                hit_count = sum(1 for pos in valid_positions if pos <= k)
                hit_metrics[k] = {
                    'count': hit_count,
                    'percentage': hit_count / total_sets * 100
                }

            print(f"Total Candidate Sets: {total_sets}")
            print(f"Valid Rankings: {len(valid_positions)}/{total_sets}")
            print(f"Average Ground Truth Position: {avg_position:.2f}")

            # Display all configured Hit@k metrics
            print(f"\nHit Rates (Intuitive Ranking):")
            for k in HIT_K_LIST:
                metric = hit_metrics[k]
                print(f"  Hit@{k:2d}: {metric['count']}/{total_sets} "
                      f"({metric['percentage']:5.1f}%)")

            print(f"\nPosition Distribution:")
            position_counts = {}
            for pos in valid_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1

            for pos in sorted(position_counts.keys())[:10]:  # Show top 10 positions
                count = position_counts[pos]
                percentage = count / total_sets * 100
                bar = '█' * int(percentage / 5)  # Visual bar
                print(f"  Position {pos:2d}: {count:2d} times ({percentage:5.1f}%) {bar}")

            if len(position_counts) > 10:
                print(f"  ... and {len(position_counts) - 10} more positions")

        else:
            print("✗ No valid rankings found")

        # Final state summary
        print(f"\n{'─' * 80}")
        print("FINAL INTUITOR STATE")
        print(f"{'─' * 80}")

        print(f"User ID: {intuitor.user_id}")
        print(f"Intuitive Profile Categories: {len(intuitor.intuitive_profile)}")
        print(
            f"Total Insights Extracted: {sum(len(v) if isinstance(v, list) else 1 for v in intuitor.intuitive_profile.values())}")
        print(f"Candidate Sets Ranked: {total_sets}")
        print(f"Ranking Type: Intuitive (Fast-thinking, Id-driven)")

        print("\n" + "=" * 80)
        print("✓ Intuitive Ranking Test Completed Successfully!")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}")
        print("\nPlease ensure the following files exist:")
        print(f"  - Prompt config: {prompt_config_path}")
        print(f"  - LLM config: {llm_config_path}")
        print(f"  - Sampled data: {sampled_data_path}")
        sys.exit(1)

    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"✗ Error: Failed to parse JSON file - {e}")
        sys.exit(1)

    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


