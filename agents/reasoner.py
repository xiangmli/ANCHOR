from typing import Any, Dict, List, Optional
import json
from agents.base import BasePsychologicalAgent
import os

class Reasoner(BasePsychologicalAgent):
    """
    Reasoner agent representing the Superego in Freud's psychological theory.

    The Superego embodies rational deliberation and long-term considerations,
    focusing on thoughtful analysis and delayed gratification principles.
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
        Initialize the Reasoner agent.

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
        Reasoner does not require specific tools for basic functionality.
        Can be extended if needed.

        Returns:
            Empty dictionary (no required tools)
        """
        return {}

    def _process_history(self, init_data: str, init_reviews_count: int) -> None:
        """
        Process the user's interaction history with rational, deliberate thinking.

        Uses open-ended analysis - lets the LLM freely structure the rational profile
        based on deep reasoning.

        Args:
            init_data: Formatted user interaction history
            init_reviews_count: Number of reviews used for initialization
        """
        if not init_data:
            raise ValueError("init_data cannot be empty")

        self.observation(f"Initializing Reasoner for user {self.user_id}")
        self.observation(f"Using {init_reviews_count} initialization reviews")

        # Get prompt template
        if 'initialize_prompt' not in self.prompts:
            raise ValueError("initialize_prompt not found in prompts config")

        prompt_template = self.prompts['initialize_prompt']
        final_prompt = prompt_template.format(interaction_history=init_data)

        # Call LLM
        self.observation("Calling LLM to perform deep analysis...")
        response = self.llm(final_prompt)

        # Parse JSON response
        try:
            clean_response = response.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('```')[1]
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            rational_profile = json.loads(clean_response)

            # No longer enforce specific fields, only require valid JSON dictionary
            if not isinstance(rational_profile, dict):
                raise ValueError("Expected a dictionary from LLM response")

            # Store the complete profile
            self.rational_profile = rational_profile

            # Dynamically count the extracted content
            total_items = sum(
                len(v) if isinstance(v, list) else 1
                for v in rational_profile.values()
            )

            self.observation(
                f"Identified {len(rational_profile)} analysis dimensions "
                f"with {total_items} total insights"
            )

            # Log which dimensions were analyzed (for debugging)
            dimensions = ', '.join(rational_profile.keys())
            self.observation(f"Analysis dimensions: {dimensions}")

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
        Rank candidate items based on rational, systematic analysis.

        Reasoner focuses on feature-level evaluation, value assessment, and logical
        consistency. Rankings prioritize items that optimize long-term satisfaction
        and practical utility through evidence-based reasoning.

        Args:
            candidate_set: Dictionary containing:
                - candidate_list: List of item IDs to rank
                - candidate_items_info: Formatted item information
                - ground_truth: The actual positive item (optional, for evaluation)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - ranked_list: List of item_ids in ranked order (most to least rationally optimal)
                - ranking_rationale: Explanation for the rational ranking decision
                - ground_truth: The ground truth item (if provided in input)
        """
        # Validate initialization
        if not self.is_initialized:
            raise RuntimeError(
                "Reasoner must be initialized with _process_history() before ranking"
            )

        if not self.rational_profile:
            raise ValueError(
                "Rational profile not found. Please run _process_history() first"
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

        # Call LLM to perform rational ranking
        self.observation(f"Calling LLM for rational ranking of {candidate_count} candidate items...")
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

            self.observation(f"Successfully completed rational ranking of {len(ranked_list)} items")

            # Prepare return dictionary
            result = {
                'ranked_list': ranked_list,
                'ranking_rationale': ranking_rationale,
                'ranking_type': 'rational'  # Identify as rational ranking
            }

            # Include ground truth if provided
            if ground_truth is not None:
                result['ground_truth'] = ground_truth

                # Calculate ground truth position for logging
                try:
                    gt_position = ranked_list.index(ground_truth) + 1
                    self.observation(
                        f"Ground truth item '{ground_truth}' ranked at position {gt_position}/{len(ranked_list)} "
                        f"(rational ranking)"
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
            self.observation(f"Unexpected error during rational ranking: {e}")
            raise

    def _format_preferences_for_prompt(self) -> str:
        """
        Format extracted rational profile for use in ranking prompts.

        Returns:
            Formatted string presenting the rational preferences
        """
        if not self.rational_profile:
            return "No rational profile available."

        output = "RATIONAL USER PROFILE:\n"

        for dimension, insights in self.rational_profile.items():
            output += f"\n{dimension.upper().replace('_', ' ')}:\n"

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
        Engage in debate with Intuitor agent to defend rational ranking.

        Reasoner argues for the superiority of systematic, evidence-based analysis
        over impulsive decision-making, using evidence from the user's thoughtful
        behavior patterns and past experiences.

        Args:
            own_ranking_result: Dictionary containing Reasoner's ranking:
                - ranked_list: List of item IDs in rational order
                - ranking_rationale: Reasoner's reasoning
                - ground_truth: The actual positive item (optional)
            opponent_ranking_result: Dictionary containing Intuitor's ranking:
                - ranked_list: List of item IDs in intuitive order
                - ranking_rationale: Intuitor's reasoning
                - ground_truth: The actual positive item (optional)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - defense_of_own_ranking: Argument for rational ranking
                - critique_of_opponent_ranking: Argument against intuitive ranking
                - debate_type: 'reasoner_vs_intuitor'
        """
        # Validate initialization
        if not self.is_initialized:
            raise RuntimeError(
                "Reasoner must be initialized before engaging in debate"
            )

        if not self.rational_profile:
            raise ValueError(
                "Rational profile not found. Cannot debate without user understanding"
            )

        # Extract ranking information
        reasoner_ranked_list = own_ranking_result.get('ranked_list', [])
        reasoner_rationale = own_ranking_result.get('ranking_rationale', '')

        intuitor_ranked_list = opponent_ranking_result.get('ranked_list', [])
        intuitor_rationale = opponent_ranking_result.get('ranking_rationale', '')

        # Validate inputs
        if not reasoner_ranked_list or not intuitor_ranked_list:
            raise ValueError("Both ranking results must contain non-empty ranked_list")

        if not reasoner_rationale or not intuitor_rationale:
            raise ValueError("Both ranking results must contain ranking_rationale")

        self.observation(f"Initiating debate with Intuitor agent...")
        self.observation(f"Defending rational ranking of {len(reasoner_ranked_list)} items")

        # Prepare top 5 items for comparison
        reasoner_top5 = reasoner_ranked_list[:5]
        intuitor_top5 = intuitor_ranked_list[:5]

        # Format rational profile for debate
        reasoner_profile_summary = self._format_preferences_for_prompt()

        # Format top 5 lists for prompt
        reasoner_top5_str = '\n'.join([
            f"{i}. {item_id}"
            for i, item_id in enumerate(reasoner_top5, 1)
        ])

        intuitor_top5_str = '\n'.join([
            f"{i}. {item_id}"
            for i, item_id in enumerate(intuitor_top5, 1)
        ])

        # Get debate prompt template
        if 'debate_prompt' not in self.prompts:
            raise ValueError("debate_prompt not found in prompts config")

        prompt_template = self.prompts['debate_prompt']

        # Format the final debate prompt
        try:
            final_prompt = prompt_template.format(
                reasoner_top5=reasoner_top5_str,
                reasoner_rationale=reasoner_rationale,
                intuitor_top5=intuitor_top5_str,
                intuitor_rationale=intuitor_rationale,
                reasoner_profile_summary=reasoner_profile_summary
            )
        except KeyError as e:
            raise ValueError(f"Missing placeholder in debate prompt: {e}")

        # Call LLM to generate debate arguments
        self.observation("Calling LLM to generate rational debate arguments...")
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
                'critique_of_intuitor_ranking'
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
            self.observation("Generated rational debate arguments successfully")

            # Prepare simplified return dictionary
            result = {
                'defense_of_own_ranking': debate_result['defense_of_own_ranking'],
                'critique_of_opponent_ranking': debate_result['critique_of_intuitor_ranking'],
                'debate_type': 'reasoner_vs_intuitor'
            }

            # Log debate highlights
            self.observation("\n=== REASONER DEBATE HIGHLIGHTS ===")
            self.observation(f"Defense: {debate_result['defense_of_own_ranking'][:100]}...")
            self.observation(f"Critique: {debate_result['critique_of_intuitor_ranking'][:100]}...")
            self.observation("==================================\n")

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

        TODO: Implement the main workflow for Reasoner:
        - Conduct thorough analysis
        - Consider long-term implications
        - Apply rational decision-making principles
        """
        pass

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Invoke the agent with the argument.

        TODO: Implement how Reasoner processes input arguments
        with a focus on rational, deliberate analysis.
        """
        pass


if __name__ == "__main__":
    """
    Test script for Reasoner ranking.
    For each candidate set:
    1. Initialize with user history (rational analysis)
    2. Rank candidates based on rational preferences
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

    prompt_config_path = prompts_dir / "reasoner.json"
    llm_config_path = agents_config_dir / "reasoner.json"
    sampled_data_path = data_dir / "sampled_users_data.json"

    print("=" * 80)
    print("REASONER RANKING TEST")
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

        # Step 3: Initialize Reasoner
        print("Step 3: Creating Reasoner instance...")
        reasoner = Reasoner(
            prompt_config=str(prompt_config_path),
            dataset="amazon",
            llm_config=llm_config,
            web_demo=False
        )
        print("✓ Reasoner created successfully\n")

        # Step 4: Get first user data
        test_user_id = list(all_user_data.keys())[0]
        test_user_data = all_user_data[test_user_id]

        print(f"Step 4: Testing with user {test_user_id}")
        print(f"  - Init reviews: {test_user_data['init_reviews_count']}")
        print(f"  - Refinement sets: {test_user_data['refinement_count']}")
        print(f"  - Total reviews: {test_user_data['total_reviews']}")
        print("-" * 80 + "\n")

        # Step 5: Process user history to extract rational profile
        print("Step 5: Extracting rational user profile...")
        reasoner.user_id = test_user_data['user_id']
        reasoner._process_history(
            init_data=test_user_data['init_data'],
            init_reviews_count=test_user_data['init_reviews_count']
        )
        reasoner.is_initialized = True

        # Display rational profile
        print(f"✓ Rational profile extracted successfully")
        print(f"\nRATIONAL PROFILE SUMMARY:")
        print(f"{'─' * 80}")
        for dimension, insights in reasoner.rational_profile.items():
            print(f"\n{dimension.upper().replace('_', ' ')}:")
            if isinstance(insights, list):
                for i, insight in enumerate(insights, 1):
                    print(f"  {i}. {insight}")
            else:
                print(f"  {insights}")
        print(f"\n{'─' * 80}\n")

        # Step 6: Ranking for all refinement sets
        print("=" * 80)
        print(f"RATIONAL RANKING LOOP ({test_user_data['refinement_count']} SETS)")
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
            print(f"[Step 6.{idx}] Ranking candidates rationally...")
            ranking_result = reasoner.rank_candidates(candidate_set)
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

            print(f"\n  Top 5 Ranked Items (Rational Ranking):")
            for rank, item_id in enumerate(ranked_list[:5], 1):
                marker = " ← GROUND TRUTH" if item_id == ground_truth else ""
                print(f"    {rank}. {item_id}{marker}")

            print(f"\n  Rational Ranking Rationale:")
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
        print("RATIONAL RANKING SUMMARY STATISTICS")
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
            print(f"\nHit Rates (Rational Ranking):")
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

            # Best and worst cases
            print(f"\nPerformance Extremes:")
            best_position = min(valid_positions)
            worst_position = max(valid_positions)
            print(f"  Best Position: {best_position}")
            print(f"  Worst Position: {worst_position}")
            print(f"  Position Range: {worst_position - best_position + 1}")

        else:
            print("✗ No valid rankings found")

        # Final state summary
        print(f"\n{'─' * 80}")
        print("FINAL REASONER STATE")
        print(f"{'─' * 80}")

        print(f"User ID: {reasoner.user_id}")
        print(f"Rational Profile Dimensions: {len(reasoner.rational_profile)}")
        print(
            f"Total Insights Extracted: {sum(len(v) if isinstance(v, list) else 1 for v in reasoner.rational_profile.values())}")
        print(f"Candidate Sets Ranked: {total_sets}")
        print(f"Ranking Type: Rational (Slow-thinking, Superego-driven)")

        # Comparative statistics
        if valid_positions:
            print(f"\nRanking Quality Indicators:")
            top3_rate = sum(1 for pos in valid_positions if pos <= 3) / total_sets * 100
            top5_rate = sum(1 for pos in valid_positions if pos <= 5) / total_sets * 100

            print(f"  Precision: {top3_rate:.1f}% in top-3")
            print(f"  Recall: {top5_rate:.1f}% in top-5")

            # Consistency metric (standard deviation)
            if len(valid_positions) > 1:
                mean_pos = sum(valid_positions) / len(valid_positions)
                variance = sum((pos - mean_pos) ** 2 for pos in valid_positions) / len(valid_positions)
                std_dev = variance ** 0.5
                print(f"  Consistency (std dev): {std_dev:.2f}")

        print("\n" + "=" * 80)
        print("✓ Rational Ranking Test Completed Successfully!")
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

