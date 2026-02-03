from typing import Any, Dict, List, Optional
import json
from agents import BasePsychologicalAgent
import os
import sys
from pathlib import Path
import json
from util import init_openai_api, read_json
import numpy as np

class Adjudicator(BasePsychologicalAgent):

    def __init__(
            self,
            prompt_config: Optional[str] = None,
            web_demo: bool = False,
            system: Optional['System'] = None,
            dataset: Optional[str] = None,
            llm_config: Optional[Dict[str, Any]] = None,
            llm_config_path: Optional[str] = None,
            model_path: Optional[str] = None,  # ✅ 新增：推荐模型权重路径
            modeltype: Optional[str] = None,  # ✅ 新增：推荐模型类型 (LightGCN, NGCF, etc.)
            use_recommender: bool = False,  # ✅ 新增：是否使用推荐模型采样
            *args,
            **kwargs
    ):
        """
        Initialize the Adjudicator agent.

        Args:
            prompts: Dictionary of prompts for the agent
            prompt_config: Path to the prompt config file
            web_demo: Whether the agent is used in a web demo
            system: The system that the agent belongs to
            dataset: The dataset that the agent is used on
            llm_config: Configuration dictionary for the LLM
            llm_config_path: Path to LLM configuration file
            model_path: Path to recommender model weights (for candidate sampling)
            modeltype: Type of recommender model (LightGCN, NGCF, NCF, MF, Pop, Random)
            use_recommender: Whether to use recommender model for candidate sampling
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

        # ✅ 推荐模型相关属性
        self.model_path = model_path
        self.modeltype = modeltype
        self.use_recommender = use_recommender
        self.recommender_model = None
        self.full_rankings = None  # shape: (n_users, n_items)，存储每个用户的物品排名

        # ✅ 如果提供了推荐模型路径，则加载模型
        if use_recommender and model_path and modeltype and dataset:
            self._load_recommender_model()

        # ✅ 初始化缓存
        self._user_interacted_cache = {}

        self.adversarial_memory: List[Dict[str, Any]] = []
        self.max_adversarial_memory_entries: int = 10

        print(f"\nAdjudicator Agent initialized successfully!")
        if self.use_recommender and self.recommender_model:
            print(f"  - Using recommender model: {self.modeltype}")
            print(f"  - Candidate sampling: Model-based")
        else:
            print(f"  - Candidate sampling: Random")

    @staticmethod
    def required_tools() -> Dict[str, type]:
        """
        Adjudicator does not require specific tools for basic functionality.
        Can be extended if needed.

        Returns:
            Empty dictionary (no required tools)
        """
        return {}

    @staticmethod
    def prepare_init_data(
            train_file: str,
            item_list_file: str,
            max_history_len: int = 50,
    ) -> Dict[int, Dict[str, Any]]:
        """
        从 train.txt 和 item_list.txt 解析得到适合丢给 LLM 的用户交互历史文本。

        参数：
            train_file      : train.txt 文件路径（格式：user_id item_id1 item_id2 ...）
            item_list_file  : item_list.txt 文件路径（格式：org_id remap_id entity_name）
            max_history_len : 每个用户最多使用多少条交互记录（防止 prompt 太长）

        返回：
            一个字典：
            {
                user_id_0: {
                    "init_data": <给 LLM 的 interaction_history 文本>,
                    "init_reviews_count": <使用了多少条交互>,
                    "item_ids": [item_id1, item_id2, ...]   # 可选辅助信息
                },
                user_id_1: { ... },
                ...
            }
        """

        # 1. 读取 item_list.txt，建立 item_id -> DBpedia URI 映射
        item_mapping: Dict[int, str] = {}
        with open(item_list_file, "r", encoding="utf-8") as f:
            header = f.readline()  # 跳过表头：org_id remap_id entity_name
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                # parts[0] = org_id, parts[1] = remap_id, parts[2:] = entity_name
                try:
                    remap_id = int(parts[1])
                except ValueError:
                    # 防止有奇怪行
                    continue

                entity_uri = " ".join(parts[2:])  # 一般只有一个 token，但这样写更稳
                item_mapping[remap_id] = entity_uri

        # 小工具：把 DBpedia URI 转成可读的书名
        def make_item_title(uri: str) -> str:
            if not uri:
                return "Unknown item"
            name = uri.rsplit("/", 1)[-1]  # 取最后一段
            name = name.replace("_", " ")  # 下划线 -> 空格
            return name

        # 2. 遍历 train.txt，为每个 user 构造 init_data 文本
        users_data: Dict[int, Dict[str, Any]] = {}

        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    # 至少要有 user_id 和一个 item_id
                    continue

                # user_id 在第一个位置
                try:
                    user_id = int(parts[0])
                except ValueError:
                    # 如果后面哪天 train.txt 有表头之类的，直接跳过
                    continue

                # 后面的都是 item_id
                item_ids: List[int] = []
                for token in parts[1:]:
                    try:
                        item_ids.append(int(token))
                    except ValueError:
                        # 防御式处理：某个 token 不是数字就忽略
                        continue

                if not item_ids:
                    continue

                # 控制历史长度
                used_item_ids = item_ids[:max_history_len]
                review_count = len(used_item_ids)

                # 3. 拼接成 LLM-friendly 的 interaction_history 文本
                lines: List[str] = []
                lines.append(f"User ID: {user_id}")
                lines.append(f"Number of interactions used: {review_count}")
                lines.append("")
                lines.append("Interacted items:")

                for idx, item_id in enumerate(used_item_ids, 1):
                    uri = item_mapping.get(item_id)
                    if uri:
                        title = make_item_title(uri)
                        lines.append(
                            f"{idx}. Item ID {item_id} | Title: {title} | DBpedia URI: {uri}"
                        )
                    else:
                        lines.append(
                            f"{idx}. Item ID {item_id} | Title: UNKNOWN | DBpedia URI: N/A"
                        )

                init_data_text = "\n".join(lines)

                users_data[user_id] = {
                    "init_data": init_data_text,
                    "init_reviews_count": review_count,
                    "item_ids": used_item_ids,
                }

        return users_data

    def _process_history(self, init_data: str, init_reviews_count: int) -> None:
        """
        Process the user's interaction history with retry logic.

        Args:
            init_data: Formatted user interaction history
            init_reviews_count: Number of reviews used for initialization
        """
        if not init_data:
            raise ValueError("init_data cannot be empty")

        self.observation(f"Initializing Adjudicator for user {self.user_id}")
        self.observation(f"Using {init_reviews_count} initialization reviews")

        # Get prompt template
        if 'initialize_prompt' not in self.prompts:
            raise ValueError("initialize_prompt not found in prompts config")

        prompt_template = self.prompts['initialize_prompt']

        # 处理嵌套的 dict 格式
        if isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("content", "")

        final_prompt = prompt_template.format(interaction_history=init_data)

        # Call LLM with retry logic
        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    self.observation(f"Retrying... (Attempt {attempt + 1}/{max_retries + 1})")

                self.observation("Calling LLM to extract user preferences...")
                response = self.llm(final_prompt)

                # Parse JSON response
                clean_response = response.strip()

                if not clean_response:
                    raise ValueError("LLM returned an empty response")

                # 删除 markdown code blocks
                if clean_response.startswith('```'):
                    clean_response = clean_response.split('```')[1]
                    if clean_response.startswith('json'):
                        clean_response = clean_response[4:]
                clean_response = clean_response.strip()

                # 检查是否是 JSON
                if not clean_response.startswith('{'):
                    raise ValueError("LLM response is not JSON format (should start with '{')")

                # 解析 JSON
                preferences_data = json.loads(clean_response)

                # ✅ 验证：确保有 preferences 字段且是数组
                if not isinstance(preferences_data, dict):
                    raise ValueError("JSON root must be an object")

                if 'preferences' not in preferences_data:
                    raise ValueError("JSON must contain 'preferences' field")

                # ✅ 现在 preferences 应该是数组（回到列表格式）
                if not isinstance(preferences_data['preferences'], list):
                    raise ValueError("preferences field must be an array (list), not a string")

                if len(preferences_data['preferences']) < 3:
                    raise ValueError("preferences array must have at least 3 items")

                if len(preferences_data['preferences']) > 10:
                    raise ValueError("preferences array should not have more than 10 items")

                # 验证每个 preference 是字符串
                for i, pref in enumerate(preferences_data['preferences']):
                    if not isinstance(pref, str):
                        raise ValueError(f"preference[{i}] must be a string, got {type(pref)}")
                    if len(pref.strip()) < 5:
                        raise ValueError(f"preference[{i}] is too short")

                # 成功！
                self.user_preferences = preferences_data

                self.observation(
                    f"✓ Successfully extracted {len(preferences_data['preferences'])} user preferences"
                )
                for i, pref in enumerate(preferences_data['preferences'], 1):
                    self.observation(f"  {i}. {pref}")
                return  # 成功退出

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                self.observation(f"Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries:
                    self.observation(f"Will retry in next attempt...")
                    continue
                else:
                    # 所有重试都失败了
                    self.observation(f"All {max_retries + 1} attempts failed")
                    self.observation(f"Raw response (first 500 chars): {response[:500]}")
                    raise

        # 这行不应该被执行到，但以防万一
        raise RuntimeError(f"Failed to process history after {max_retries + 1} attempts: {last_error}")

    # 其他辅助方法保持不变
    def _format_preferences_for_prompt(self) -> str:
        """
        Format extracted user preferences for use in ranking prompts.

        ✅ preferences 回到数组格式
        """
        if not self.user_preferences:
            return "No user preferences available."

        # ✅ 现在 preferences 是数组（列表）
        preferences_list = self.user_preferences.get('preferences', [])

        if not preferences_list:
            return "No user preferences available."

        output = "USER PREFERENCES:\n"
        for i, pref in enumerate(preferences_list, 1):
            output += f"{i}. {pref}\n"

        return output

    def _format_preferences_output(self, preferences_data: Dict[str, str]) -> str:
        """
        Format preferences as readable text.

        ✅ 新版本：只显示 preferences（字符串格式）
        """
        output = "=" * 60 + "\n"
        output += "USER PREFERENCE ANALYSIS\n"
        output += "=" * 60 + "\n\n"

        output += "PREFERENCES:\n"
        preferences_text = preferences_data.get('preferences', 'No preferences available.')
        output += preferences_text

        output += "\n" + "=" * 60

        return output

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass of the agent.

        TODO: Implement the main workflow for Adjudicator:
        - Receive rankings from Intuitor and Reasoner
        - Facilitate debate between them
        - Ask clarifying questions
        - Make final ranking decision
        """
        raise NotImplementedError("Adjudicator.forward() will be implemented later")

    def invoke(self, argument: Any, json_mode: bool) -> str:
        """
        Invoke the agent with the argument.

        TODO: Implement how Adjudicator processes input arguments
        """
        raise NotImplementedError("Adjudicator.invoke() will be implemented later")



    def _format_item_summary(self, item_info: Dict[str, Any]) -> str:
        """
        Format item information into a concise summary.

        Args:
            item_info: Dictionary containing item details

        Returns:
            Formatted summary string
        """
        summary_parts = []

        # Main category
        if 'main_category' in item_info:
            summary_parts.append(f"Category: {item_info['main_category']}")

        # Title
        if 'title' in item_info:
            summary_parts.append(f"Title: {item_info['title']}")

        # Key features (limit to 3)
        if 'features' in item_info and item_info['features']:
            features = item_info['features'][:3]
            features_str = '; '.join(features)
            if len(features_str) > 150:
                features_str = features_str[:147] + '...'
            summary_parts.append(f"Features: {features_str}")

        # Price
        if 'price' in item_info and item_info['price']:
            summary_parts.append(f"Price: {item_info['price']}")

        return '\n'.join(summary_parts)



    def _update_core_preferences(self, core_updates: Dict[str, Any]) -> None:
        """
        Update core preferences based on high-confidence reflection.

        Args:
            core_updates: Dictionary containing modification suggestions
        """
        modifications = core_updates.get('modifications', [])

        if not modifications:
            return

        self.observation(f"Applying {len(modifications)} core preference updates...")

        # ✅ 新版本：preferences 是字符串，不能追加到数组
        # 而是应该更新字符串内容
        current_preferences = self.user_preferences.get('preferences', '')

        # 将修改内容添加到 preferences 字符串的末尾
        for modification in modifications:
            current_preferences += f" Additionally, {modification}."

        self.user_preferences['preferences'] = current_preferences
        self.observation("Core preferences updated")

    def _get_timestamp(self) -> str:
        """Get current timestamp for memory entries."""
        from datetime import datetime
        return datetime.now().isoformat()

    def add_adversarial_memory_entry(
            self,
            summary: str,
            details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录一次对抗过程的反思/反馈（供后续 Prompt 使用）

        Args:
            summary: 面向 LLM 的简短自然语言总结
            details: 可选的结构化信息（如本轮 hard 样本统计等）
        """
        if not hasattr(self, "adversarial_memory"):
            self.adversarial_memory = []

        entry = {
            "timestamp": self._get_timestamp(),
            "summary": summary,
            "details": details or {},
        }
        self.adversarial_memory.append(entry)

        # 只保留最近 N 条
        if len(self.adversarial_memory) > getattr(self, "max_adversarial_memory_entries", 10):
            self.adversarial_memory = self.adversarial_memory[-self.max_adversarial_memory_entries:]

    def _format_adversarial_memory_for_prompt(self) -> str:
        """
        将对抗记忆格式化成 Prompt 中的一段文本。

        - 如果没有记忆：返回一段说明文字
        - 如果有：列出最近几轮的 summary
        """
        if not getattr(self, "adversarial_memory", None):
            return (
                "No previous adversarial training rounds are available. "
                "You are generating near-boundary noise for this user for the first time."
            )

        lines: List[str] = []
        lines.append("Recent adversarial rounds and reflections:")
        # 只展示最近 5 条，避免 Prompt 过长
        recent_entries = self.adversarial_memory[-5:]
        for idx, entry in enumerate(recent_entries, 1):
            ts = entry.get("timestamp", "")
            summary = entry.get("summary", "").strip()
            if ts:
                lines.append(f"{idx}. [{ts}] {summary}")
            else:
                lines.append(f"{idx}. {summary}")

        return "\n".join(lines)

    def _format_discriminator_feedback_for_prompt(
            self,
            feedback: Dict[str, Any],
            max_examples_per_type: int = 10,
    ) -> str:
        """
        将判别器的对抗评估结果(metrics)格式化成可读文本，放到反思 Prompt 里。

        参数:
            feedback: 来自 adversarial_eval_and_train 的 metrics 字典
            max_examples_per_type: 每类错误例子最多展示多少条，避免 Prompt 爆炸

        返回:
            一段多行字符串，包含整体统计 + 若干条样例
        """
        lines: List[str] = []

        acc = feedback.get("accuracy", None)
        tp = feedback.get("tp", None)
        tn = feedback.get("tn", None)
        fp = feedback.get("fp", None)
        fn = feedback.get("fn", None)
        avg_p_clean = feedback.get("avg_pred_noise_prob_clean", None)
        avg_p_noise = feedback.get("avg_pred_noise_prob_noise", None)

        lines.append("Overall discriminator feedback on this batch:")
        if acc is not None:
            lines.append(f"- Accuracy: {acc:.4f}")
        if tp is not None and tn is not None and fp is not None and fn is not None:
            lines.append(f"- Confusion matrix (noise=1, clean=0):")
            lines.append(f"    TP (noise correctly detected)   : {tp}")
            lines.append(f"    TN (clean correctly accepted)   : {tn}")
            lines.append(f"    FP (clean misclassified as noise): {fp}")
            lines.append(f"    FN (noise misclassified as clean): {fn}")
        if avg_p_clean is not None and avg_p_noise is not None:
            lines.append(f"- Avg predicted noise prob on CLEAN (label=0): {avg_p_clean:.4f}")
            lines.append(f"- Avg predicted noise prob on NOISE (label=1): {avg_p_noise:.4f}")

        # 逐样本结果
        per_sample: List[Dict[str, Any]] = feedback.get("per_sample_results", [])
        if not per_sample:
            return "\n".join(lines)

        # 分类：TN, TP, FP, FN
        tn_cases, tp_cases, fp_cases, fn_cases = [], [], [], []
        for r in per_sample:
            y = r.get("true_label")
            y_hat = r.get("pred_label")
            if y == 0 and y_hat == 0:
                tn_cases.append(r)
            elif y == 1 and y_hat == 1:
                tp_cases.append(r)
            elif y == 0 and y_hat == 1:
                fp_cases.append(r)
            elif y == 1 and y_hat == 0:
                fn_cases.append(r)

        # 为了便于 Agent 理解，辅助加上物品名称（如果能加载到）
        dataset_name = self.dataset if isinstance(self.dataset, str) else "dbbook2014"
        try:
            item_mapping = self._load_item_mapping(dataset_name)
        except Exception:
            item_mapping = {}

        def _format_case_list(title: str, cases: List[Dict[str, Any]]) -> None:
            if not cases:
                lines.append(f"\n{title}: (none)")
                return
            lines.append(f"\n{title}: (showing up to {max_examples_per_type} examples)")
            for idx, r in enumerate(cases[:max_examples_per_type], 1):
                uid = r.get("user_id")
                iid = r.get("item_id")
                prob = r.get("pred_noise_prob", None)
                name = item_mapping.get(int(iid), f"Item {iid}")
                y = r.get("true_label")
                y_hat = r.get("pred_label")
                lines.append(
                    f"{idx}. user={uid}, item={iid} ({name}), "
                    f"true_label={y}, pred_label={y_hat}, "
                    f"pred_noise_prob={prob:.4f}" if prob is not None
                    else f"{idx}. user={uid}, item={iid} ({name}), true_label={y}, pred_label={y_hat}"
                )

        _format_case_list("False NEGATIVES (noise=1 misclassified as clean)", fn_cases)
        _format_case_list("False POSITIVES (clean=0 misclassified as noise)", fp_cases)
        _format_case_list("True POSITIVES (noise correctly detected)", tp_cases)
        _format_case_list("True NEGATIVES (clean correctly accepted)", tn_cases)

        return "\n".join(lines)
    def _load_item_mapping(self, dataset_name: str) -> Dict[int, str]:
        """
        从 item_list.txt 加载物品 ID 到名称的映射
        格式: org_id remap_id entity_name

        Args:
            dataset_name: 数据集名称

        Returns:
            {remap_id: entity_name} 的字典
        """
        if not hasattr(self, '_item_mapping_cache'):
            self._item_mapping_cache = {}

        # 如果缓存中已有该数据集的映射，直接返回
        if dataset_name in self._item_mapping_cache:
            return self._item_mapping_cache[dataset_name]

        # 构建 item_list.txt 路径
        current_dir = Path(__file__).parent  # agents/
        project_root = current_dir.parent  # mycode/
        item_list_file = project_root / "data" / dataset_name / "item_list.txt"

        item_mapping = {}

        try:
            with open(item_list_file, 'r', encoding='utf-8') as f:
                # 跳过表头
                header = f.readline()

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 3:
                        continue

                    try:
                        org_id = int(parts[0])
                        remap_id = int(parts[1])

                        # 获取 entity_name（可能有多个单词）
                        entity_uri = " ".join(parts[2:])

                        # 处理 entity_uri：去掉前缀和下划线，提取最后的名称
                        # 例如：http://dbpedia.org/resource/Dragonfly_in_Amber -> Dragonfly in Amber
                        if entity_uri.startswith('http://dbpedia.org/resource/'):
                            entity_name = entity_uri.replace('http://dbpedia.org/resource/', '')
                            entity_name = entity_name.replace('_', ' ')
                        else:
                            entity_name = entity_uri

                        item_mapping[remap_id] = entity_name

                    except (ValueError, IndexError):
                        continue

            self.observation(f"✓ Loaded {len(item_mapping)} items from {item_list_file}")

        except FileNotFoundError:
            self.observation(f"Warning: {item_list_file} not found. Item mapping will be empty.")

        # 缓存起来
        self._item_mapping_cache[dataset_name] = item_mapping

        return item_mapping

    def _format_candidate_items_with_info(self, candidate_items: List[int], dataset_name: str) -> str:
        """
        将候选物品 ID 转换为带有名称的文本格式

        Args:
            candidate_items: 候选物品 ID 列表
            dataset_name: 数据集名称

        Returns:
            格式化的候选物品文本，例如：
            1. Item 0: Dragonfly in Amber
            2. Item 1: Drums of Autumn
            ...
        """
        # 加载物品映射
        item_mapping = self._load_item_mapping(dataset_name)

        formatted_items = []
        for idx, item_id in enumerate(candidate_items, 1):
            # 从映射中获取物品名称，如果找不到就用占位符
            item_name = item_mapping.get(item_id, f"Unknown Item {item_id}")
            formatted_items.append(f"{idx}. Item {item_id}: {item_name}")

        return "\n".join(formatted_items)

    def _load_recommender_model(self) -> None:
        """
        加载训练好的推荐模型，生成全量排名。

        Expected directory structure:
        recommenders/weights/{dataset}/{modeltype}/
            - args.txt                           (模型参数)
            - epoch=X.checkpoint.pth.tar        (权重文件)
        """
        import torch
        import re

        print("\n" + "=" * 80)
        print(f"Loading recommender model: {self.modeltype} on {self.dataset}")
        print("=" * 80)

        try:
            # ---- Step 1: 验证模型路径 ----
            model_dir = Path(
                __file__).parent.parent / "recommenders" / "weights" / self.dataset / self.modeltype / self.model_path

            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")

            print(f"✓ Model directory found: {model_dir}")

            # ---- Step 2: 加载模型参数 ----
            args_file = model_dir / "args.txt"
            if not args_file.exists():
                raise FileNotFoundError(f"args.txt not found in {model_dir}")

            with open(args_file, 'r') as f:
                args_dict = json.load(f)

            from argparse import Namespace
            saved_args = Namespace(**args_dict)
            print(f"✓ Loaded model arguments")

            # ---- Step 3: 判断是否需要加载模型权重 ----
            if self.modeltype.lower() in ['random', 'pop']:
                print(f"✓ Model type '{self.modeltype}' doesn't require weight loading (non-parametric)")
                self.recommender_model = None
                self._generate_rankings_without_model()
                return

            # ---- Step 4: 加载权重 ----
            checkpoint_files = list(model_dir.glob("epoch=*.checkpoint.pth.tar"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {model_dir}")

            # 获取最新的检查点（epoch 最大的）
            epochs = []
            for cp in checkpoint_files:
                match = re.search(r'epoch=(\d+)', cp.name)
                if match:
                    epochs.append((int(match.group(1)), cp))

            if not epochs:
                raise ValueError("Could not parse epoch numbers from checkpoint files")

            latest_epoch, latest_checkpoint = max(epochs, key=lambda x: x[0])
            print(f"✓ Found checkpoint: epoch={latest_epoch}")

            # ---- Step 5: 加载并初始化模型 ----
            print(f"Initializing {self.modeltype} model...")

            # 这里需要导入对应的模型类
            # 假设你的模型在 recommenders/models 中
            try:
                if self.modeltype == 'LightGCN':
                    from recommenders.models.LightGCN import LightGCN, LightGCN_Data
                    data = LightGCN_Data(saved_args)
                    model = LightGCN(saved_args, data)
                elif self.modeltype == 'NGCF':
                    from recommenders.models.NGCF import NGCF, NGCF_Data
                    data = NGCF_Data(saved_args)
                    model = NGCF(saved_args, data)
                elif self.modeltype == 'NCF':
                    from recommenders.models.NCF import NCF, NCF_Data
                    data = NCF_Data(saved_args)
                    model = NCF(saved_args, data)
                elif self.modeltype == 'MF':
                    from recommenders.models.MF import MF, MF_Data
                    data = MF_Data(saved_args)
                    model = MF(saved_args, data)
                else:
                    raise ValueError(f"Unsupported model type: {self.modeltype}")

                self.data = data
                print(f"✓ Model initialized")

            except ImportError as e:
                print(f"✗ Failed to import {self.modeltype}: {e}")
                raise

            # ---- Step 6: 加载权重到模型 ----
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()

            checkpoint = torch.load(str(latest_checkpoint), map_location=device)
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print(f"✓ Weights loaded from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load state dict with strict=True: {e}")
                print(f"Attempting to load with compatible keys only...")

            self.recommender_model = model
            self.device = device

            # ---- Step 7: 生成全量排名 ----
            print(f"Generating full rankings for all users...")
            self._generate_full_rankings()

            print("=" * 80)
            print(f"✓ Recommender model loaded successfully")
            print("=" * 80 + "\n")

        except Exception as e:
            print(f"✗ Failed to load recommender model: {e}")
            print(f"Will fall back to random sampling\n")
            import traceback
            traceback.print_exc()
            self.recommender_model = None
            self.full_rankings = None

    def _generate_full_rankings(self) -> None:
        """
        使用加载的推荐模型为所有用户生成排名。
        存储为 self.full_rankings: shape (n_users, n_items)
        """
        import torch
        import numpy as np

        if self.recommender_model is None:
            print("Warning: No recommender model available for ranking generation")
            return

        self.recommender_model.eval()
        n_users = self.data.n_users
        n_items = self.data.n_items

        # 获取用户已交互的物品（用于置 -inf）
        from pathlib import Path
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        train_file = project_root / "data" / self.dataset / "train.txt"

        user_items_dict = {}
        try:
            with open(train_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        user_id = int(parts[0])
                        items = [int(x) for x in parts[1:]]
                        user_items_dict[user_id] = set(items)
        except FileNotFoundError:
            print(f"Warning: train.txt not found, won't filter interacted items")

        # 生成评分矩阵
        score_matrix = np.zeros((n_users, n_items))
        batch_size = 256

        with torch.no_grad():
            for start_user_id in range(0, n_users, batch_size):
                end_user_id = min(start_user_id + batch_size, n_users)
                batch_user_ids = list(range(start_user_id, end_user_id))

                # 调用模型的 predict 方法获取评分
                try:
                    batch_scores = self.recommender_model.predict(batch_user_ids, None)
                except Exception as e:
                    print(f"Warning: Failed to predict for user batch {start_user_id}-{end_user_id}: {e}")
                    # 如果预测失败，使用随机分数
                    batch_scores = np.random.randn(len(batch_user_ids), n_items)

                # 确保是 numpy 数组
                if not isinstance(batch_scores, np.ndarray):
                    batch_scores = np.array(batch_scores, dtype=np.float32)

                # 将已交互物品设为 -inf（这样排序时会排到后面）
                for idx, user_id in enumerate(batch_user_ids):
                    interacted = user_items_dict.get(user_id, set())
                    for item_id in interacted:
                        batch_scores[idx, item_id] = -np.inf

                score_matrix[start_user_id:end_user_id] = batch_scores

        # 按分数从高到低排序，得到每个用户的排名（item_id）
        # argsort(-score) 会给出从高到低的索引
        self.full_rankings = np.argsort(-score_matrix, axis=1)  # shape: (n_users, n_items)
        print(f"✓ Generated full rankings: {self.full_rankings.shape}")

    def _generate_rankings_without_model(self) -> None:
        """
        对于 Random 和 Pop 模型，直接生成排名而无需加载权重。

        Pop: 按全局流行度排序
        Random: 随机排序
        """
        import numpy as np
        from pathlib import Path

        n_users = self.data.n_users if hasattr(self, 'data') else 2680  # 默认值
        n_items = self.data.n_items if hasattr(self, 'data') else 2680

        if self.modeltype.lower() == 'pop':
            # ---- Pop: 按流行度排序 ----
            print(f"Generating popularity-based rankings...")

            # 加载流行度信息
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            popularity_file = project_root / "data" / self.dataset / "item_popularity.txt"

            item_popularity = {}
            try:
                with open(popularity_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            item_id = int(parts[0])
                            popularity = int(parts[1])
                            item_popularity[item_id] = popularity
            except FileNotFoundError:
                print(f"Warning: popularity file not found, using equal popularity")
                item_popularity = {i: 1 for i in range(n_items)}

            # 为所有用户生成相同的流行度排序
            popularity_scores = np.array([item_popularity.get(i, 0) for i in range(n_items)])
            ranking = np.argsort(-popularity_scores)  # 从高到低

            self.full_rankings = np.tile(ranking, (n_users, 1))  # 广播到所有用户
            print(f"✓ Generated pop-based rankings: {self.full_rankings.shape}")

        elif self.modeltype.lower() == 'random':
            # ---- Random: 随机排序 ----
            print(f"Generating random rankings...")

            self.full_rankings = np.zeros((n_users, n_items), dtype=int)
            for user_id in range(n_users):
                self.full_rankings[user_id] = np.random.permutation(n_items)

            print(f"✓ Generated random rankings: {self.full_rankings.shape}")

    def _sample_candidate_items(
            self,
            user_id: int,
            num_candidates: int = 50,
    ) -> List[int]:
        """
        采样候选物品。支持两种方式：

        1. 使用推荐模型（如果可用）：
           - 从推荐模型生成的排名中采样前 N 个未交互的物品
           - 保证了候选物品的相关性

        2. 随机采样（fallback）：
           - 从用户未交互的物品中随机采样
           - 当推荐模型不可用时使用

        Args:
            user_id: 用户 ID
            num_candidates: 采样的候选物品数量，默认 50

        Returns:
            候选物品 ID 列表

        Raises:
            ValueError: 如果用户 ID 超出范围
        """
        import numpy as np

        # ---- 参数验证 ----
        if user_id < 0:
            raise ValueError(f"Invalid user_id: {user_id}")

        # ---- 方式 1：使用推荐模型采样 ----
        if self.use_recommender and self.full_rankings is not None:
            try:
                # 检查 user_id 是否在有效范围内
                if user_id >= len(self.full_rankings):
                    print(f"Warning: user_id {user_id} exceeds model's user range {len(self.full_rankings)}")
                    # 回退到随机采样
                    return self._sample_candidate_items_random(user_id, num_candidates)

                # 获取该用户的排名（从高到低）
                user_ranking = self.full_rankings[user_id]  # shape: (n_items,)

                # 加载用户已交互的物品集合
                interacted_items = self._get_user_interacted_items(user_id)

                # 从排名中筛选出未交互的物品，取前 num_candidates 个
                candidates = []
                for item_id in user_ranking:
                    if item_id not in interacted_items and item_id not in candidates:
                        candidates.append(int(item_id))
                        if len(candidates) >= num_candidates:
                            break

                # 如果候选物品不足，使用随机采样补充
                if len(candidates) < num_candidates:
                    print(f"Warning: Only got {len(candidates)} candidates from recommender model")
                    additional = self._sample_candidate_items_random(
                        user_id,
                        num_candidates - len(candidates),
                        exclude_items=set(candidates) | interacted_items
                    )
                    candidates.extend(additional)

                return candidates[:num_candidates]

            except Exception as e:
                print(f"Error in model-based sampling: {e}")
                print(f"Falling back to random sampling")
                return self._sample_candidate_items_random(user_id, num_candidates)

        # ---- 方式 2：随机采样（默认/回退） ----
        else:
            return self._sample_candidate_items_random(user_id, num_candidates)

    def _sample_candidate_items_random(
            self,
            user_id: int,
            num_candidates: int = 50,
            exclude_items: Optional[set] = None,
    ) -> List[int]:
        """
        随机采样候选物品（不使用推荐模型）。

        Args:
            user_id: 用户 ID
            num_candidates: 采样的候选物品数量
            exclude_items: 需要排除的物品集合（已交互或已添加到候选集）

        Returns:
            候选物品 ID 列表
        """
        import numpy as np

        # ---- 获取用户已交互的物品 ----
        if exclude_items is None:
            exclude_items = self._get_user_interacted_items(user_id)

        # ---- 获取所有可用物品 ----
        # 假设从 system 或 dataset 中获取物品数量
        n_items = self.get_n_items()
        all_items = set(range(n_items))

        # ---- 计算可采样的物品集合 ----
        available_items = list(all_items - exclude_items)

        if len(available_items) == 0:
            print(f"Warning: No available items for user {user_id}")
            return []

        # ---- 采样 ----
        num_to_sample = min(num_candidates, len(available_items))
        candidates = np.random.choice(available_items, size=num_to_sample, replace=False)

        return list(candidates)

    def _get_user_interacted_items(self, user_id: int) -> set:
        """
        获取用户已交互的物品集合。

        Args:
            user_id: 用户 ID

        Returns:
            已交互的物品 ID 集合

        Notes:
            从 train.txt 文件加载，或从 system 的数据结构中获取。
            结果会被缓存以提高性能。
        """
        # 如果已缓存，直接返回
        if not hasattr(self, '_user_interacted_cache'):
            self._user_interacted_cache = {}

        if user_id in self._user_interacted_cache:
            return self._user_interacted_cache[user_id]

        # ---- 尝试从 system 获取 ----
        try:
            if hasattr(self.system, 'train_interaction_data'):
                if user_id in self.system.train_interaction_data:
                    items = set(self.system.train_interaction_data[user_id])
                    self._user_interacted_cache[user_id] = items
                    return items
        except (AttributeError, TypeError, KeyError):
            pass

        # ---- 尝试从 train.txt 文件加载 ----
        from pathlib import Path

        try:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent
            train_file = project_root / "data" / self.dataset / "train.txt"

            if train_file.exists():
                with open(train_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            uid = int(parts[0])
                            if uid == user_id and len(parts) > 1:
                                items = set(int(x) for x in parts[1:])
                                self._user_interacted_cache[user_id] = items
                                return items
        except (FileNotFoundError, ValueError, IndexError):
            pass

        # ---- 回退：返回空集合 ----
        print(f"Warning: Could not find interacted items for user {user_id}")
        self._user_interacted_cache[user_id] = set()
        return set()

    def get_n_items(self) -> int:
        """
        获取物品总数。

        Returns:
            物品总数

        Notes:
            优先级：
            1. self.data.n_items (推荐模型已加载)
            2. self.system.n_items (从 system 获取)
            3. 从数据文件推断
        """
        # ---- 尝试从已加载的数据获取 ----
        if hasattr(self, 'data') and hasattr(self.data, 'n_items'):
            return self.data.n_items

        # ---- 尝试从 system 获取 ----
        if hasattr(self.system, 'n_items'):
            return self.system.n_items

        # ---- 从数据文件推断 ----
        from pathlib import Path

        try:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent

            # 方法 1：从 item_id_map.txt 获取
            item_map_file = project_root / "data" / self.dataset / "item_id_map.txt"
            if item_map_file.exists():
                with open(item_map_file, 'r') as f:
                    n_items = len(f.readlines())
                return n_items

            # 方法 2：从 train.txt 的最大 item_id 推断
            train_file = project_root / "data" / self.dataset / "train.txt"
            if train_file.exists():
                max_item_id = -1
                with open(train_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 1:
                            for item_id in map(int, parts[1:]):
                                max_item_id = max(max_item_id, item_id)
                if max_item_id >= 0:
                    return max_item_id + 1

        except (FileNotFoundError, ValueError, IndexError):
            pass

        # ---- 回退值 ----
        print(f"Warning: Could not determine n_items, using default 2680")
        return 2680

    def get_n_users(self) -> int:
        """
        获取用户总数。

        Returns:
            用户总数
        """
        # ---- 尝试从已加载的数据获取 ----
        if hasattr(self, 'data') and hasattr(self.data, 'n_users'):
            return self.data.n_users

        # ---- 尝试从 system 获取 ----
        if hasattr(self.system, 'n_users'):
            return self.system.n_users

        # ---- 从数据文件推断 ----
        from pathlib import Path

        try:
            current_dir = Path(__file__).parent
            project_root = current_dir.parent

            # 方法 1：从 user_id_map.txt 获取
            user_map_file = project_root / "data" / self.dataset / "user_id_map.txt"
            if user_map_file.exists():
                with open(user_map_file, 'r') as f:
                    n_users = len(f.readlines())
                return n_users

            # 方法 2：从 train.txt 的最大 user_id 推断
            train_file = project_root / "data" / self.dataset / "train.txt"
            if train_file.exists():
                max_user_id = -1
                with open(train_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            user_id = int(parts[0])
                            max_user_id = max(max_user_id, user_id)
                if max_user_id >= 0:
                    return max_user_id + 1

        except (FileNotFoundError, ValueError, IndexError):
            pass

        # ---- 回退值 ----
        print(f"Warning: Could not determine n_users, using default 2680")
        return 2680

    def generate_misclick_noise(
            self,
            init_data: str,
            noise_ratio: float,
    ) -> Dict[str, Any]:
        """
        Generate synthetic misclick (accidental click) noise interactions for the current user.
        Misclick noise is completely independent of user preferences.
        """
        # 1. 校验初始化状态
        if not getattr(self, "is_initialized", False):
            raise RuntimeError(
                "Adjudicator must be initialized with _process_history() "
                "and is_initialized must be set to True before generating noise."
            )

        if not init_data:
            raise ValueError("init_data cannot be empty when generating misclick noise.")

        if self.user_id is None:
            raise ValueError("self.user_id is not set. Cannot generate user-specific noise.")

        # 2. 检查对应的 prompt 是否存在
        if "misclick_noise_prompt" not in self.prompts:
            raise ValueError("misclick_noise_prompt not found in prompts config")

        prompt_template = self.prompts["misclick_noise_prompt"]

        # ✅ 如果是 PromptTemplate 对象，获取 template 属性
        if hasattr(prompt_template, 'template'):
            prompt_template = prompt_template.template
        # 如果是嵌套的 dict（有 type 和 content），提取 content
        elif isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("content", "")

        # 3. 获取数据集信息
        dataset_item_counts = {
            'dbbook2014': 2680,
            'book-crossing': 8853,
            'ml1m': 3260,
            'gowalla': 107092,
            'yelp2018': 20033,
            'amazon-book': 96978,
            'lastfm': 11946,
        }

        if isinstance(self.dataset, str):
            dataset_name = self.dataset
        else:
            dataset_name = 'dbbook2014'

        total_items = dataset_item_counts.get(dataset_name, 2680)
        max_item_id = total_items - 1

        # 4. 采样候选物品
        candidate_items = self._sample_candidate_items(self.user_id, num_candidates=50)

        # ✅ 获取物品的详细信息（ID + 名称）
        candidate_items_info = self._format_candidate_items_with_info(candidate_items, dataset_name)

        # ✅ 创建候选物品 ID 的列表字符串，用于约束 LLM
        candidate_item_ids_str = ", ".join(str(item_id) for item_id in candidate_items)

        self.observation(f"Candidate items prepared with descriptions")

        # 5. 使用 replace 而不是 format()，避免双大括号问题
        final_prompt = prompt_template.replace(
            "{interaction_history}", init_data
        ).replace(
            "{noise_ratio}", str(noise_ratio)
        ).replace(
            "{user_id}", str(self.user_id)
        ).replace(
            "{total_items}", str(total_items)
        ).replace(
            "{max_item_id}", str(max_item_id)
        ).replace(
            "{candidate_items}", candidate_items_info
        ).replace(
            "{candidate_item_ids}", candidate_item_ids_str  # ✅ 添加纯 ID 列表用于约束
        )

        # 6. 调用 LLM
        self.observation(
            f"Calling LLM to generate MISCLICK noise interactions "
            f"for user {self.user_id} with noise_ratio={noise_ratio}..."
        )
        self.observation(
            f"Candidate items for selection: {len(candidate_items)} items"
        )
        response = self.llm(final_prompt)

        # 7. 清洗 LLM 返回
        try:
            clean_response = response.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('```')[1]
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            noise_data = json.loads(clean_response)

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse misclick noise LLM response as JSON: {e}")
            self.observation(f"Raw response: {response}")
            raise

        # 8. 基本结构校验
        if not isinstance(noise_data, dict):
            raise ValueError(
                f"Expected top-level JSON object for misclick noise, got {type(noise_data)}"
            )

        if "noise_interactions" not in noise_data:
            raise ValueError(
                "LLM misclick noise response missing 'noise_interactions' field. "
                f"Got keys: {list(noise_data.keys())}"
            )

        interactions = noise_data.get("noise_interactions", [])

        # ✅ 验证所有 item_id 都在候选物品中
        candidate_set = set(candidate_items)
        invalid_items = []
        for inter in interactions:
            item_id = inter.get("item_id")
            if item_id is not None and item_id not in candidate_set:
                invalid_items.append(item_id)

        if invalid_items:
            self.observation(
                f"Warning: Generated {len(invalid_items)} item_ids not in candidate list: {invalid_items}"
            )

        self.observation(
            f"Generated {len(interactions)} misclick noise interactions "
            f"for user {self.user_id}."
        )

        # 组装返回格式
        noise_item_ids = []
        for inter in interactions:
            item_id = inter.get("item_id")
            if item_id is not None:
                noise_item_ids.append(item_id)

        result = {
            "user_id": self.user_id,
            "noise_type": "misclick",
            "noise_items": noise_item_ids,
        }

        return result

    def generate_curiosity_noise(
            self,
            init_data: str,
            noise_ratio: float,
    ) -> Dict[str, Any]:
        """
        Generate synthetic curiosity-driven noise interactions for the current user.
        """
        # 1. 校验初始化状态
        if not getattr(self, "is_initialized", False):
            raise RuntimeError(
                "Adjudicator must be initialized with _process_history() "
                "and is_initialized must be set to True before generating noise."
            )

        if not self.user_preferences or not self.user_preferences.get("preferences"):
            raise ValueError(
                "User preferences not found. Please run _process_history() first."
            )

        if not init_data:
            raise ValueError("init_data cannot be empty when generating curiosity noise.")

        if self.user_id is None:
            raise ValueError("self.user_id is not set. Cannot generate user-specific noise.")

        # 2. 检查对应的 prompt 是否存在
        if "curiosity_noise_prompt" not in self.prompts:
            raise ValueError("curiosity_noise_prompt not found in prompts config")

        prompt_template = self.prompts["curiosity_noise_prompt"]

        # ✅ 如果是 PromptTemplate 对象，获取 template 属性
        if hasattr(prompt_template, 'template'):
            prompt_template = prompt_template.template
        # 如果是嵌套的 dict（有 type 和 content），提取 content
        elif isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("content", "")

        # 3. 准备提示词所需信息
        user_preferences_summary = self._format_preferences_for_prompt()

        # 获取数据集信息
        dataset_item_counts = {
            'dbbook2014': 2680,
            'book-crossing': 8853,
            'ml1m': 3260,
            'gowalla': 107092,
            'yelp2018': 20033,
            'amazon-book': 96978,
            'lastfm': 11946,
        }

        if isinstance(self.dataset, str):
            dataset_name = self.dataset
        else:
            dataset_name = 'dbbook2014'

        total_items = dataset_item_counts.get(dataset_name, 2680)
        max_item_id = total_items - 1

        # 采样候选物品
        candidate_items = self._sample_candidate_items(self.user_id, num_candidates=50)

        # ✅ 获取物品的详细信息（ID + 名称）
        candidate_items_info = self._format_candidate_items_with_info(candidate_items, dataset_name)

        # ✅ 创建候选物品 ID 的列表字符串，用于约束 LLM
        candidate_item_ids_str = ", ".join(str(item_id) for item_id in candidate_items)

        self.observation(f"Candidate items prepared with descriptions")

        # ✅ 使用 replace 而不是 format()，避免双大括号问题
        final_prompt = prompt_template.replace(
            "{interaction_history}", init_data
        ).replace(
            "{user_preferences_summary}", user_preferences_summary
        ).replace(
            "{noise_ratio}", str(noise_ratio)
        ).replace(
            "{user_id}", str(self.user_id)
        ).replace(
            "{total_items}", str(total_items)
        ).replace(
            "{max_item_id}", str(max_item_id)
        ).replace(
            "{candidate_items}", candidate_items_info
        ).replace(
            "{candidate_item_ids}", candidate_item_ids_str  # ✅ 添加纯 ID 列表用于约束
        )

        # 4. 调用 LLM
        self.observation(
            f"Calling LLM to generate CURIOSITY noise interactions "
            f"for user {self.user_id} with noise_ratio={noise_ratio}..."
        )
        self.observation(
            f"Candidate items for selection: {len(candidate_items)} items"
        )
        response = self.llm(final_prompt)

        # 5. 清洗 LLM 返回
        try:
            clean_response = response.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('```')[1]
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            noise_data = json.loads(clean_response)

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse curiosity noise LLM response as JSON: {e}")
            self.observation(f"Raw response: {response}")
            raise

        # 6. 基本结构校验
        if not isinstance(noise_data, dict):
            raise ValueError(
                f"Expected top-level JSON object for curiosity noise, got {type(noise_data)}"
            )

        if "noise_interactions" not in noise_data:
            raise ValueError(
                "LLM curiosity noise response missing 'noise_interactions' field. "
                f"Got keys: {list(noise_data.keys())}"
            )

        interactions = noise_data.get("noise_interactions", [])

        # ✅ 验证所有 item_id 都在候选物品中
        candidate_set = set(candidate_items)
        invalid_items = []
        for inter in interactions:
            item_id = inter.get("item_id")
            if item_id is not None and item_id not in candidate_set:
                invalid_items.append(item_id)

        if invalid_items:
            self.observation(
                f"Warning: Generated {len(invalid_items)} item_ids not in candidate list: {invalid_items}"
            )

        self.observation(
            f"Generated {len(interactions)} curiosity noise interactions "
            f"for user {self.user_id}."
        )

        # 组装返回格式
        noise_item_ids = []
        for inter in interactions:
            item_id = inter.get("item_id")
            if item_id is not None:
                noise_item_ids.append(item_id)

        result = {
            "user_id": self.user_id,
            "noise_type": "curiosity",
            "noise_items": noise_item_ids,
        }

        return result

    def generate_caption_bias_noise(
            self,
            init_data: str,
            noise_ratio: float,
    ) -> Dict[str, Any]:
        """
        Generate synthetic CAPTION/TITLE BIAS noise interactions for the current user.

        Caption/Snippet Bias:
        - Clicks are driven by attractive titles, famous IPs, or intriguing wording,
          rather than genuine long-term interest.
        - 模拟“看封面就点进去”的行为。

        Args:
            init_data: 用户的交互历史文本（_process_history 里构造好的那种）
            noise_ratio: 噪声比例，控制生成多少条噪声交互

        Returns:
            {
                "user_id": int,
                "noise_type": "caption_bias",
                "noise_items": [item_id1, item_id2, ...],
                "bias_pattern": "<LLM 总结的标题偏好模式，可选>"
            }
        """
        # 1. 校验初始化状态
        if not getattr(self, "is_initialized", False):
            raise RuntimeError(
                "Adjudicator must be initialized with _process_history() "
                "and is_initialized must be set to True before generating noise."
            )

        if not self.user_preferences or not self.user_preferences.get("preferences"):
            raise ValueError(
                "User preferences not found. Please run _process_history() first."
            )

        if not init_data:
            raise ValueError("init_data cannot be empty when generating caption bias noise.")

        if self.user_id is None:
            raise ValueError("self.user_id is not set. Cannot generate user-specific noise.")

        # 2. 检查对应的 prompt 是否存在
        if "caption_bias_noise_prompt" not in self.prompts:
            raise ValueError("caption_bias_noise_prompt not found in prompts config")

        prompt_template = self.prompts["caption_bias_noise_prompt"]

        # 如果是 PromptTemplate 对象，获取 template 属性
        if hasattr(prompt_template, "template"):
            prompt_template = prompt_template.template
        # 如果是嵌套的 dict（有 type 和 content），提取 content
        elif isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("content", "")

        # 3. 准备提示词信息：用户偏好摘要
        user_preferences_summary = self._format_preferences_for_prompt()

        # 获取数据集信息（和其它噪声函数保持一致）
        dataset_item_counts = {
            "dbbook2014": 2680,
            "book-crossing": 8853,
            "ml1m": 3260,
            "gowalla": 107092,
            "yelp2018": 20033,
            "amazon-book": 96978,
            "lastfm": 11946,
        }

        if isinstance(self.dataset, str):
            dataset_name = self.dataset
        else:
            dataset_name = "dbbook2014"

        total_items = dataset_item_counts.get(dataset_name, 2680)
        max_item_id = total_items - 1

        # 4. 采样候选物品（只从“未交互”的物品中采样，因此天然满足
        #    “Do NOT include items from the user's history.”）
        candidate_items = self._sample_candidate_items(self.user_id, num_candidates=50)

        # 带名称的描述，用于 prompt 中帮助 LLM 判断“标题吸引力”
        candidate_items_info = self._format_candidate_items_with_info(candidate_items, dataset_name)

        # 纯 ID 列表，用于硬约束 LLM 输出
        candidate_item_ids_str = ", ".join(str(item_id) for item_id in candidate_items)

        self.observation("Candidate items prepared with descriptions for CAPTION BIAS noise")

        # 5. 组装最终 prompt（用 replace 避免和 JSON 花括号冲突）
        final_prompt = (
            prompt_template.replace("{interaction_history}", init_data)
            .replace("{user_preferences_summary}", user_preferences_summary)
            .replace("{noise_ratio}", str(noise_ratio))
            .replace("{user_id}", str(self.user_id))
            .replace("{total_items}", str(total_items))
            .replace("{max_item_id}", str(max_item_id))
            .replace("{candidate_items}", candidate_items_info)
            .replace("{candidate_item_ids}", candidate_item_ids_str)
        )

        # 6. 调用 LLM
        self.observation(
            f"Calling LLM to generate CAPTION BIAS noise interactions "
            f"for user {self.user_id} with noise_ratio={noise_ratio}..."
        )
        self.observation(
            f"Candidate items for CAPTION BIAS selection: {len(candidate_items)} items"
        )
        response = self.llm(final_prompt)

        # 7. 清洗 LLM 返回并解析 JSON
        try:
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            noise_data = json.loads(clean_response)

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse caption bias noise LLM response as JSON: {e}")
            self.observation(f"Raw response: {response}")
            raise

        # 8. 基本结构校验
        if not isinstance(noise_data, dict):
            raise ValueError(
                f"Expected top-level JSON object for caption bias noise, got {type(noise_data)}"
            )

        if "noise_interactions" not in noise_data:
            raise ValueError(
                "LLM caption bias noise response missing 'noise_interactions' field. "
                f"Got keys: {list(noise_data.keys())}"
            )

        interactions = noise_data.get("noise_interactions", [])

        # 验证所有 item_id 都在候选物品中
        candidate_set = set(candidate_items)
        invalid_items = []
        noise_item_ids: List[int] = []

        for inter in interactions:
            item_id = inter.get("item_id")
            if item_id is None:
                continue

            if item_id not in candidate_set:
                invalid_items.append(item_id)

            noise_item_ids.append(item_id)

        if invalid_items:
            self.observation(
                f"Warning: Generated {len(invalid_items)} item_ids not in candidate list: {invalid_items}"
            )

        self.observation(
            f"Generated {len(interactions)} caption bias noise interactions "
            f"for user {self.user_id}."
        )

        # 9. 组装返回结果
        result: Dict[str, Any] = {
            "user_id": self.user_id,
            "noise_type": "caption_bias",
            "noise_items": noise_item_ids,
        }

        # 如果 LLM 给了 bias_pattern，就顺手带出来，方便后续分析（不影响现有流水线）
        if isinstance(noise_data.get("bias_pattern"), str):
            result["bias_pattern"] = noise_data["bias_pattern"]

        return result

    def _load_item_popularity(self, dataset_name: str) -> Dict[int, int]:
        """
        从 item_popularity.txt 加载物品的流行度（交互次数）
        文件格式：
            item_id popularity_count

        Args:
            dataset_name: 数据集名称

        Returns:
            {item_id: popularity_count}
        """
        if not hasattr(self, "_item_popularity_cache"):
            self._item_popularity_cache = {}

        if dataset_name in self._item_popularity_cache:
            return self._item_popularity_cache[dataset_name]

        from pathlib import Path

        current_dir = Path(__file__).parent      # agents/
        project_root = current_dir.parent        # mycode/
        popularity_file = project_root / "data" / dataset_name / "item_popularity.txt"

        item_popularity: Dict[int, int] = {}

        try:
            with open(popularity_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 2:
                        continue

                    try:
                        item_id = int(parts[0])
                        count = int(parts[1])
                        item_popularity[item_id] = count
                    except ValueError:
                        continue

            self.observation(
                f"✓ Loaded popularity for {len(item_popularity)} items from {popularity_file}"
            )

        except FileNotFoundError:
            self.observation(
                f"Warning: {popularity_file} not found. Popularity info will be empty (treated as 0)."
            )

        self._item_popularity_cache[dataset_name] = item_popularity
        return item_popularity

    def _format_candidate_items_with_popularity(
            self,
            candidate_items: List[int],
            dataset_name: str,
            item_popularity: Dict[int, int],
    ) -> str:
        """
        将候选物品格式化为：
        1. Item 123: Title: XXX | Popularity: 87
        """
        item_mapping = self._load_item_mapping(dataset_name)
        lines = []

        for idx, item_id in enumerate(candidate_items, 1):
            name = item_mapping.get(item_id, f"Unknown Item {item_id}")
            popularity = item_popularity.get(item_id, 0)
            lines.append(
                f"{idx}. Item {item_id}: {name} | Popularity: {popularity}"
            )

        return "\n".join(lines)

    def generate_popularity_bias_noise(
            self,
            init_data: str,
            noise_ratio: float,
    ) -> Dict[str, Any]:
        """
        Generate synthetic POPULARITY BIAS noise interactions for the current user.

        流行度偏置：
        - 点击主要是因为“很火”“大家都在看”，而不是个人偏好匹配。
        - 这里会优先从全局最热门、且该用户未交互过的物品中选出候选集，
          再让 LLM 按“从众心理 / 社会认同”来模拟点击。

        Args:
            init_data: 用户交互历史文本（_process_history 里构造的那种）
            noise_ratio: 噪声比例，用于指导生成数量

        Returns:
            {
                "user_id": int,
                "noise_type": "popularity_bias",
                "noise_items": [item_id1, item_id2, ...],
                "bias_pattern": "<可选，由 LLM 给出>"
            }
        """
        # 1. 校验初始化状态
        if not getattr(self, "is_initialized", False):
            raise RuntimeError(
                "Adjudicator must be initialized with _process_history() "
                "and is_initialized must be set to True before generating noise."
            )

        if not self.user_preferences or not self.user_preferences.get("preferences"):
            raise ValueError(
                "User preferences not found. Please run _process_history() first."
            )

        if not init_data:
            raise ValueError("init_data cannot be empty when generating popularity bias noise.")

        if self.user_id is None:
            raise ValueError("self.user_id is not set. Cannot generate user-specific noise.")

        # 2. 检查对应的 prompt 是否存在
        if "popularity_bias_noise_prompt" not in self.prompts:
            raise ValueError("popularity_bias_noise_prompt not found in prompts config")

        prompt_template = self.prompts["popularity_bias_noise_prompt"]

        # 如果是 PromptTemplate 对象，获取 template 属性
        if hasattr(prompt_template, "template"):
            prompt_template = prompt_template.template
        elif isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("content", "")

        # 3. 基本信息：用户偏好摘要 & 数据集信息
        user_preferences_summary = self._format_preferences_for_prompt()

        dataset_item_counts = {
            "dbbook2014": 2680,
            "book-crossing": 8853,
            "ml1m": 3260,
            "gowalla": 107092,
            "yelp2018": 20033,
            "amazon-book": 96978,
            "lastfm": 11946,
        }

        if isinstance(self.dataset, str):
            dataset_name = self.dataset
        else:
            dataset_name = "dbbook2014"

        total_items = dataset_item_counts.get(dataset_name, 2680)
        max_item_id = total_items - 1

        # 4. 加载用户交互缓存（和 _sample_candidate_items 一致）
        from pathlib import Path

        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        train_file = project_root / "data" / dataset_name / "train.txt"

        if not hasattr(self, "_user_interactions_cache"):
            self._user_interactions_cache = {}

        if not hasattr(self, "_all_users_interactions_loaded"):
            self.observation(f"Loading user interactions from {train_file}")
            try:
                with open(train_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 2:
                            continue

                        try:
                            uid = int(parts[0])
                            items = [int(item_id) for item_id in parts[1:]]
                            self._user_interactions_cache[uid] = set(items)
                        except ValueError:
                            continue

                self._all_users_interactions_loaded = True
                self.observation(
                    f"✓ Loaded interactions for {len(self._user_interactions_cache)} users"
                )
            except FileNotFoundError:
                self.observation(
                    f"Warning: {train_file} not found. Using empty interaction set."
                )
                self._all_users_interactions_loaded = True

        user_pos_items = self._user_interactions_cache.get(self.user_id, set())

        # 5. 加载全局流行度信息
        item_popularity = self._load_item_popularity(dataset_name)

        # 6. 构造候选物品：
        #    - 限制为“未交互过”的物品
        #    - 按流行度从高到低排序，取前 N 个（比如 50）
        if item_popularity:
            all_items = set(item_popularity.keys())
        else:
            all_items = set(range(total_items))

        uninteracted_items = list(all_items - user_pos_items)

        if not uninteracted_items:
            self.observation(
                f"Warning: User {self.user_id} has no uninteracted items for popularity bias."
            )
            return {
                "user_id": self.user_id,
                "noise_type": "popularity_bias",
                "noise_items": [],
            }

        # 按 popularity 降序排序
        uninteracted_items.sort(
            key=lambda x: item_popularity.get(x, 0),
            reverse=True,
        )

        num_candidates = min(50, len(uninteracted_items))
        candidate_items = uninteracted_items[:num_candidates]

        # 格式化“物品 + 流行度 + 名称”
        candidate_popularity_table = self._format_candidate_items_with_popularity(
            candidate_items=candidate_items,
            dataset_name=dataset_name,
            item_popularity=item_popularity,
        )

        candidate_item_ids_str = ", ".join(str(item_id) for item_id in candidate_items)

        self.observation(
            f"Prepared {len(candidate_items)} popularity-based candidate items for user {self.user_id}"
        )

        # 7. 组装 final prompt
        final_prompt = (
            prompt_template.replace("{interaction_history}", init_data)
            .replace("{user_preferences_summary}", user_preferences_summary)
            .replace("{noise_ratio}", str(noise_ratio))
            .replace("{user_id}", str(self.user_id))
            .replace("{total_items}", str(total_items))
            .replace("{max_item_id}", str(max_item_id))
            .replace("{candidate_item_ids}", candidate_item_ids_str)
            .replace("{candidate_popularity_table}", candidate_popularity_table)
        )

        # 8. 调用 LLM
        self.observation(
            f"Calling LLM to generate POPULARITY BIAS noise interactions "
            f"for user {self.user_id} with noise_ratio={noise_ratio}..."
        )
        self.observation(
            f"Candidate items for POPULARITY BIAS selection: {len(candidate_items)} items"
        )

        response = self.llm(final_prompt)

        # 9. 解析 JSON
        try:
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            noise_data = json.loads(clean_response)

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse popularity bias noise LLM response as JSON: {e}")
            self.observation(f"Raw response: {response}")
            raise

        if not isinstance(noise_data, dict):
            raise ValueError(
                f"Expected top-level JSON object for popularity bias noise, got {type(noise_data)}"
            )

        if "noise_interactions" not in noise_data:
            raise ValueError(
                "LLM popularity bias noise response missing 'noise_interactions' field. "
                f"Got keys: {list(noise_data.keys())}"
            )

        interactions = noise_data.get("noise_interactions", [])

        candidate_set = set(candidate_items)
        invalid_items = []
        noise_item_ids: List[int] = []

        for inter in interactions:
            item_id = inter.get("item_id")
            if item_id is None:
                continue

            if item_id not in candidate_set:
                invalid_items.append(item_id)

            noise_item_ids.append(item_id)

        if invalid_items:
            self.observation(
                f"Warning: Generated {len(invalid_items)} item_ids not in candidate list: {invalid_items}"
            )

        self.observation(
            f"Generated {len(interactions)} popularity bias noise interactions "
            f"for user {self.user_id}."
        )

        result: Dict[str, Any] = {
            "user_id": self.user_id,
            "noise_type": "popularity_bias",
            "noise_items": noise_item_ids,
        }

        if isinstance(noise_data.get("bias_pattern"), str):
            result["bias_pattern"] = noise_data["bias_pattern"]

        return result

    def _format_positioned_candidate_items(self, candidate_items: List[int], dataset_name: str) -> str:
        """
        将候选物品格式化为强调位置顺序的列表。
        格式示例:
        Position 1. Item 123: Title XXX
        Position 2. Item 456: Title YYY
        ...
        """
        # 1. 加载物品 ID 到名称的映射
        item_mapping = self._load_item_mapping(dataset_name)

        lines = []
        # 2. 遍历并格式化
        for idx, item_id in enumerate(candidate_items, 1):
            name = item_mapping.get(item_id, f"Unknown Item {item_id}")
            # 强调 Position，暗示这是推荐列表的前几名
            lines.append(f"Position {idx}. Item {item_id}: {name}")

        return "\n".join(lines)


    def generate_position_bias_noise(
            self,
            init_data: str,
            noise_ratio: float,
    ) -> Dict[str, Any]:
        """
        Generate synthetic POSITION BIAS noise interactions for the current user.

        位置偏置：
        - 用户点击主要是因为物品在列表靠前（position index 小），
          而不是和用户真实兴趣高度匹配。
        - 我们将构造一个“有顺序的候选列表”，让 LLM 偏向选前面的 item。

        Args:
            init_data: 用户交互历史的文本描述（_process_history 里构造好的）
            noise_ratio: 噪声比例，用于控制大致生成数量

        Returns:
            {
                "user_id": int,
                "noise_type": "position_bias",
                "noise_items": [item_id1, item_id2, ...],
                "bias_pattern": "<可选，由 LLM 总结>"
            }
        """
        # 1. 基本校验
        if not getattr(self, "is_initialized", False):
            raise RuntimeError(
                "Adjudicator must be initialized with _process_history() "
                "and is_initialized must be set to True before generating noise."
            )

        if not self.user_preferences or not self.user_preferences.get("preferences"):
            raise ValueError(
                "User preferences not found. Please run _process_history() first."
            )

        if not init_data:
            raise ValueError("init_data cannot be empty when generating position bias noise.")

        if self.user_id is None:
            raise ValueError("self.user_id is not set. Cannot generate user-specific noise.")

        # 2. 获取对应的 prompt 模板
        if "position_bias_noise_prompt" not in self.prompts:
            raise ValueError("position_bias_noise_prompt not found in prompts config")

        prompt_template = self.prompts["position_bias_noise_prompt"]

        # 兼容 PromptTemplate / dict / 纯字符串 三种情况
        if hasattr(prompt_template, "template"):
            prompt_template = prompt_template.template
        elif isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("content", "")

        # 3. 用户偏好摘要 & 数据集信息
        user_preferences_summary = self._format_preferences_for_prompt()

        dataset_item_counts = {
            "dbbook2014": 2680,
            "book-crossing": 8853,
            "ml1m": 3260,
            "gowalla": 107092,
            "yelp2018": 20033,
            "amazon-book": 96978,
            "lastfm": 11946,
        }

        if isinstance(self.dataset, str):
            dataset_name = self.dataset
        else:
            dataset_name = "dbbook2014"

        total_items = dataset_item_counts.get(dataset_name, 2680)
        max_item_id = total_items - 1

        # 4. 准备候选物品
        #    这里沿用你已有的逻辑：从“未交互”物品中抽样一批 candidate_items
        candidate_items = self._sample_candidate_items(self.user_id, num_candidates=50)

        self.observation(
            f"Prepared {len(candidate_items)} candidate items for POSITION BIAS noise "
            f"for user {self.user_id}"
        )

        # 带 position 的描述表，用于 prompt 展示
        positioned_candidate_items = self._format_positioned_candidate_items(
            candidate_items=candidate_items,
            dataset_name=dataset_name,
        )

        # 纯 ID 列表，用于硬约束
        candidate_item_ids_str = ", ".join(str(item_id) for item_id in candidate_items)

        # 5. 组装最终 prompt
        final_prompt = (
            prompt_template.replace("{interaction_history}", init_data)
            .replace("{user_preferences_summary}", user_preferences_summary)
            .replace("{noise_ratio}", str(noise_ratio))
            .replace("{user_id}", str(self.user_id))
            .replace("{total_items}", str(total_items))
            .replace("{max_item_id}", str(max_item_id))
            .replace("{candidate_item_ids}", candidate_item_ids_str)
            .replace("{positioned_candidate_items}", positioned_candidate_items)
        )

        # 6. 调用 LLM
        self.observation(
            f"Calling LLM to generate POSITION BIAS noise interactions "
            f"for user {self.user_id} with noise_ratio={noise_ratio}..."
        )
        self.observation(
            f"Candidate items for POSITION BIAS selection: {len(candidate_items)} items"
        )

        response = self.llm(final_prompt)

        # 7. 解析 JSON，兼容 ```json ... ``` 包裹
        try:
            clean_response = response.strip()
            if clean_response.startswith("```"):
                parts = clean_response.split("```")
                if len(parts) >= 2:
                    clean_response = parts[1]
                    if clean_response.startswith("json"):
                        clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            noise_data = json.loads(clean_response)

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse position bias noise LLM response as JSON: {e}")
            self.observation(f"Raw response: {response}")
            raise

        # 8. 校验字段 & 过滤非法 item
        if not isinstance(noise_data, dict):
            raise ValueError(
                f"Expected top-level JSON object for position bias noise, got {type(noise_data)}"
            )

        if "noise_interactions" not in noise_data:
            raise ValueError(
                "LLM position bias noise response missing 'noise_interactions' field. "
                f"Got keys: {list(noise_data.keys())}"
            )

        interactions = noise_data.get("noise_interactions", [])

        candidate_set = set(candidate_items)
        invalid_items = []
        noise_item_ids: List[int] = []

        for inter in interactions:
            item_id = inter.get("item_id")
            if item_id is None:
                continue

            if item_id not in candidate_set:
                invalid_items.append(item_id)

            noise_item_ids.append(item_id)

        if invalid_items:
            self.observation(
                f"Warning: Generated {len(invalid_items)} item_ids not in candidate list: "
                f"{invalid_items}"
            )

        self.observation(
            f"Generated {len(interactions)} position bias noise interactions "
            f"for user {self.user_id}."
        )

        # 9. 返回结果（主流水线用 noise_items 即可）
        result: Dict[str, Any] = {
            "user_id": self.user_id,
            "noise_type": "position_bias",
            "noise_items": noise_item_ids,
        }

        # 把 LLM 总结的模式顺手带出来，方便分析
        if isinstance(noise_data.get("bias_pattern"), str):
            result["bias_pattern"] = noise_data["bias_pattern"]

        return result

    def generate_preference_boundary_noise(
            self,
            init_data: str,
            noise_ratio: float,
    ) -> Dict[str, Any]:
        """
        Generate synthetic PREFERENCE-BOUNDARY noise interactions for the current user.

        这类噪声：
        - 与用户偏好高度相关（表面上很像“正样本”）
        - 但并不符合用户最终意图 / 长期满意度（真正意义上是“负样本”）
        - 属于决策边界附近的 high-information hard negatives
        """
        # 1. 基本状态校验
        if not getattr(self, "is_initialized", False):
            raise RuntimeError(
                "Adjudicator must be initialized with _process_history() "
                "and is_initialized must be set to True before generating noise."
            )

        if not self.user_preferences or not self.user_preferences.get("preferences"):
            raise ValueError(
                "User preferences not found. Please run _process_history() first."
            )

        if not init_data:
            raise ValueError("init_data cannot be empty when generating boundary noise.")

        if self.user_id is None:
            raise ValueError("self.user_id is not set. Cannot generate user-specific noise.")

        # 2. 检查对应的 prompt 是否存在
        if "preference_boundary_noise_prompt" not in self.prompts:
            raise ValueError("preference_boundary_noise_prompt not found in prompts config")

        prompt_template = self.prompts["preference_boundary_noise_prompt"]

        # 兼容 PromptTemplate / dict / 纯字符串
        if hasattr(prompt_template, "template"):
            prompt_template = prompt_template.template
        elif isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("content", "")

        # 3. 准备提示词所需信息
        user_preferences_summary = self._format_preferences_for_prompt()
        adversarial_memory_text = self._format_adversarial_memory_for_prompt()

        dataset_item_counts = {
            "dbbook2014": 2680,
            "book-crossing": 8853,
            "ml1m": 3260,
            "gowalla": 107092,
            "yelp2018": 20033,
            "amazon-book": 96978,
            "lastfm": 11946,
        }

        if isinstance(self.dataset, str):
            dataset_name = self.dataset
        else:
            dataset_name = "dbbook2014"

        total_items = dataset_item_counts.get(dataset_name, 2680)
        max_item_id = total_items - 1

        # 4. 采样候选物品（未交互）
        candidate_items = self._sample_candidate_items(self.user_id, num_candidates=50)

        # 获取候选物品的详细信息（ID + 名称）
        candidate_items_info = self._format_candidate_items_with_info(candidate_items, dataset_name)

        # 创建候选物品 ID 的列表字符串，用于约束 LLM
        candidate_item_ids_str = ", ".join(str(item_id) for item_id in candidate_items)

        self.observation("Candidate items prepared with descriptions for PREFERENCE-BOUNDARY noise")

        # 5. 组装最终 Prompt（用 replace 避免与 JSON 花括号冲突）
        final_prompt = (
            prompt_template.replace("{interaction_history}", init_data)
            .replace("{user_preferences_summary}", user_preferences_summary)
            .replace("{adversarial_memory}", adversarial_memory_text)
            .replace("{noise_ratio}", str(noise_ratio))
            .replace("{user_id}", str(self.user_id))
            .replace("{total_items}", str(total_items))
            .replace("{max_item_id}", str(max_item_id))
            .replace("{candidate_items}", candidate_items_info)
            .replace("{candidate_item_ids}", candidate_item_ids_str)
        )

        # 6. 调用 LLM
        self.observation(
            f"Calling LLM to generate PREFERENCE-BOUNDARY noise interactions "
            f"for user {self.user_id} with noise_ratio={noise_ratio}..."
        )
        self.observation(
            f"Candidate items for PREFERENCE-BOUNDARY selection: {len(candidate_items)} items"
        )
        response = self.llm(final_prompt)

        # 7. 清洗 / 解析 LLM 返回
        try:
            clean_response = response.strip()
            if clean_response.startswith("```"):
                parts = clean_response.split("```")
                if len(parts) >= 2:
                    clean_response = parts[1]
                    if clean_response.startswith("json"):
                        clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            noise_data = json.loads(clean_response)

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse preference-boundary noise LLM response as JSON: {e}")
            self.observation(f"Raw response: {response[:500]}")
            raise

        # 8. 基本结构校验
        if not isinstance(noise_data, dict):
            raise ValueError(
                f"Expected top-level JSON object for preference-boundary noise, got {type(noise_data)}"
            )

        if "noise_interactions" not in noise_data:
            raise ValueError(
                "LLM preference-boundary noise response missing 'noise_interactions' field. "
                f"Got keys: {list(noise_data.keys())}"
            )

        interactions = noise_data.get("noise_interactions", [])

        # 9. 验证所有 item_id 都在候选物品中
        candidate_set = set(candidate_items)
        invalid_items: List[int] = []
        noise_item_ids: List[int] = []

        for inter in interactions:
            if not isinstance(inter, dict):
                continue

            item_id = inter.get("item_id")
            if item_id is None:
                continue

            # 这里容忍 LLM 把数字当字符串返回
            try:
                item_id_int = int(item_id)
            except (TypeError, ValueError):
                self.observation(f"Skipping invalid item_id (not int): {item_id}")
                continue

            if item_id_int not in candidate_set:
                invalid_items.append(item_id_int)

            noise_item_ids.append(item_id_int)

        if invalid_items:
            self.observation(
                f"Warning: Generated {len(invalid_items)} item_ids not in candidate list: {invalid_items}"
            )

        self.observation(
            f"Generated {len(noise_item_ids)} preference-boundary noise interactions "
            f"for user {self.user_id}."
        )

        adversarial_notes = noise_data.get("adversarial_notes")
        if isinstance(adversarial_notes, str) and adversarial_notes.strip():
            self.add_adversarial_memory_entry(
                summary=f"LLM-generated adversarial notes for user {self.user_id}: {adversarial_notes.strip()}",
                details={"noise_type": "preference_boundary"},
            )

        # 10. 组装返回格式（主流水线用 noise_items 即可）
        result: Dict[str, Any] = {
            "user_id": self.user_id,
            "noise_type": "preference_boundary",
            "noise_items": noise_item_ids,
        }

        return result

    def reflect_on_adversarial_feedback(
            self,
            init_data: str,
            discriminator_feedback: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        给 Agent 看判别器的对抗评估结果，让 LLM 进行反思，然后更新对抗记忆。

        参数:
            init_data: 用户交互历史文本（和加噪时用的是同一类内容）
            discriminator_feedback: 来自 adversarial_eval_and_train 的 metrics 字典

        返回:
            reflection_result: 解析后的反思 JSON，包含:
                - reflection_summary
                - hard_patterns
                - easy_patterns
                - updated_noise_generation_strategies
                - memory_entry
        """
        # 基本检查
        if not getattr(self, "is_initialized", False):
            raise RuntimeError(
                "Adjudicator must be initialized before reflection."
            )

        if "adversarial_reflection_prompt" not in self.prompts:
            raise ValueError("adversarial_reflection_prompt not found in prompts config")

        prompt_template = self.prompts["adversarial_reflection_prompt"]

        # 兼容 PromptTemplate / dict / str
        if hasattr(prompt_template, "template"):
            prompt_template = prompt_template.template
        elif isinstance(prompt_template, dict):
            prompt_template = prompt_template.get("content", "")

        if not init_data:
            raise ValueError("init_data cannot be empty when reflecting on adversarial feedback.")

        # 准备 Prompt 上下文
        user_preferences_summary = self._format_preferences_for_prompt()
        adversarial_memory_text = self._format_adversarial_memory_for_prompt()
        discriminator_feedback_text = self._format_discriminator_feedback_for_prompt(
            discriminator_feedback
        )

        # 组装最终 Prompt
        final_prompt = (
            prompt_template.replace("{interaction_history}", init_data)
            .replace("{user_preferences_summary}", user_preferences_summary)
            .replace("{adversarial_memory}", adversarial_memory_text)
            .replace("{discriminator_feedback}", discriminator_feedback_text)
        )

        # 调 LLM
        self.observation(
            "Calling LLM for adversarial reflection based on discriminator feedback..."
        )
        response = self.llm(final_prompt)

        # 解析 JSON
        try:
            clean_response = response.strip()
            if clean_response.startswith("```"):
                parts = clean_response.split("```")
                if len(parts) >= 2:
                    clean_response = parts[1]
                    if clean_response.startswith("json"):
                        clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            reflection_data = json.loads(clean_response)

        except json.JSONDecodeError as e:
            self.observation(f"Failed to parse adversarial reflection LLM response as JSON: {e}")
            self.observation(f"Raw response (first 500 chars): {response[:500]}")
            raise

        if not isinstance(reflection_data, dict):
            raise ValueError(
                f"Expected top-level JSON object for adversarial reflection, got {type(reflection_data)}"
            )

        # 取出 memory_entry（优先）或 reflection_summary 写进长期记忆
        memory_entry = reflection_data.get("memory_entry")
        if not isinstance(memory_entry, str) or len(memory_entry.strip()) == 0:
            memory_entry = reflection_data.get("reflection_summary", "")

        if isinstance(memory_entry, str) and memory_entry.strip():
            self.add_adversarial_memory_entry(
                summary=memory_entry.strip(),
                details={
                    "hard_patterns": reflection_data.get("hard_patterns", []),
                    "easy_patterns": reflection_data.get("easy_patterns", []),
                    "updated_noise_generation_strategies": reflection_data.get(
                        "updated_noise_generation_strategies", []
                    ),
                    "discriminator_stats": {
                        "accuracy": discriminator_feedback.get("accuracy", None),
                        "tp": discriminator_feedback.get("tp", None),
                        "tn": discriminator_feedback.get("tn", None),
                        "fp": discriminator_feedback.get("fp", None),
                        "fn": discriminator_feedback.get("fn", None),
                    },
                },
            )
            self.observation("Adversarial memory updated from reflection.")

        # 也可以顺便挂在实例上，方便调试查看
        self.last_adversarial_reflection = reflection_data

        return reflection_data

def load_existing_preferences(output_path: str) -> tuple[Dict[int, Dict[str, Any]], set[int]]:
    """
    从已有的 user_preferences.json 中加载已处理的用户

    Args:
        output_path: user_preferences.json 文件路径

    Returns:
        (user_profiles, processed_user_ids)
        - user_profiles: 已有的用户偏好字典
        - processed_user_ids: 已处理的用户 ID 集合
    """
    if not os.path.exists(output_path):
        print(f"ℹ No existing preferences file found at {output_path}")
        return {}, set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            user_profiles = json.load(f)

        processed_user_ids = set(int(uid) for uid in user_profiles.keys())
        print(f"✓ Loaded {len(user_profiles)} existing user preferences")
        print(
            f"  Already processed user IDs: {sorted(processed_user_ids)[:10]}{'...' if len(processed_user_ids) > 10 else ''}")

        return user_profiles, processed_user_ids

    except json.JSONDecodeError as e:
        print(f"✗ Error: Corrupted JSON file - {e}")
        print(f"  Please check {output_path}")
        sys.exit(1)


def save_single_user_preference(output_path: str, user_id: int, preference_data: Dict[str, Any]) -> None:
    """
    将单个用户的偏好追加到 JSON 文件

    Args:
        output_path: user_preferences.json 文件路径
        user_id: 用户 ID
        preference_data: 用户偏好数据
    """
    # 读取现有数据
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                user_profiles = json.load(f)
            except json.JSONDecodeError:
                user_profiles = {}
    else:
        user_profiles = {}

    # 添加新用户
    user_profiles[str(user_id)] = preference_data

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=2)


def extract_user_preferences_only(
        prompt_config_path: str,
        llm_config_path: str,
        train_file: str,
        item_list_file: str,
        dataset_path: str,
        max_history_len: int = 50,
        max_users: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    只提取用户偏好，不生成噪声。
    支持断点续传：每个用户处理完后立即保存，重新运行时从上次中断的位置继续。
    结果保存到 dataset_path/user_preferences.json

    Args:
        prompt_config_path: Prompt 配置文件路径
        llm_config_path: LLM 配置文件路径
        train_file: 训练文件路径 (train.txt)
        item_list_file: 物品列表文件路径 (item_list.txt)
        dataset_path: 数据集目录路径（结果将保存在此目录下）
        max_history_len: 每个用户最多使用多少条交互记录
        max_users: 最多处理多少个用户（None 表示全部）

    Returns:
        user_profiles: {user_id: preferences_data}
    """
    print("=" * 80)
    print("EXTRACTING USER PREFERENCES ONLY (WITH CHECKPOINT SUPPORT)")
    print("=" * 80)
    print(f"Prompt Config  : {prompt_config_path}")
    print(f"LLM Config     : {llm_config_path}")
    print(f"Train File     : {train_file}")
    print(f"Item List      : {item_list_file}")
    print(f"Dataset Path   : {dataset_path}")
    print("-" * 80 + "\n")

    try:
        # ======================== 第一步：检查是否有断点 ========================
        print("Step 1: Checking for existing preferences and resuming from checkpoint...")
        os.makedirs(dataset_path, exist_ok=True)
        output_path = os.path.join(dataset_path, 'user_preferences.json')

        user_profiles, processed_user_ids = load_existing_preferences(output_path)
        print()

        # ======================== 第二步：读取 LLM 配置 ========================
        print("Step 2: Loading LLM configuration...")
        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_config = json.load(f)
        print(f"✓ LLM config loaded: {llm_config.get('model_name', 'N/A')}\n")

        # ======================== 第三步：初始化 Adjudicator ========================
        print("Step 3: Creating Adjudicator instance...")
        adjudicator = Adjudicator(
            prompt_config=str(prompt_config_path),
            dataset="dbbook2014",
            llm_config=llm_config,
            web_demo=False,
        )
        print("✓ Adjudicator created successfully\n")

        # ======================== 第四步：准备数据 ========================
        print("Step 4: Preparing init_data from raw txt...")
        users_data = Adjudicator.prepare_init_data(
            train_file=train_file,
            item_list_file=item_list_file,
            max_history_len=max_history_len,
        )
        all_user_ids = sorted(users_data.keys())
        if max_users is not None:
            all_user_ids = all_user_ids[:max_users]

        print(f"✓ Parsed {len(all_user_ids)} users from train.txt\n")

        # ======================== 第五步：过滤已处理的用户 ========================
        remaining_user_ids = [uid for uid in all_user_ids if uid not in processed_user_ids]
        print(f"Step 5: Checkpoint status")
        print(f"  - Total users in dataset: {len(all_user_ids)}")
        print(f"  - Already processed: {len(processed_user_ids)}")
        print(f"  - Remaining to process: {len(remaining_user_ids)}")

        if len(remaining_user_ids) == 0:
            print("\n✓ All users have been processed!")
            print("=" * 80 + "\n")
            return user_profiles

        print(f"  - Next batch will start from user_id: {remaining_user_ids[0]}\n")

        # ======================== 第六步：主循环 - 处理剩余用户 ========================
        print("Step 6: Processing remaining users")
        print("-" * 80)

        successful_count = len(processed_user_ids)
        failed_count = 0

        for batch_idx, user_id in enumerate(remaining_user_ids, 1):
            user_blob = users_data[user_id]
            init_data = user_blob["init_data"]
            init_reviews_count = user_blob["init_reviews_count"]

            print(f"\n[User {successful_count + batch_idx}/{len(all_user_ids)}] user_id = {user_id}")
            print(f"  - init_reviews_count = {init_reviews_count}")

            try:
                # 5.1 设置当前 user_id
                adjudicator.user_id = user_id

                # 5.2 提取用户画像
                adjudicator._process_history(
                    init_data=init_data,
                    init_reviews_count=init_reviews_count,
                )
                adjudicator.is_initialized = True

                # 5.3 保存当前用户的画像到内存
                preference_data = adjudicator.user_preferences
                user_profiles[user_id] = preference_data

                # ✅ 5.4 立即保存到文件（断点续传关键）
                save_single_user_preference(output_path, user_id, preference_data)
                successful_count += 1

                print(f"  ✓ Extracted and saved preferences for user {user_id}")
                print(f"    Preferences: {len(preference_data.get('preferences', []))} | "
                      f"Dislikes: {len(preference_data.get('dislikes', []))}")

            except Exception as e:
                failed_count += 1
                print(f"  ✗ Error processing user {user_id}: {e}")
                print(f"     Skipping this user and continuing...")
                continue

        # ======================== 第七步：最终统计 ========================
        print("\n" + "=" * 80)
        print("✓ User preference extraction completed!")
        print("=" * 80)
        print(f"Summary:")
        print(f"  - Total users in dataset: {len(all_user_ids)}")
        print(f"  - Successfully processed: {successful_count}")
        print(f"  - Failed to process: {failed_count}")
        print(f"  - Output file: {output_path}")
        print("=" * 80 + "\n")

        return user_profiles

    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Failed to parse JSON file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





def generate_noise_for_all_users(
        prompt_config_path: str,
        llm_config_path: str,
        preferences_file: str,
        dataset_path: str,
        misclick_ratio: float = 0.2,
        curiosity_ratio: float = 0.2,
        caption_ratio: float = 0.2,
        popularity_ratio: float = 0.2,
        position_ratio: float = 0.2,
        save_interval: int = 20,
        max_users: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    为所有用户生成噪声数据（按指定间隔定期保存）。
    支持断点续传：每隔 save_interval 个用户保存一次，重新运行时从上次中断的位置继续。

    结果保存到：
    - dataset_path/noise_data_all_users.txt (格式: user_id item_id1 item_id2 ...)

    噪声类型包括：
    - Misclick
    - Curiosity
    - Caption Bias
    - Popularity Bias
    - Position Bias
    """
    print("=" * 80)
    print("GENERATING NOISE FOR ALL USERS (WITH PERIODIC CHECKPOINT)")
    print("=" * 80)
    print(f"Prompt Config        : {prompt_config_path}")
    print(f"LLM Config           : {llm_config_path}")
    print(f"Preferences File     : {preferences_file}")
    print(f"Dataset Path         : {dataset_path}")
    print(f"Misclick Ratio       : {misclick_ratio}")
    print(f"Curiosity Ratio      : {curiosity_ratio}")
    print(f"Caption Ratio        : {caption_ratio}")
    print(f"Popularity Ratio     : {popularity_ratio}")
    print(f"Position Ratio       : {position_ratio}")
    print(f"Save Interval        : Every {save_interval} users")
    print(f"Max Users            : {max_users if max_users else 'All'}")
    print("-" * 80 + "\n")

    try:
        # ======================== 第一步：加载用户偏好文件 ========================
        print("Step 1: Loading user preferences from file...")
        if not os.path.exists(preferences_file):
            raise FileNotFoundError(f"Preferences file not found: {preferences_file}")

        with open(preferences_file, "r", encoding="utf-8") as f:
            user_profiles = json.load(f)

        all_user_ids = sorted([int(uid) for uid in user_profiles.keys()])
        if max_users is not None:
            all_user_ids = all_user_ids[:max_users]

        print(f"✓ Loaded preferences for {len(all_user_ids)} users\n")

        # ======================== 第二步：检查已生成的噪声数据 ========================
        print("Step 2: Checking for existing noise data and resuming from checkpoint...")
        os.makedirs(dataset_path, exist_ok=True)
        noise_output_path = os.path.join(dataset_path, 'noise_data_all_users.txt')

        # 加载已有的噪声数据
        processed_user_ids = set()
        existing_noise_records = []

        if os.path.exists(noise_output_path):
            try:
                with open(noise_output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) >= 1:
                            try:
                                user_id = int(parts[0])
                                noise_items = [int(x) for x in parts[1:]] if len(parts) > 1 else []

                                existing_noise_records.append({
                                    'user_id': user_id,
                                    'noise_items': noise_items
                                })
                                processed_user_ids.add(user_id)

                            except (ValueError, IndexError):
                                print(f"Warning: Failed to parse line: {line}")
                                continue
            except Exception as e:
                print(f"Warning: Could not read existing noise data: {e}")
                existing_noise_records = []
                processed_user_ids = set()

        print(f"✓ Found {len(existing_noise_records)} existing noise records")
        print(f"✓ Already processed {len(processed_user_ids)} users\n")

        # 计算剩余需要处理的用户
        remaining_user_ids = [uid for uid in all_user_ids if uid not in processed_user_ids]

        print(f"Step 3: Checkpoint status")
        print(f"  - Total users in dataset: {len(all_user_ids)}")
        print(f"  - Already processed: {len(processed_user_ids)}")
        print(f"  - Remaining to process: {len(remaining_user_ids)}")

        if len(remaining_user_ids) == 0:
            print("\n✓ All users have been processed!")
            print("=" * 80 + "\n")
            return existing_noise_records

        if remaining_user_ids:
            print(f"  - Next batch will start from user_id: {remaining_user_ids[0]}\n")
        else:
            print()

        # ======================== 第三步：读取 LLM 配置 ========================
        print("Step 4: Loading LLM configuration...")
        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_config = json.load(f)
        print(f"✓ LLM config loaded: {llm_config.get('model_name', 'N/A')}\n")

        # ======================== 第四步：初始化 Adjudicator ========================
        print("Step 5: Creating Adjudicator instance...")
        adjudicator = Adjudicator(
            prompt_config=str(prompt_config_path),
            dataset="dbbook2014",
            llm_config=llm_config,
            web_demo=False,
        )
        print("✓ Adjudicator created successfully\n")

        # ======================== 第五步：主循环 - 为每个用户生成噪声 ========================
        print("Step 6: Generating noise for all remaining users")
        print("-" * 80)

        batch_buffer = []  # 临时缓冲区，用于存储本批数据
        successful_count = len(processed_user_ids)
        failed_count = 0

        for idx, user_id in enumerate(remaining_user_ids, 1):
            print(f"\n[User {successful_count + idx}/{len(all_user_ids)}] user_id = {user_id}")

            try:
                # 5.1 从已保存的偏好中恢复用户状态
                adjudicator.user_id = user_id
                adjudicator.user_preferences = user_profiles[str(user_id)]
                adjudicator.is_initialized = True

                # 5.2 构造虚拟的 init_data（这里只需要一个简短的上下文）
                init_data = f"User ID: {user_id}\nPreferences loaded from checkpoint."

                # ---- Misclick 噪声 ----
                print(f" - Generating misclick noise (ratio={misclick_ratio})...")
                misclick_row = adjudicator.generate_misclick_noise(
                    init_data=init_data,
                    noise_ratio=misclick_ratio,
                )
                misclick_items = misclick_row.get('noise_items', [])
                print(f"   ✓ Generated {len(misclick_items)} misclick items")

                # ---- Curiosity 噪声 ----
                print(f" - Generating curiosity noise (ratio={curiosity_ratio})...")
                curiosity_row = adjudicator.generate_curiosity_noise(
                    init_data=init_data,
                    noise_ratio=curiosity_ratio,
                )
                curiosity_items = curiosity_row.get('noise_items', [])
                print(f"   ✓ Generated {len(curiosity_items)} curiosity items")

                # ---- Caption Bias 噪声 ----
                print(f" - Generating caption bias noise (ratio={caption_ratio})...")
                caption_row = adjudicator.generate_caption_bias_noise(
                    init_data=init_data,
                    noise_ratio=caption_ratio,
                )
                caption_items = caption_row.get('noise_items', [])
                print(f"   ✓ Generated {len(caption_items)} caption-bias items")

                # ---- Popularity Bias 噪声 ----
                print(f" - Generating popularity bias noise (ratio={popularity_ratio})...")
                popularity_row = adjudicator.generate_popularity_bias_noise(
                    init_data=init_data,
                    noise_ratio=popularity_ratio,
                )
                popularity_items = popularity_row.get('noise_items', [])
                print(f"   ✓ Generated {len(popularity_items)} popularity-bias items")

                # ---- Position Bias 噪声 ----
                print(f" - Generating position bias noise (ratio={position_ratio})...")
                position_row = adjudicator.generate_position_bias_noise(
                    init_data=init_data,
                    noise_ratio=position_ratio,
                )
                position_items = position_row.get('noise_items', [])
                print(f"   ✓ Generated {len(position_items)} position-bias items")

                # 5.5 合并所有噪声的 items（去重，保持顺序）
                all_noise_items = (
                    misclick_items
                    + curiosity_items
                    + caption_items
                    + popularity_items
                    + position_items
                )
                # 去重但保持顺序
                all_noise_items = list(dict.fromkeys(all_noise_items))

                noise_record = {
                    'user_id': user_id,
                    'noise_items': all_noise_items
                }

                # 添加到缓冲区
                batch_buffer.append(noise_record)
                successful_count += 1

                print(f" ✓ Successfully generated noise for user {user_id}")
                print(
                    f"   Total noise items: {len(all_noise_items)} "
                    f"(misclick: {len(misclick_items)}, "
                    f"curiosity: {len(curiosity_items)}, "
                    f"caption: {len(caption_items)}, "
                    f"popularity: {len(popularity_items)}, "
                    f"position: {len(position_items)})"
                )

                # ✅ 定期保存
                if len(batch_buffer) >= save_interval:
                    _save_noise_batch_to_file(noise_output_path, batch_buffer)
                    print(f"\n📝 Checkpoint: Saved {len(batch_buffer)} records to file "
                          f"(total processed: {successful_count})")
                    batch_buffer = []  # 清空缓冲区

            except Exception as e:
                failed_count += 1
                print(f" ✗ Error processing user {user_id}: {e}")
                continue

        # ======================== 第六步：保存剩余数据 ========================
        if batch_buffer:
            _save_noise_batch_to_file(noise_output_path, batch_buffer)
            print(f"\n📝 Final checkpoint: Saved {len(batch_buffer)} records to file")

        print("\n" + "=" * 80)
        print("✓ Noise generation completed!")
        print("=" * 80)
        print(f"Summary:")
        print(f"  - Total users in dataset: {len(all_user_ids)}")
        print(f"  - Previously processed: {len(processed_user_ids)}")
        print(f"  - Newly processed this run: {successful_count - len(processed_user_ids)}")
        print(f"  - Total successfully processed: {successful_count}")
        print(f"  - Failed to process: {failed_count}")
        print(f"  - Output file: {noise_output_path}")
        print(f"  - Output format: user_id item_id1 item_id2 ... "
              f"(misclick + curiosity + caption + popularity + position mixed)")
        print(f"  - Save interval: Every {save_interval} users")
        print("=" * 80 + "\n")

        # 返回已存在的记录即可（真正的数据都在文件里）
        return existing_noise_records

    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Failed to parse JSON file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



def _save_noise_batch_to_file(noise_output_path: str, batch_records: List[Dict[str, Any]]) -> None:
    """
    将一批噪声记录追加到文件

    格式：user_id item_id1 item_id2 item_id3 ...

    Args:
        noise_output_path: 噪声数据文件路径
        batch_records: 一批噪声记录列表
    """
    try:
        with open(noise_output_path, 'a', encoding='utf-8') as f:
            for record in batch_records:
                uid = record["user_id"]
                items = record.get("noise_items", [])
                items_str = " ".join(str(i) for i in items)

                line = f"{uid}"
                if items_str:
                    line += " " + items_str

                f.write(line + "\n")

    except Exception as e:
        print(f"Warning: Failed to save noise batch to file: {e}")


def _save_noise_record_to_file(noise_output_path: str, noise_record: Dict[str, Any]) -> None:
    """
    将单条噪声记录追加到文件

    新格式：user_id item_id1 item_id2 item_id3 ...
    （不再分离 noise_type，直接混合所有 item_id）

    Args:
        noise_output_path: 噪声数据文件路径
        noise_record: 噪声记录字典
    """
    try:
        uid = noise_record["user_id"]
        items = noise_record.get("noise_items", [])

        # 只保存 user_id + item_ids，不保存 noise_type
        items_str = " ".join(str(i) for i in items)

        line = f"{uid}"
        if items_str:
            line += " " + items_str

        # 以追加模式写入
        with open(noise_output_path, 'a', encoding='utf-8') as f:
            f.write(line + "\n")

    except Exception as e:
        print(f"Warning: Failed to save noise record to file: {e}")


def generate_noise_from_sampled_users(
        prompt_config_path: str,
        llm_config_path: str,
        preferences_file: str,
        dataset_path: str,
        num_sampled_users: int = 1000,
        misclick_ratio: float = 0.2,
        curiosity_ratio: float = 0.2,
        caption_ratio: float = 0.2,
        popularity_ratio: float = 0.2,
        position_ratio: float = 0.2,
        random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    从已有的用户偏好文件中随机采样指定数量的用户，然后生成噪声数据。
    结果保存到 dataset_path/noise_data_sampled.txt

    噪声类型包括：
    - Misclick
    - Curiosity
    - Caption Bias
    - Popularity Bias
    - Position Bias
    """
    print("=" * 80)
    print("GENERATING NOISE FROM SAMPLED USERS (NO CHECKPOINT)")
    print("=" * 80)
    print(f"Prompt Config        : {prompt_config_path}")
    print(f"LLM Config           : {llm_config_path}")
    print(f"Preferences File     : {preferences_file}")
    print(f"Dataset Path         : {dataset_path}")
    print(f"Sampled Users        : {num_sampled_users}")
    print(f"Misclick Ratio       : {misclick_ratio}")
    print(f"Curiosity Ratio      : {curiosity_ratio}")
    print(f"Caption Ratio        : {caption_ratio}")
    print(f"Popularity Ratio     : {popularity_ratio}")
    print(f"Position Ratio       : {position_ratio}")
    print(f"Random Seed          : {random_seed}")
    print("-" * 80 + "\n")

    try:
        # ======================== 第一步：加载用户偏好文件 ========================
        print("Step 1: Loading user preferences from file...")
        if not os.path.exists(preferences_file):
            raise FileNotFoundError(f"Preferences file not found: {preferences_file}")

        with open(preferences_file, "r", encoding="utf-8") as f:
            user_profiles = json.load(f)

        all_user_ids = sorted([int(uid) for uid in user_profiles.keys()])
        print(f"✓ Loaded preferences for {len(all_user_ids)} users\n")

        # ======================== 第二步：随机采样用户 ========================
        print("Step 2: Randomly sampling users...")
        if random_seed is not None:
            np.random.seed(random_seed)

        actual_sample_size = min(num_sampled_users, len(all_user_ids))
        if actual_sample_size < num_sampled_users:
            print(f"Warning: Requested {num_sampled_users} users, but only {len(all_user_ids)} available.")
            print(f"Using all {actual_sample_size} users instead.\n")
        else:
            print(f"Sampling {actual_sample_size} out of {len(all_user_ids)} users.\n")

        sampled_user_ids = sorted(
            np.random.choice(all_user_ids, size=actual_sample_size, replace=False).tolist()
        )
        print(f"✓ Sampled {len(sampled_user_ids)} users")
        print(f"  Sample user IDs: {sampled_user_ids[:10]}{'...' if len(sampled_user_ids) > 10 else ''}\n")

        # ======================== 第三步：读取 LLM 配置 ========================
        print("Step 3: Loading LLM configuration...")
        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_config = json.load(f)
        print(f"✓ LLM config loaded: {llm_config.get('model_name', 'N/A')}\n")

        # ======================== 第四步：初始化 Adjudicator ========================
        print("Step 4: Creating Adjudicator instance...")
        adjudicator = Adjudicator(
            prompt_config=str(prompt_config_path),
            dataset="dbbook2014",
            llm_config=llm_config,
            web_demo=False,
        )
        print("✓ Adjudicator created successfully\n")

        # ======================== 第五步：主循环 - 为每个采样用户生成噪声 ========================
        print("Step 5: Generating noise for sampled users")
        print("-" * 80)

        noise_records: List[Dict[str, Any]] = []
        successful_count = 0
        failed_count = 0

        for idx, user_id in enumerate(sampled_user_ids, 1):
            print(f"\n[User {idx}/{len(sampled_user_ids)}] user_id = {user_id}")

            try:
                # 5.1 从已保存的偏好中恢复用户状态
                adjudicator.user_id = user_id
                adjudicator.user_preferences = user_profiles[str(user_id)]
                adjudicator.is_initialized = True

                # 5.2 构造虚拟的 init_data
                init_data = f"User ID: {user_id}\nPreferences loaded from checkpoint."

                # ---- Misclick 噪声 ----
                print(f" - Generating misclick noise (ratio={misclick_ratio})...")
                misclick_row = adjudicator.generate_misclick_noise(
                    init_data=init_data,
                    noise_ratio=misclick_ratio,
                )
                misclick_items = misclick_row.get('noise_items', [])
                print(f"   ✓ Generated {len(misclick_items)} misclick items")

                # ---- Curiosity 噪声 ----
                print(f" - Generating curiosity noise (ratio={curiosity_ratio})...")
                curiosity_row = adjudicator.generate_curiosity_noise(
                    init_data=init_data,
                    noise_ratio=curiosity_ratio,
                )
                curiosity_items = curiosity_row.get('noise_items', [])
                print(f"   ✓ Generated {len(curiosity_items)} curiosity items")

                # ---- Caption Bias 噪声 ----
                print(f" - Generating caption bias noise (ratio={caption_ratio})...")
                caption_row = adjudicator.generate_caption_bias_noise(
                    init_data=init_data,
                    noise_ratio=caption_ratio,
                )
                caption_items = caption_row.get('noise_items', [])
                print(f"   ✓ Generated {len(caption_items)} caption-bias items")

                # ---- Popularity Bias 噪声 ----
                print(f" - Generating popularity bias noise (ratio={popularity_ratio})...")
                popularity_row = adjudicator.generate_popularity_bias_noise(
                    init_data=init_data,
                    noise_ratio=popularity_ratio,
                )
                popularity_items = popularity_row.get('noise_items', [])
                print(f"   ✓ Generated {len(popularity_items)} popularity-bias items")

                # ---- Position Bias 噪声 ----
                print(f" - Generating position bias noise (ratio={position_ratio})...")
                position_row = adjudicator.generate_position_bias_noise(
                    init_data=init_data,
                    noise_ratio=position_ratio,
                )
                position_items = position_row.get('noise_items', [])
                print(f"   ✓ Generated {len(position_items)} position-bias items")

                # 5.5 合并所有噪声的 items（去重，保持顺序）
                all_noise_items = (
                    misclick_items
                    + curiosity_items
                    + caption_items
                    + popularity_items
                    + position_items
                )
                all_noise_items = list(dict.fromkeys(all_noise_items))

                noise_record = {
                    'user_id': user_id,
                    'noise_items': all_noise_items
                }
                noise_records.append(noise_record)

                successful_count += 1
                print(f" ✓ Successfully generated noise for user {user_id}")
                print(
                    f"   Total noise items: {len(all_noise_items)} "
                    f"(misclick: {len(misclick_items)}, "
                    f"curiosity: {len(curiosity_items)}, "
                    f"caption: {len(caption_items)}, "
                    f"popularity: {len(popularity_items)}, "
                    f"position: {len(position_items)})"
                )

            except Exception as e:
                failed_count += 1
                print(f" ✗ Error processing user {user_id}: {e}")
                continue

        # ======================== 第六步：保存噪声数据 ========================
        print("\nStep 6: Saving noise data to file...")
        os.makedirs(dataset_path, exist_ok=True)
        noise_output_path = os.path.join(dataset_path, 'noise_data_sampled.txt')

        with open(noise_output_path, "w", encoding="utf-8") as f:
            for record in noise_records:
                uid = record["user_id"]
                items = record.get("noise_items", [])
                items_str = " ".join(str(i) for i in items)

                line = f"{uid}"
                if items_str:
                    line += " " + items_str

                f.write(line + "\n")

        print(f"✓ Noise data saved to: {noise_output_path}")
        print(f"  Format: user_id item_id1 item_id2 ... "
              f"(misclick + curiosity + caption + popularity + position mixed)\n")

        # ======================== 第七步：统计总结 ========================
        print("=" * 80)
        print("✓ Noise generation completed!")
        print("=" * 80)
        print(f"Summary:")
        print(f"  - Total users in dataset: {len(all_user_ids)}")
        print(f"  - Sampled users: {len(sampled_user_ids)}")
        print(f"  - Successfully processed: {successful_count}")
        print(f"  - Failed to process: {failed_count}")
        print(f"  - Output file: {noise_output_path}")
        print(f"  - Output format: user_id item_id1 item_id2 ... "
              f"(misclick + curiosity + caption + popularity + position mixed)")
        print("=" * 80 + "\n")

        return noise_records

    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Failed to parse JSON file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def find_recommender_model(dataset_name, model_name):
    """找到推荐模型的最新 checkpoint"""

    recommender_weights_dir = Path(
        __file__).parent / "recommender" / "weights" / dataset_name / model_name / "best_model"

    print(f"Looking for model in: {recommender_weights_dir}")
    print(f"Directory exists: {recommender_weights_dir.exists()}\n")

    if not recommender_weights_dir.exists():
        print(f"✗ Error: Model directory not found")
        return None

    # ✅ 用 os.listdir() 代替 glob()（更稳健，处理中文路径）
    try:
        files = os.listdir(str(recommender_weights_dir))
        print(f"Found {len(files)} files in directory:")
        for f in files:
            print(f"  - {f}")
        print()
    except Exception as e:
        print(f"✗ Error reading directory: {e}")
        return None

    # 筛选 .pth.tar 文件
    checkpoints = [f for f in files if f.endswith('.pth.tar')]

    print(f"Found {len(checkpoints)} .pth.tar checkpoint files:")
    for cp in checkpoints:
        print(f"  - {cp}")
    print()

    if not checkpoints:
        print(f"✗ Error: No .pth.tar checkpoint files found")
        return None

    # 按修改时间排序，取最新的
    checkpoint_paths = [recommender_weights_dir / f for f in checkpoints]
    latest_checkpoint = max(checkpoint_paths, key=lambda p: os.path.getmtime(str(p)))

    print(f"✓ Using latest checkpoint: {latest_checkpoint.name}")
    print(f"  Full path: {latest_checkpoint}\n")

    return str(latest_checkpoint)


if __name__ == "__main__":
    # ==================== 运行模式配置 ====================
    RUN_MODE = 'all'
    CANDIDATE_SAMPLING_STRATEGY = 'recommender'

    NUM_CANDIDATES = 100
    NOISE_GENERATION_MODE = 'all'
    NUM_SAMPLED_USERS = 100
    RANDOM_SEED = 42
    SAVE_INTERVAL = 20

    MISCLICK_RATIO = 0.05
    CURIOSITY_RATIO = 0.1
    CAPTION_RATIO = 0.1
    POPULARITY_RATIO = 0.2
    POSITION_RATIO = 0.1

    MAX_HISTORY_LEN = 50
    MAX_USERS = None

    # ==================== 路径配置 ====================
    current_dir = Path(__file__).parent  # agents 目录
    project_root = current_dir.parent  # mycode 目录

    api_config_path = project_root / "config" / "api-config.json"
    init_openai_api(read_json(str(api_config_path)))

    config_dir = project_root / "config"
    prompts_dir = config_dir / "prompts"
    agents_config_dir = config_dir / "agents"
    prompt_config_path = str(prompts_dir / "adjudicator.json")
    llm_config_path = str(agents_config_dir / "adjudicator.json")

    dataset_name = "dbbook2014"
    dataset_path = str(project_root / "data" / dataset_name)
    preferences_file = str(project_root / "data" / dataset_name / "user_preferences.json")

    # -------- 推荐模型采样参数 --------
    model_name = 'LightGCN'

    # ✅ 正确的相对路径：从 agents 回到 mycode，再进入 recommenders
    recommender_weights_dir = project_root / "recommenders" / "weights" / dataset_name / model_name / "best_model"

    print(f"DEBUG: Current script location: {current_dir}")
    print(f"DEBUG: Project root: {project_root}")
    print(f"DEBUG: Model directory: {recommender_weights_dir}")
    print(f"DEBUG: Directory exists: {recommender_weights_dir.exists()}\n")

    # 找到最新的 checkpoint 文件
    if not recommender_weights_dir.exists():
        print(f"✗ Error: Model directory not found")
        print(f"  Expected: {recommender_weights_dir}")
        sys.exit(1)

    # ✅ 用 os.listdir() 处理中文路径
    import os

    all_files = os.listdir(str(recommender_weights_dir))
    checkpoints = [f for f in all_files if f.endswith('.pth.tar')]

    print(f"Files in directory: {all_files}")
    print(f".pth.tar files found: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  - {cp}\n")

    if checkpoints:
        # 按修改时间排序，取最新的
        checkpoint_paths = [recommender_weights_dir / f for f in checkpoints]
        latest_checkpoint = max(checkpoint_paths, key=lambda p: p.stat().st_mtime)
        RECOMMENDER_MODEL_PATH = str(latest_checkpoint)
        print(f"✓ Using latest checkpoint: {latest_checkpoint.name}\n")
    else:
        print(f"✗ Error: No checkpoint found in {recommender_weights_dir}")
        print(f"Available files:")
        for f in all_files:
            print(f"  - {f}")
        sys.exit(1)

    # ==================== 初始化输出 ====================
    print("\n" + "=" * 100)
    print("NOISE GENERATION PIPELINE".center(100))
    print("=" * 100)
    print(f"{'Run Mode':<30}: {RUN_MODE}")
    print(f"{'Dataset':<30}: {dataset_name}")
    print(f"{'Preferences File':<30}: {preferences_file}")
    print(f"{'Candidate Sampling Strategy':<30}: {CANDIDATE_SAMPLING_STRATEGY}")

    if CANDIDATE_SAMPLING_STRATEGY == 'recommender':
        print(f"  ├─ Model Type              : {model_name}")
        print(f"  └─ Model Path              : {RECOMMENDER_MODEL_PATH}")
    elif CANDIDATE_SAMPLING_STRATEGY == 'random':
        print(f"  └─ Num Candidates          : {NUM_CANDIDATES}")

    print(f"{'Noise Generation Mode':<30}: {NOISE_GENERATION_MODE}")
    if NOISE_GENERATION_MODE == 'sampled':
        print(f"  └─ Num Sampled Users       : {NUM_SAMPLED_USERS}")
    elif NOISE_GENERATION_MODE == 'all':
        print(f"  └─ Save Interval           : {SAVE_INTERVAL}")

    print(f"{'Noise Ratios':<30}")
    print(f"  ├─ Misclick                : {MISCLICK_RATIO}")
    print(f"  ├─ Curiosity               : {CURIOSITY_RATIO}")
    print(f"  ├─ Caption                 : {CAPTION_RATIO}")
    print(f"  ├─ Popularity              : {POPULARITY_RATIO}")
    print(f"  └─ Position                : {POSITION_RATIO}")

    print(f"{'Max Users':<30}: {MAX_USERS if MAX_USERS else 'All'}")
    print("=" * 100 + "\n")

    # ==================== 参数校验 ====================
    required_files = {
        "preferences_file": preferences_file,
        "prompt_config": prompt_config_path,
        "llm_config": llm_config_path,
    }

    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"  ✗ {name}: {path}")

    if missing_files:
        print("✗ Error: Missing required files:")
        for msg in missing_files:
            print(msg)
        sys.exit(1)

    if CANDIDATE_SAMPLING_STRATEGY not in ['random', 'recommender']:
        print(f"✗ Error: Invalid CANDIDATE_SAMPLING_STRATEGY: {CANDIDATE_SAMPLING_STRATEGY}")
        print("  Please set to 'random' or 'recommender'")
        sys.exit(1)

    if CANDIDATE_SAMPLING_STRATEGY == 'recommender':
        if not os.path.exists(RECOMMENDER_MODEL_PATH):
            print(f"✗ Error: Recommender model not found: {RECOMMENDER_MODEL_PATH}")
            sys.exit(1)
        print(f"✓ Recommender model found: {RECOMMENDER_MODEL_PATH}\n")

    print("✓ All required files found\n")

    # ==================== 构建候选采样参数字典 ====================
    candidate_sampling_kwargs = {}

    if CANDIDATE_SAMPLING_STRATEGY == 'random':
        candidate_sampling_kwargs = {
            'use_recommender': False,
            'num_candidates': NUM_CANDIDATES,
        }
        print(f"▶ Using RANDOM candidate sampling")
        print(f"  └─ Number of candidates per user: {NUM_CANDIDATES}\n")

    elif CANDIDATE_SAMPLING_STRATEGY == 'recommender':
        candidate_sampling_kwargs = {
            'use_recommender': True,
            'recommender_model_type': model_name,
            'recommender_model_path': RECOMMENDER_MODEL_PATH,
        }
        print(f"▶ Using RECOMMENDER MODEL candidate sampling")
        print(f"  ├─ Model Type: {model_name}")
        print(f"  └─ Model Path: {RECOMMENDER_MODEL_PATH}\n")

    # ==================== 生成噪声 ====================
    print("=" * 100)
    print(f"STARTING NOISE GENERATION ({NOISE_GENERATION_MODE.upper()} mode)".center(100))
    print("=" * 100 + "\n")

    try:
        if NOISE_GENERATION_MODE == 'all':
            print(">>> NOISE MODE: Generate for ALL users (with periodic checkpoint)\n")

            noise_records = generate_noise_for_all_users(
                prompt_config_path=prompt_config_path,
                llm_config_path=llm_config_path,
                preferences_file=preferences_file,
                dataset_path=dataset_path,
                misclick_ratio=MISCLICK_RATIO,
                curiosity_ratio=CURIOSITY_RATIO,
                caption_ratio=CAPTION_RATIO,
                popularity_ratio=POPULARITY_RATIO,
                position_ratio=POSITION_RATIO,
                save_interval=SAVE_INTERVAL,
                max_users=MAX_USERS,
            )

        elif NOISE_GENERATION_MODE == 'sampled':
            print(">>> NOISE MODE: Generate for SAMPLED users (no checkpoint)\n")

            noise_records = generate_noise_from_sampled_users(
                prompt_config_path=prompt_config_path,
                llm_config_path=llm_config_path,
                preferences_file=preferences_file,
                dataset_path=dataset_path,
                num_sampled_users=NUM_SAMPLED_USERS,
                misclick_ratio=MISCLICK_RATIO,
                curiosity_ratio=CURIOSITY_RATIO,
                caption_ratio=CAPTION_RATIO,
                popularity_ratio=POPULARITY_RATIO,
                position_ratio=POSITION_RATIO,
                random_seed=RANDOM_SEED,
            )

        else:
            raise ValueError(f"Unknown NOISE_GENERATION_MODE: {NOISE_GENERATION_MODE}")

        # ==================== 完成 ====================
        print("\n" + "=" * 100)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!".center(100))
        print("=" * 100)
        print(f"\nSummary:")
        print(f"  • Users processed: {len(noise_records)}")
        print(f"  • Candidate sampling: {CANDIDATE_SAMPLING_STRATEGY}")
        if CANDIDATE_SAMPLING_STRATEGY == 'recommender':
            print(f"  • Recommender model: {model_name}")
        print(f"  • Output file: {dataset_path}/noise_data_{NOISE_GENERATION_MODE}_users.txt")
        print("=" * 100 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR during noise generation:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


