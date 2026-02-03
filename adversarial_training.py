import sys
import os
from pathlib import Path

# ============================================================
# 【核心修复】动态添加子模块路径到 sys.path
# ============================================================
# 获取当前脚本所在目录 (例如 .../mycode/)
current_dir = Path(__file__).resolve().parent
# 构建子模块代码目录路径 (例如 .../mycode/noise_discriminator/code/)
submodule_path = current_dir / "noise_discriminator" / "code"

# 将该路径添加到 sys.path 的最前面
if str(submodule_path) not in sys.path:
    sys.path.insert(0, str(submodule_path))
# ============================================================

import json
from typing import Dict, Any
import numpy as np
import torch

# -------------------- imports 修改 --------------------
# 现在 sys.path 里已经包含了 noise_discriminator/code/
# 所以我们可以直接 import 里面的模块，就像在那个目录下运行一样！

import world  # 之前是 import noise_discriminator.code.world as world
import utils  # 之前是 import noise_discriminator.code.utils as utils
from world import cprint

# 注意：这里直接引用 model，而不是 noise_discriminator.code.model
from model import NoiseDiscriminator
import Procedure

import dataloader # 直接 import dataloader

# 你的其他 import
from agents.adjudicator import Adjudicator
from util import init_openai_api, read_json  # 这里注意：跟上面的 utils 是否重名冲突？



# ============================================================
# 加载 LLM + Agent
# ============================================================

def load_llm_and_agent(dataset_name: str, project_root: Path) -> Adjudicator:

    api_config_path = project_root / "config" / "api-config.json"
    init_openai_api(read_json(str(api_config_path)))

    agents_config_dir = project_root / "config" / "agents"
    prompts_dir = project_root / "config" / "prompts"

    prompt_config_path = prompts_dir / "adjudicator.json"
    llm_config_path = agents_config_dir / "adjudicator.json"

    with open(llm_config_path, "r", encoding="utf-8") as f:
        llm_config = json.load(f)

    adjudicator = Adjudicator(
        prompt_config=str(prompt_config_path),
        dataset=dataset_name,
        llm_config=llm_config,
        web_demo=False,
    )

    cprint("Adjudicator initialized for adversarial training.")
    return adjudicator


# ============================================================
# 加载用户偏好
# ============================================================

def load_user_preferences(dataset_name: str, project_root: Path):
    pref_path = project_root / "data" / dataset_name / "user_preferences.json"
    if not pref_path.exists():
        raise FileNotFoundError(
            f"user_preferences.json not found at {pref_path}. "
            f"Run extract_user_preferences_only() first."
        )

    with open(pref_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {int(uid): pref for uid, pref in data.items()}


# ============================================================
# 生成 init_data（LLM 用的历史记录文本）
# ============================================================

def load_init_data_for_users(dataset_name: str, project_root: Path, max_history_len=50):
    train_file = project_root / "data" / dataset_name / "train.txt"
    item_list_file = project_root / "data" / dataset_name / "item_list.txt"

    users_data = Adjudicator.prepare_init_data(
        train_file=str(train_file),
        item_list_file=str(item_list_file),
        max_history_len=max_history_len,
    )

    cprint(f"Prepared init_data for {len(users_data)} users.")
    return users_data


# ============================================================
# 构造对抗 batch（真实 + Agent 噪声）
# ============================================================

def build_mixed_batch_for_user(
    dataset,
    target_user_id: int,
    noise_items: np.ndarray,
    num_clean_per_round: int = 10,
):
    user_pos = dataset.getUserPosItems([target_user_id])[0]
    user_pos = np.array(list(user_pos), dtype=np.int64)

    if user_pos.size == 0:
        raise ValueError(f"User {target_user_id} has no interactions.")

    clean_count = min(num_clean_per_round, user_pos.size)
    clean_items = np.random.choice(user_pos, size=clean_count, replace=False)

    noise_items = np.array(noise_items, dtype=np.int64)

    users_clean = np.full(len(clean_items), target_user_id, dtype=np.int64)
    users_noise = np.full(len(noise_items), target_user_id, dtype=np.int64)

    batch_users = np.concatenate([users_clean, users_noise])
    batch_items = np.concatenate([clean_items, noise_items])
    batch_labels = np.concatenate([
        np.zeros_like(clean_items),
        np.ones_like(noise_items),
    ])

    return {
        "users": torch.from_numpy(batch_users),
        "items": torch.from_numpy(batch_items),
        "labels": torch.from_numpy(batch_labels),
    }


# ============================================================
# 主流程
# ============================================================
def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def main():
    set_seed(world.seed)

    dataset_name = world.dataset
    project_root = Path(__file__).parent

    cprint(f"Project root = {project_root}")

    # 1) 加载 dataset
    dataset_path = project_root / "data" / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    cprint(f"Loading dataset from: {dataset_path}")
    dataset = dataloader.Loader(path=str(dataset_path))
    cprint("Dataset loaded.")

    # 2) 初始化判别器 + 预训练权重
    Recmodel = NoiseDiscriminator(world.config, dataset).to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)

    weights_dir = project_root / "noise_discriminator" / "code" / "weights" / dataset_name
    pretrained_path = weights_dir / "final_noise_discriminator.pth"
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
    Recmodel.load_state_dict(torch.load(pretrained_path, map_location=world.device))
    cprint(f"Loaded pretrained discriminator from {pretrained_path}")

    # 3) 初始化 Agent + 用户数据
    adjudicator = load_llm_and_agent(dataset_name, project_root)
    user_preferences = load_user_preferences(dataset_name, project_root)
    users_init_data = load_init_data_for_users(dataset_name, project_root)

    # ====== 对抗训练超参数 ======
    MAX_ADVERSARIAL_ROUNDS = 20
    NOISE_RATIO = 0.3
    NUM_CLEAN_PER_ROUND = 10
    USERS_PER_ROUND = 5    # ✅ 每轮随机选 5 个用户

    # 可用用户集合：既在 user_preferences 又在 init_data 里
    available_user_ids = sorted(
        set(user_preferences.keys()) & set(users_init_data.keys())
    )
    cprint(f"Total users available for adversarial training: {len(available_user_ids)}")

    if len(available_user_ids) == 0:
        raise ValueError("No overlapping users between user_preferences and init_data.")

    # ====== 对抗训练循环（每轮多个用户） ======
    for round_idx in range(1, MAX_ADVERSARIAL_ROUNDS + 1):
        cprint(f"\n=== ROUND {round_idx}/{MAX_ADVERSARIAL_ROUNDS} ===")

        # 本轮要参与对抗的用户
        num_users_this_round = min(USERS_PER_ROUND, len(available_user_ids))
        sampled_users = np.random.choice(
            available_user_ids,
            size=num_users_this_round,
            replace=False,
        )

        cprint(f"Users in this round: {list(sampled_users)}")

        for uid in sampled_users:
            cprint(f"\n  [User {uid}]")

            # 3.1 设置 Agent 当前 user 状态
            adjudicator.user_id = int(uid)
            adjudicator.user_preferences = user_preferences[uid]
            adjudicator.is_initialized = True
            init_data = users_init_data[uid]["init_data"]

            # 3.2 Agent 生成边界噪声
            noise_pack = adjudicator.generate_preference_boundary_noise(
                init_data=init_data,
                noise_ratio=NOISE_RATIO,
            )
            noise_items = noise_pack.get("noise_items", [])
            cprint(f"    Generated {len(noise_items)} noise items.")
            if len(noise_items) == 0:
                cprint("    No noise items, skip this user.")
                continue

            # 3.3 构造该用户的混合 batch
            try:
                batch = build_mixed_batch_for_user(
                    dataset=dataset,
                    target_user_id=uid,
                    noise_items=np.array(noise_items, dtype=np.int64),
                    num_clean_per_round=NUM_CLEAN_PER_ROUND,
                )
            except ValueError as e:
                cprint(f"    ERROR building batch: {e}")
                continue

            # 3.4 判别器对该用户进行一次对抗 eval + train
            metrics = Procedure.adversarial_eval_and_train(
                Recmodel=Recmodel,
                loss_class=bpr,
                batch_users=batch["users"],
                batch_items=batch["items"],
                batch_noise_labels=batch["labels"],
            )

            cprint(
                f"    ACC={metrics['accuracy']:.4f}, "
                f"noise_loss={metrics['noise_loss']:.4f}, "
                f"avg_p_clean={metrics['avg_pred_noise_prob_clean']:.4f}, "
                f"avg_p_noise={metrics['avg_pred_noise_prob_noise']:.4f}, "
                f"FP={metrics['fp']}, FN={metrics['fn']}"
            )

            # 3.5 Agent 对这一次反馈进行反思
            reflection = adjudicator.reflect_on_adversarial_feedback(
                init_data=init_data,
                discriminator_feedback=metrics,
            )
            cprint(f"    Reflection: {reflection['reflection_summary'][:150]}...")

    # ====== 保存最终对抗训练后的权重 ======
    final_path = weights_dir / "final_noise_discriminator_adversarial_multiuser.pth"
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(Recmodel.state_dict(), final_path)
    cprint(f"\nSaved adversarially trained weights to:\n{final_path}")



if __name__ == "__main__":
    main()
