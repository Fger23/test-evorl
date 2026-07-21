import argparse
import logging
import shutil
from pathlib import Path
from typing import List

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
# from lerobot.datasets.lerobot_dataset import (
#     LeRobotDataset,
#     LeRobotDatasetMetadata,
#     MultiLeRobotDataset,
# )
from tqdm import tqdm
import einops
import numpy as np

# 配置日志 (包含文件名和行号)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

def parse_image(image) -> np.ndarray:
    # print('image shape:', image.shape) # image shape: torch.Size([3, 480, 640])
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

def _ensure_human_inloop_compatible_features(
    dataset_features: dict[str, dict],
    *,
    action_feature_names: list[str],
) -> None:
    # Keep human-in-loop datasets schema-stable across teleop-only and policy-assisted phases so they can merge.
    dataset_features["complementary_info.policy_action"] = {
        "dtype": "float32",
        "shape": (len(action_feature_names),),
        "names": action_feature_names,
    }
    dataset_features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_intervention"],
    }
    dataset_features["complementary_info.state"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state"],
    }

def resolve_collector_policy_id(
    *,
    intervention_enabled: bool,
    is_intervention: bool,
    selected_from_policy: bool,
    policy_id: str,
    human_id: str,
) -> str:
    """Resolve frame-level `collector_policy_id` from control mode and source."""
    if intervention_enabled:
        return human_id if is_intervention else policy_id
    return policy_id if selected_from_policy else human_id

def lerobot_to_evo_datasets(input_dir: Path, output_dir: Path):
    """
    将LeRobot数据集转换为EVO-RL数据集
    
    """

    # 验证所有输入目录都存在
    if not input_dir.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    
    # 加载第一个数据集的元数据作为模板
    input_metadata = LeRobotDatasetMetadata("", root=input_dir)
    dataset_features = input_metadata.features.copy()
    action_names = dataset_features["action"]["names"]
    action_names = list(action_names)
    _ensure_human_inloop_compatible_features(dataset_features, action_feature_names=action_names)
    dataset_features["complementary_info.collector_policy_id"] = {
            "dtype": "string",
            "shape": (1,),
            "names": ["collector_policy_id"],
        }
    # logging.info(f"转换后的数据集的features: {dataset_features}")
    # exit()

    action_feature_names = dataset_features["action"]["names"]
    zero_policy_action = dict.fromkeys(action_feature_names, 0.0)
    
    # 创建新的EVO-RL数据集
    logging.info(f"创建EVO-RL后的数据集: {output_dir}")
    output_dataset = LeRobotDataset.create(
        repo_id=None,
        robot_type=input_metadata.robot_type,
        root=output_dir,
        fps=input_metadata.fps,
        use_videos=True,
        features=dataset_features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    
    
    total_episodes = 0


    # 开始转换数据
    is_intervention = 0.0
    intervention_state = 0.0
    intervention_enabled = False
    selected_from_policy = False
    collector_policy_id_policy = "policy"
    collector_policy_id_human = "human"
    # 加载当前数据集的元数据
    metadata = LeRobotDatasetMetadata("", root=input_dir)
    for episode_idx in tqdm(range(len(metadata.episodes))):
        # try:
        if True:
            # 加载原始episode数据
            dataset = LeRobotDataset("", input_dir, episodes=[episode_idx])
            
            # 遍历每个step并添加到合并的数据集
            for data_item in dataset:
                # 准备图像数据
                image_dict = {}
                for key, value in data_item.items():
                    if "observation.image" in key:
                        image_key = key.split(".")[-1]
                        image_dict[f"observation.images.{image_key}"] = parse_image(value)
                frame = {
                        **image_dict,
                        "observation.state": data_item["observation.state"],
                        "action": data_item["action"],
                        "task": data_item["task"],
                    }
                frame["complementary_info.policy_action"] = np.array([zero_policy_action[name] for name in action_feature_names], dtype=np.float32)
                frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
                frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
                frame["complementary_info.collector_policy_id"] = resolve_collector_policy_id(
                    intervention_enabled=intervention_enabled,
                    is_intervention=bool(is_intervention),
                    selected_from_policy=selected_from_policy,
                    policy_id=collector_policy_id_policy,
                    human_id=collector_policy_id_human,
                )
                # 添加帧数据
                output_dataset.add_frame(frame)
            
            # 保存当前episode
            extra_episode_metadata = (
                    {"episode_success": "success"}
                )
            output_dataset.save_episode(extra_episode_metadata=extra_episode_metadata)
            
        # except Exception as e:
        #     logging.error(f"合并数据集 {dataset_idx + 1} 的episode {episode_idx} 时出错: {e}")
        #     continue
    
    logging.info(f"数据集转换完成! 输出目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="合并多个LeRobot数据集")
    
    parser.add_argument("--input-dir", type=Path, required=True,
                       help="输入LeRobot数据集")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="输出转换后的数据集目录")

    args = parser.parse_args()
    
    lerobot_to_evo_datasets(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )



if __name__ == "__main__":
    main()

    '''
    用法示例:

    python lerobot_to_evo_dataset.py \
        --input-dir /nfs/lerobot/s101/datasets/aggregation/take_and_read_card_06160617/ \
        --output-dir /nfs/lerobot/s101/datasets/aggregation/evo-rl/take_and_read_card_06160617/

    python lerobot_to_evo_dataset.py \
        --input-dir /nfs/lerobot/s101/datasets/aggregation/bimanual_banknotes_0625_26/ \
        --output-dir /nfs/lerobot/s101/datasets/aggregation/evo-rl/bimanual_banknotes_0625_26/
        
    '''