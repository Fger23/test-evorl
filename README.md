<h1 align="center">Evo-RL</h1>

<p align="center">
  <a href="https://MINT-SJTU.github.io/Evo-RL/"><img alt="项目网站" src="https://img.shields.io/badge/项目-网站-0ea5e9"/></a>
  <a href="https://github.com/huggingface/lerobot"><img alt="lerobot 版本" src="https://img.shields.io/badge/LeRobot-0.4.4-f59e0b"/></a>
  <a href="https://evorl.example.com/wechat-post"><img alt="微信推文" src="https://img.shields.io/badge/微信-官方推文-07c160"/></a>
  <a href="#社区渠道"><img alt="微信群加入我们" src="https://img.shields.io/badge/微信群-加入我们-a855f7?logo=wechat&logoColor=white"/></a>
  <a href="#引用"><img alt="论文即将发布" src="https://img.shields.io/static/v1?label=论文&message=即将发布&color=9ca3af"/></a>
  <a href="#模型--数据集"><img alt="hugging face 模型即将发布" src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20模型&message=即将发布&color=9ca3af"/></a>
  <a href="#模型--数据集"><img alt="hugging face 数据集即将发布" src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20数据集&message=即将发布&color=9ca3af"/></a>
  <a href="./LICENSE"><img alt="许可证" src="https://img.shields.io/badge/许可证-Apache--2.0-ef4444"/></a>
</p>

<p align="center"><strong>上海交通大学 (SJTU) &amp; Evo-Tech</strong></p>

<p align="center"><strong>架构总览</strong></p>

<p align="center">
  <img alt="Evo-RL 流水线总览" src="./website/assets/images/overview.png" width="96%"/>
</p>

## 🎯 Evo-RL 的定位

- **在两个平台上开放真机 RL**：我们在 SO101 和 AgileX (PiPER/PiPER-X) 上构建并发布了完整的真机 RL 流水线。
- **开放代码、模型和数据集以支持复现**：持续发布可运行的离线 RL 资产，让更多人能够复现结果并应用到真实任务中。
- **开放算法与社区共同进化**：我们复现已有的真机 RL 方法、提出新方法，并持续发布数据/基准，推动协作型开源社区的发展。

## 🚀 最新动态

- **[2026-03-07]** 真机 RL 新增 AgileX (PiPER/PiPER-X) 支持。
- **[2026-02-26]** 首个 SO101 真机 RL 基线及可复现的 CLI 工作流发布。

## 🧭 目录

| 入门                                   | 训练流水线                                                   | 项目信息                                     |
| -------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| [⚡ 快速开始](#快速开始)               | [4) 价值函数训练](#4-价值函数训练)                           | [模型 & 数据集](#模型--数据集)               |
| [1) 安装](#1-安装)                     | [5) 价值推理](#5-价值推理)                                   | [社区渠道](#社区渠道)                        |
| [2) 硬件配置](#2-硬件配置)             | [6) 策略训练](#6-策略训练)                                   | [隶属机构](#隶属机构)                        |
| [3) 数据采集](#3-数据采集)             | [7) 闭环部署与下一轮迭代](#7-闭环部署与下一轮迭代)           | [引用](#引用) / [许可证](#许可证)            |

<p align="center"><strong>价值函数可视化结果</strong></p>

<p align="center"><small><strong>成功案例</strong></small></p>

<p align="center">
  <img alt="价值叠加 成功 episode 0405" src="./website/assets/gifs/value_success.gif" width="96%"/>
</p>

<p align="center"><small><strong>失败案例</strong></small></p>

<p align="center">
  <img alt="价值叠加 失败 episode 0697" src="./website/assets/gifs/value_failure.gif" width="96%"/>
</p>

<p align="center"><strong>策略部署可视化结果</strong></p>

<p align="center">
  <img alt="策略部署结果 1" src="./website/assets/gifs/policy_rollout_1.gif" width="48%"/>
  <img alt="策略部署结果 2" src="./website/assets/gifs/policy_rollout_2.gif" width="48%"/>
</p>

<p align="center"><strong>人机协同（Human-in-the-Loop）可视化结果</strong></p>

<p align="center">
  <img alt="人机协同结果 1" src="./website/assets/gifs/hitl_1.gif" width="48%"/>
  <img alt="人机协同结果 2" src="./website/assets/gifs/hitl_2.gif" width="48%"/>
</p>

<a id="快速开始"></a>

## ⚡ 快速开始

**基于 LeRobot 的基座**：我们使用 LeRobot 作为本代码库的基础，因为它的推理和数据采集逻辑与真机 RL 的工作流程高度契合。

<a id="1-安装"></a>

### 1) 安装

```bash
git clone https://github.com/MINT-SJTU/Evo-RL.git
cd Evo-RL
conda create -y -n evo-rl python=3.10
conda activate evo-rl
pip install -e .
```

详细的安装步骤和平台特定依赖，请参考 [LeRobot 官方配置指南](https://huggingface.co/docs/lerobot/installation)。

<a id="2-硬件配置"></a>

### 2) 硬件配置

#### SO 系列（SO100/SO101）

SO 系列的配置请严格按照 [官方教程](https://wiki.seeedstudio.com/cn/lerobot_so100m/) 完成所有安装和设置步骤后再继续。
以下示例以 **SO101** 作为参考配置。

#### 设备路径建议

推荐的路径策略：

- **机器人串口**：使用 `/dev/serial/by-id/`（重启后路径稳定）。
- **相机**：优先使用 `/dev/v4l/by-id/`；如果 ID 不唯一，则使用 `/dev/v4l/by-path/`。
- 下方示例中：机器人端口使用 `by-id`，相机路径使用 `by-path`。

可以用以下命令查看可用的稳定路径：

```bash
ls -l /dev/serial/by-id/
ls -l /dev/v4l/by-id/
ls -l /dev/v4l/by-path/
```

**单臂用户**无需做大的改动。配置完成后，运行以下命令验证系统是否就绪：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<SO101_FOLLOWER_PORT> \
  --robot.id=my_so101_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<SO101_LEADER_PORT> \
  --teleop.id=my_so101_leader
```

**双臂用户**，我们建议将左主控臂和左从动臂上对应 4/5/6 号舵机的机械部件做镜像处理，这样通常能获得更自然的双手操作手感。

运行双臂命令前，请确保 `~/.cache/huggingface/lerobot/calibration/` 下存在以下校准文件：

```text
calibration/
├── robots
│   └── so_follower
│       ├── bi_so101_follower_left.json
│       └── bi_so101_follower_right.json
└── teleoperators
    └── so_leader
        ├── bi_so101_leader_left.json
        └── bi_so101_leader_right.json
```

该布局与单臂设置略有不同。

然后运行以下命令验证双臂配置：

```bash
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=bi_so101_follower \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=bi_so101_leader
```

#### 相机配置

在采集数据前，先验证相机映射。

检查每个相机是否支持目标参数（例如 `640x480 @ 30`）：

```bash
v4l2-ctl -d /dev/v4l/by-path/<CAM_PATH> --list-formats-ext
```

单臂相机检查（示例）：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<SO101_FOLLOWER_PORT> \
  --robot.id=my_so101_follower \
  --robot.cameras='{ front: {type: opencv, index_or_path: "/dev/v4l/by-path/<FRONT_CAM>", width: 640, height: 480, fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<SO101_LEADER_PORT> \
  --teleop.id=my_so101_leader \
  --display_data=true
```

双臂相机检查（示例）：

```bash
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=my_bi_so101_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<LEFT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<RIGHT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: "/dev/v4l/by-path/<FRONT_CAM_PATH>", width: 640, height: 480, fps: 30}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=my_bi_so101_leader \
  --display_data=true
```

对于双臂相机映射，`front` 挂在左臂或右臂的相机配置下都可以。如果使用更多视角的相机，同样放在任意一侧臂的相机配置下即可。

如有需要，初始调试时也可以使用临时设备路径（例如 `/dev/ttyACM*` 和 `/dev/video*`）。

<a id="agilex-piper-setup"></a>

#### AgileX（PiPER/PiPER-X）

PiPER 机械臂在主控/示教模式下无法接收外部控制指令，因此所有机械臂必须配置为从动/运动输出模式（0xFC），固件版本需为 1.8.5 或以上。

对于 PiPER 系列机器人，在运行遥操作前，请确保已拉取 Git LFS 资产：

```bash
git lfs pull --include="src/lerobot/assets/piper_description/**,src/lerobot/assets/piper_x_description/**" --exclude="*"
git lfs checkout src/lerobot/assets/piper_description src/lerobot/assets/piper_x_description
```

PiPER 使用 CAN 接口而非串口。先运行 `lerobot-setup-can` 确认 CAN 接口可用：

```bash
lerobot-setup-can --mode=setup --interfaces=<LEFT_FOLLOWER_CAN_PORT>,<LEFT_LEADER_CAN_PORT>,<RIGHT_FOLLOWER_CAN_PORT>,<RIGHT_LEADER_CAN_PORT>
```

**单臂用户**，运行以下命令验证系统就绪：

```bash
lerobot-teleoperate \
  --robot.type=piperx_follower \
  --robot.port=<FOLLOWER_CAN_PORT> \
  --robot.id=my_piperx_follower \
  --robot.require_calibration=false \
  --teleop.type=piperx_leader \
  --teleop.port=<LEADER_CAN_PORT> \
  --teleop.id=my_piperx_leader \
  --teleop.require_calibration=false
```

**双臂用户**，运行以下命令验证双臂遥操作：

```bash
lerobot-teleoperate \
  --robot.type=bi_piperx_follower \
  --robot.id=my_bi_piperx_follower \
  --robot.left_arm_config.port=<LEFT_FOLLOWER_CAN_PORT> \
  --robot.right_arm_config.port=<RIGHT_FOLLOWER_CAN_PORT> \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --teleop.type=bi_piperx_leader \
  --teleop.id=my_bi_piperx_leader \
  --teleop.left_arm_config.port=<LEFT_LEADER_CAN_PORT> \
  --teleop.right_arm_config.port=<RIGHT_LEADER_CAN_PORT> \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false
```

如果是 PiPER（非 X 版），将 `bi_piperx_follower`/`bi_piperx_leader` 替换为 `bi_piper_follower`/`bi_piper_leader`。

<a id="3-数据采集"></a>

### 3) 数据采集

使用 `lerobot-human-inloop-record` 采集 rollout 数据。

#### SO 系列（SO100/SO101）

双臂模板：

```bash
lerobot-human-inloop-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=my_bi_so101_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<LEFT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<RIGHT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "<REALSENSE_SN>", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=my_bi_so101_leader \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --dataset.single_task="<YOUR_TASK_DESCRIPTION>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.episode_time_s=<EPISODE_SECONDS> \
  --dataset.reset_time_s=<RESET_SECONDS> \
  --dataset.push_to_hub=true \
  --display_data=true
```

建议：OpenCV 相机使用 **`fourcc: "MJPG"`**，RealSense 相机使用 **`warmup_s`**。本例中 `front` 用的是 RealSense，你也可以用同样的结构换成 OpenCV。

#### AgileX（PiPER/PiPER-X）

双臂模板（左/右，以 PiPER-X 为例）：

```bash
lerobot-human-inloop-record \
  --robot.type=bi_piperx_follower \
  --robot.id=my_bi_piperx_follower \
  --robot.left_arm_config.port=<LEFT_FOLLOWER_CAN_PORT> \
  --robot.right_arm_config.port=<RIGHT_FOLLOWER_CAN_PORT> \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --teleop.type=bi_piperx_leader \
  --teleop.id=my_bi_piperx_leader \
  --teleop.left_arm_config.port=<LEFT_LEADER_CAN_PORT> \
  --teleop.right_arm_config.port=<RIGHT_LEADER_CAN_PORT> \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --dataset.single_task="<YOUR_TASK_DESCRIPTION>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.episode_time_s=<EPISODE_SECONDS> \
  --dataset.reset_time_s=<RESET_SECONDS> \
  --dataset.push_to_hub=true \
  --display_data=true
```

快捷键：

- `i`：切换介入模式（策略 ↔ 遥操作接管）
- `s`：标记成功并结束当前 episode
- `f`：标记失败并结束当前 episode
- `右箭头`：提前结束当前循环
- `左箭头`：提前结束并重新录制当前 episode
- `Esc`：停止录制会话

快速质量检查：

```bash
lerobot-dataset-report --dataset <HF_USERNAME_OR_ORG>/<DATASET_NAME>
```

会输出：数据集元信息、总量统计、episode 长度统计/直方图、成功/介入指标、任务列表以及完整的特征 schema。

<a id="4-价值函数训练"></a>

### 4) 价值函数训练

在当前数据集上训练价值函数。当前默认：[Pi\*0.6](https://www.pi.website/blog/pistar06)（`--value.type=pistar06`）。

**单 GPU 模板：**

```bash
lerobot-value-train \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.push_to_hub=true \
  --value.repo_id=<HF_USERNAME_OR_ORG>/<VALUE_MODEL_REPO> \
  --batch_size=64 \
  --output_dir=outputs/value_train/<RUN_NAME> \
  --job_name=<RUN_NAME> \
  --wandb.enable=true
```

**多 GPU 模板：**

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID_LIST> accelerate launch \
  --multi_gpu \
  --num_processes=<NUM_GPUS> \
  --mixed_precision=bf16 \
  $(which lerobot-value-train) \
  --batch_size=32/<NUM_GPUS> \
  <VALUE_TRAIN_ARGS>
```

**接入自定义价值函数的最小步骤：**

- 在 `src/lerobot/values/<your_value>/configuration_<your_value>.py` 中添加 `@PreTrainedConfig.register_subclass("<your_value>")`。
- 在 `src/lerobot/values/<your_value>/modeling_<your_value>.py` 中添加 `<YourValue>Policy(PreTrainedPolicy)`（至少实现 `forward`、`predict_value` 和供 `lerobot-value-train` 使用的 `build_training_raw_batch_hook`）。
- 在 `src/lerobot/values/<your_value>/processor_<your_value>.py` 中添加 `make_<your_value>_pre_post_processors(...)`。
- 在 `src/lerobot/configs/value_train.py` 和 `src/lerobot/scripts/lerobot_value_infer.py` 中移除/替换当前仅限 `pistar06` 的类型检查。

<a id="5-价值推理"></a>

### 5) 价值推理

推理价值信号并将 value/advantage/indicator 写回数据集：

- `value`：当前帧的估计 return-to-go（未来累计回报）。
- `advantage`：相对改进信号（值越高表示轨迹质量优于基线）。
- `indicator`：由 advantage 二值化得到的训练标签。

**单 GPU 模板：**

```bash
lerobot-value-infer \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --inference.checkpoint_path=outputs/value_train/<RUN_NAME> \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_<TAG> \
  --acp.advantage_field=complementary_info.advantage_<TAG> \
  --acp.indicator_field=complementary_info.acp_indicator_<TAG> \
  --output_dir=outputs/value_infer/<RUN_NAME> \
  --job_name=<RUN_NAME>.infer
```

**多 GPU 模板：**

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID_LIST> accelerate launch \
  --multi_gpu \
  --num_processes=<NUM_GPUS> \
  --mixed_precision=bf16 \
  $(which lerobot-value-infer) \
  <VALUE_INFER_ARGS>
```

参数说明：

```bash
--acp.n_step：n-step advantage 的时间跨度。
--acp.positive_ratio：advantage 二值化后正样本的比例（例如 0.3 = 每个任务取前 30%）。
```

预期新增列：

```bash
complementary_info.value_<TAG>
complementary_info.advantage_<TAG>
complementary_info.acp_indicator_<TAG>
```

这些列会写回到 `--dataset.repo_id` 指定的原始数据集中。

<a id="6-策略训练"></a>

### 6) 策略训练

使用 advantage 条件标签训练策略。
**策略要求**：必须支持**文本/任务输入**，因为 Advantage-Conditioned 标签会注入到**任务文本**中。

**单 GPU 模板：**

```bash
lerobot-train \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --policy.type=<POLICY_TYPE> \
  --policy.pretrained_path=<POLICY_PRETRAINED_PATH> \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=32 \
  --steps=30000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_<TAG> \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=outputs/train/<RUN_NAME> \
  --job_name=<RUN_NAME> \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=<HF_USERNAME_OR_ORG>/<POLICY_REPO>
```

`--acp.indicator_dropout_prob` 控制任务文本中标签的丢弃率；`0.3` 有助于同时学习带标签和不带标签的条件。

**重要检查**：

- `--acp.indicator_field` 必须存在于数据集中，且为**二值（`0/1`）**。

**多 GPU 模板：**

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID_LIST> accelerate launch \
  --multi_gpu \
  --num_processes=<NUM_GPUS> \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --batch_size=32/<NUM_GPUS> \
  <POLICY_TRAIN_ARGS>
```

<a id="7-闭环部署与下一轮迭代"></a>

### 7) 闭环部署与下一轮迭代

在人机协同模式下部署训练好的策略，采集下一轮数据集：

```bash
lerobot-human-inloop-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=my_bi_so101_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<LEFT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<RIGHT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "<REALSENSE_SN>", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=my_bi_so101_leader \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME_NEXT_ROUND> \
  --dataset.single_task="<YOUR_TASK_DESCRIPTION>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.episode_time_s=<EPISODE_SECONDS> \
  --dataset.reset_time_s=<RESET_SECONDS> \
  --dataset.push_to_hub=true \
  --display_data=true \
  --policy.path=<POLICY_CHECKPOINT_OR_HUB_ID> \
  --resume=true
```

数据集续写选项：

- **原地追加**：保持 `--resume=true`，继续往同一个数据集录制。
- **合并多轮**：使用官方数据集编辑器合并不同数据集。

```bash
lerobot-edit-dataset \
  --repo_id=<HF_USERNAME_OR_ORG>/<MERGED_DATASET_NAME> \
  --operation.type=merge \
  --operation.repo_ids="['<HF_USERNAME_OR_ORG>/<DATASET_ROUND_1>','<HF_USERNAME_OR_ORG>/<DATASET_ROUND_2>']"
```

相比默认 `lerobot-record` 行为，额外记录的数据属性：

- `complementary_info.policy_action`：每一步策略输出的动作。
- `complementary_info.is_intervention`：当前步是否处于人工介入。
- `complementary_info.state`：介入状态机的状态。
- `complementary_info.collector_policy_id`：逐步动作来源 ID（`human` 或策略 ID）。
- Episode 元数据 `episode_success`：按 episode 保存的成功/失败标签。

迭代训练循环（抽象表示）：

```text
[多任务示教数据池]
        |
        v
[针对视觉-语言-动作策略的离线 RL 预训练]
        |
        v
[基于示教的任务特定初始化 / 微调]
        |
        v
|---- 迭代 k = 1..K -------------------------------------------|
| 1) 部署当前策略 π_k 并采集新的 rollout 数据                  |
| 2) 合并到数据池：D <- D ∪ new_data                           |
| 3) 在 D 上训练价值函数                                       |
| 4) 推理 advantage 并二值化为 indicator 标签                  |
| 5) 训练 advantage 条件策略，得到 π_{k+1}                     |
|--------------------------------------------------------------|
        |
        v
[更强的策略，成功率与吞吐量均提升]
```

## 模型 & 数据集

- Hugging Face 模型发布：即将上线
- Hugging Face 数据集发布：即将上线
- 发布后，本节将固定标准仓库和确切版本标签。

## 社区渠道

- 微信公众号文章：[即将上线](https://evorl.example.com/wechat-post)
- 文档：[`docs/README.md`](./docs/README.md)
- GitHub Issues：[提交 issue](https://github.com/MINT-SJTU/Evo-RL/issues)
- 邮箱：business@evomind-tech.com
- 微信群二维码：

<p align="center">
  <img alt="EvoMind 微信二维码" src="./website/assets/images/rlgroup.jpg" width="220"/>
</p>

- SO101 设备提供商微信联系方式：
<p align="center">
  <img alt="SO101 设备提供商" src="./website/assets/images/so101provider.jpg" width="220"/>
</p>

## 隶属机构

<p align="center">
  <img alt="上海交通大学社区" src="./website/assets/images/sjtu.png" height="68"/>
  <img alt="EvoMind" src="./website/assets/images/evomind1.png" height="60"/>
</p>

## 引用

```bibtex
@misc{evorl2026,
  title        = {Evo-RL: Towards Iterative Policy Improvement in Real-World Offline RL},
  author       = {Evo-RL Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/MINT-SJTU/Evo-RL}}
}
```

## 许可证

Apache-2.0。详见 [LICENSE](./LICENSE)。

## Star 历史

[![Star 历史图表](https://api.star-history.com/image?repos=MINT-SJTU/Evo-RL&type=date&legend=top-left)](https://www.star-history.com/?repos=MINT-SJTU%2FEvo-RL&type=date&legend=top-left)
