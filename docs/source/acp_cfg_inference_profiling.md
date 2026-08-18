# ACP-CFG batch=2 推理采集

这个版本用于先测量 Pi0.5 的双分支 CFG 推理延迟和动作分布，暂不启用 RTC。

启用后，每次动作缓存耗尽时只执行一次模型调用：

1. 构造固定顺序的 batch：第 0 行为原始任务 `U`，第 1 行为带
   `Advantage: positive` 的任务 `C`。
2. 生成一份 `[1, T, Amax]` 初始噪声，并复制成两行完全相同的噪声。
3. 一次 `predict_action_chunk()` 得到 `U` 和 `C`。
4. 在模型归一化动作空间计算 `A = U + beta * (C - U)`。
5. 对合成后的整个 chunk 只执行一次 postprocess，并把前
   `policy.n_action_steps` 个动作放入缓存逐帧执行。

## 启动参数

在现有 `lerobot-record` 或 `lerobot-human-inloop-record` 命令后添加：

```shell
--acp_inference.enable=true \
--acp_inference.use_cfg=true \
--acp_inference.batched_cfg=true \
--acp_inference.cfg_beta=1.5 \
--acp_inference.profile=true \
--acp_inference.profile_output_dir=outputs/acp_inference_profile \
--acp_inference.profile_run_name=h200_trial_01 \
--acp_inference.profile_warmup_chunks=3 \
--acp_inference.profile_save_chunks=false
```

其余机器人、数据集、任务和 `--policy.path` 参数沿用原来的推理命令。
当前 batch CFG 仅支持 Pi0.5；采集基线时必须关闭 `policy.rtc_config`。

默认不保存完整 tensor，避免同步磁盘写入拖慢机器人控制循环。需要检查每次推理的
噪声、U/C 和 CFG chunk 时，可在单独的低风险采集运行中设置：

```shell
--acp_inference.profile_save_chunks=true
```

## 输出

每次运行保存在：

```text
outputs/acp_inference_profile/h200_trial_01/
├── records.jsonl
├── summary.json
└── chunks/            # 仅在 profile_save_chunks=true 时存在
```

`records.jsonl` 每个模型 chunk 一行，包含端到端 wall latency、模型 CUDA latency、
显存峰值、共享噪声误差、raw CFG 越界比例、U/C 差异和缓存步数等。

`summary.json` 同时给出全样本与排除 warm-up 后的稳态 p50/p95/p99/max。
`D99 = ceil(steady_wall_p99 * fps)`，并给出两个候选
`execution_horizon`。后续实现 RTC 时应优先查看：

- `recommendation_source` 是否为 `steady_state`；
- `steady_state_count` 是否足够（建议至少 30 个 chunk）；
- `D99`、`recommended_precision_horizon` 和 `recommended_smooth_horizon`；
- `raw_cfg_out_of_range_ratio` 是否持续偏高。

相同 `profile_run_name`、fps 和 beta 可以跨进程续写；换 fps 或 beta 时应使用新的
run name。当前同步基线只在缓存为 0 时触发推理，因此 `queue_depth_before=0` 是预期值；
未来异步 RTC 的提前触发阈值将根据这里测出的 `D99` 决定。
