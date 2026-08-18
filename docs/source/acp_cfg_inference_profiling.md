# H200 PolicyServer 的 ACP-CFG batch=2 纯推理测试

这个模式在 H200 的 `policy_server` 内测量 Pi0.5 双分支 CFG 推理延迟和动作分布，
暂不启用 RTC，也不要求客户端采集或保存 LeRobot 数据集。

每次服务端收到一次推理请求时：

1. 在 tokenizer 前构造固定顺序的 batch：第 0 行为原始任务 `U`，第 1 行为带
   `Advantage: positive` 的任务 `C`。
2. 生成一份 `[1, T, Amax]` 初始噪声，并复制成两行完全相同的噪声。
3. 一次 `predict_action_chunk()` 得到 `U` 和 `C`。
4. 在模型归一化动作空间计算 `A = U + beta * (C - U)`。
5. 对合成后的 `[1, T, A]` chunk 只执行一次 postprocess。
6. 按原 RPC 格式向客户端返回一个 CFG action chunk；客户端协议和执行逻辑保持不变。

## H200 服务端启动参数

把参数加到 H200 的 `policy_server` 命令，而不是客户端的
`lerobot-human-inloop-record` 命令：

```shell
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8099 \
  --fps=30 \
  --acp_inference.enable=true \
  --acp_inference.use_cfg=true \
  --acp_inference.batched_cfg=true \
  --acp_inference.cfg_beta=1.5 \
  --acp_inference.profile=true \
  --acp_inference.profile_output_dir=/local_nvme/acp_inference_profile \
  --acp_inference.profile_run_name=h200_trial_01 \
  --acp_inference.profile_warmup_chunks=3 \
  --acp_inference.profile_save_chunks=false
```

`--fps` 必须与客户端机器人控制频率一致。当前服务端 batch CFG 仅支持 Pi0.5，
并要求加载的 Pi0.5 checkpoint 中 `rtc_config` 为 `null`。
每次更换 checkpoint、`fps` 或 beta 后都应改用新的 `profile_run_name`，避免把不同实验混入
同一份统计；仅在完全相同的实验断点续测时复用名称，chunk 编号会从已有记录继续。

客户端继续使用原来的 `--remote_policy.enable=true` 命令，不要给客户端增加
`acp_inference` 参数。

建议把 profiling 输出写到 H200 本地 NVMe，而不是 NFS。默认只保存数值日志；需要检查
每次推理的噪声、U/C 和 CFG tensor 时，才在单独的低风险测试中设置：

```shell
--acp_inference.profile_save_chunks=true
```

## 输出

上述示例保存在：

```text
/local_nvme/acp_inference_profile/h200_trial_01/
├── records.jsonl
├── summary.json
└── chunks/            # 仅在 profile_save_chunks=true 时存在
```

`records.jsonl` 每个服务端模型 chunk 一行，包含：

- H200 从 raw observation 处理到 action chunk 完成 D2H 的 wall latency；
- Pi0.5 模型调用的 CUDA latency、绝对显存峰值和本次推理显存峰值增量；
- 共享噪声误差、raw CFG 越界比例、U/C 差异；
- prepare、preprocess、model+CFG、postprocess+D2H 的分段耗时；
- checkpoint、客户端 observation timestep 和实际返回步数。

`summary.json` 给出全样本与排除 warm-up 后的稳态 p50/p95/p99/max，
并计算 `D99 = ceil(steady_wall_p99 * fps)` 和候选 `execution_horizon`。
建议至少积累 30 个稳态 chunk 后再选 RTC 参数。

这里的 wall latency 不包含网络传输、客户端 action queue 和 profiling 文件写入时间，
所以当前 `D99` 是 H200 推理侧基线。后续实现 RTC 时，还需要客户端上报真实
`left_over` 和时间戳，才能得到端到端 `inference_delay`。
