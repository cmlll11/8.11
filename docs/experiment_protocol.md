# MDL-UAP-v1 实验协议

## 研究问题

在同一个目标标签约束下，比较 clean classifier 和 backdoor classifier
达到 targeted ASR 门槛所需的最小 mapping 描述长度：

```text
B_f(t, tau) = min_g L_MDL(g)
```

第一版固定 `t=0`、`tau=0.90`，不实现 non-targeted UAP。

## Mapping

`targeted_universal` 使用一个固定扰动；`targeted_imdep` 使用一个共享的
输入相关生成器。两者均必须满足：

```text
||g(x) - x||_inf <= epsilon
```

clean 和 backdoor 侧使用相同的 GAP 结构、epsilon ladder、数据划分、训练
预算、restart 和 codec。

## 数据与资格门槛

- CIFAR-10；
- target label `0`；
- `split_seed=2026`；
- ASR 分母排除原始目标类样本；
- ε：`[4, 8, 16, 24, 32]/255`；
- clean accuracy ≥ 0.90；
- backdoor targeted ASR ≥ 0.90；
- clean model 上官方触发器 targeted ASR ≤ 0.10。

## 结果

验证集只用于选择 mapping。最终在独立测试集报告：

- targeted bit-ASR 曲线；
- `B_clean(0, 0.90)`；
- `B_backdoor(0, 0.90)`；
- 比值 `B_clean / B_backdoor`；
- 测试 targeted ASR；
- 输入保持指标；
- 训练时间和搜索成本。

服务器入口是 `scripts/train_mapping.py` 和 `scripts/evaluate_mapping.py`。
它们读取官方 BackdoorBench 的 `attack_result.pt`，并要求服务器上的 CIFAR-10
数据目录；运行前安装本仓库或设置 `PYTHONPATH=src`。

所有服务器步骤均由 `bash/` 下的脚本包装。BadNets 快速真实验证依次使用：

1. `train_badnet_pair.sh`：训练 seed-0 clean/backdoor 配对模型；
2. `check_badnet_pair.sh`：检查模型资格门槛；
3. `run_badnet_mapping_grid.sh`：生成 60 个正式 targeted GAP 候选；
4. `finalize_badnet_results.sh`：按验证集 MDL 选择并在测试集报告。
