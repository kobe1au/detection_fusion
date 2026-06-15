# 融合互补性测试工具

这个目录用于评估多个 `best.pt` 在同一数据集上的预测互补性，帮助判断：

- 融合跑不过最强单模态，是不是因为 API 与程序图本身没有互补收益；
- API / graph / concat / cross-attention / ours 的错误样本是否高度重合；
- 理论 oracle fusion 上限有多高；
- 简单概率加权 ensemble 是否已经能超过单模态。

推荐直接使用 `test_m` 里的通用脚本：

```bash
python test_m/complementarity.py \
  --base config/base.yaml \
  --split test \
  --model api=experiments/api_baseline/42/best_api_baseline.pt \
  --model graph=experiments/gatv2_baseline/42/best_gatv2_baseline.pt \
  --model late_fusion=experiments/api_graph_late_fusion/42/best_api_graph_late_fusion.pt \
  --model concat=experiments/api_graph_concat/42/best_api_graph_concat.pt \
  --model cross_attention=experiments/api_graph_cross_attention/42/best_api_graph_cross_attention.pt \
  --model ours=experiments/api_graph_ours_quality_adaptive_align/42/best_api_graph_ours_quality_adaptive_align.pt
```

如果你的 checkpoint 路径就是默认命名，也可以直接：

```bash
python test_m/complementarity.py --base config/base.yaml --split test --use-default-models
```

只评估 API 单模态和 graph 单模态互补性：

```bash
python test_m/single_modal_complementarity.py \
  --base config/base.yaml \
  --split test \
  --api-ckpt experiments/api_baseline/42/best_api_baseline.pt \
  --graph-ckpt experiments/gatv2_baseline/42/best_gatv2_baseline.pt
```

主要输出文件：

- `model_metrics.csv`：每个模型自己的 F1 / Acc。
- `pairwise_complementarity.csv`：两两互补性。
- `multi_model_oracle.json`：所有模型的理论 oracle 上限。
- `predictions_{name}.csv`：每个样本的预测、置信度和是否预测正确。

如果 `oracle_gain_over_best_f1 < 0.01`，通常说明两个模态错误高度重合，融合很难在 clean F1 上明显超过单模态。
