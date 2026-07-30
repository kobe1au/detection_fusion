# Paper Baselines and Reporting Protocol

本目录保存论文对比方法、实验脚本与写作口径。当前主方法协议为
`tcp_joint_anchor_crc_v1`。旧协议生成的检查点、摘要、诊断文件和自然困难
子集不得混入当前结果表。

`review_draft14/` 与 `定稿13_统一图表/` 是历史导出材料，只用于追溯，不是
当前实验或汇总流程的输入。

## A. Android 恶意软件检测范式

由于当前三模态 PT 未保存部分原论文所需的原始特征模板，下列方法必须写成
adapted / inspired 基线，不得声称为严格复现。

凡是需要逐 epoch 选择 checkpoint 的经典基线，必须读取
`labels/validation_roles_protocol_v2.json`，且只使用其中固定的
`model_selection` 身份；不得按基线自己的 seed 重新切分验证集，也不得用
`decision_calibration` 选模型。角色加载会校验 `labels/val.csv` 的 SHA-256、
完整身份覆盖及 package-group 跨角色不相交。Drebin-style 与
MaMaDroid-inspired 当前采用固定超参数一次性拟合，因此不消费验证角色做
模型选择。

### Drebin-style sparse static baseline

使用当前 PT 中可获得的 Manifest 稀疏指示、API 哈希计数、API 语义类别计数
以及敏感 API 统计。

```bash
python -m paper.baselines.drebin_style_sparse \
  --train-pt-dir /root/autodl-tmp/pts_all \
  --val-pt-dir /root/autodl-tmp/pts_all \
  --test-pt-dir /root/autodl-tmp/pts_all \
  --train-csv labels/train.csv \
  --val-csv labels/val.csv \
  --test-csv labels/test.csv \
  --extra-test-csv labels/natural_subsets/test_branch_disagreement.csv \
  --extra-test-csv labels/natural_subsets/test_api_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_graph_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_manifest_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_competence_imbalance.csv \
  --extra-test-csv labels/natural_subsets/test_high_cross_modal_conflict.csv \
  --out-dir paper/outputs/drebin_style
```

### MalDozer-inspired API sequence baseline

仅使用 `api_ids` 训练轻量 API 序列 CNN，用于代表 API 序列深度检测范式，
不构成 MalDozer 的严格复现。

```bash
python -m paper.baselines.maldozer_inspired_api_sequence \
  --train-pt-dir /root/autodl-tmp/pts_all \
  --val-pt-dir /root/autodl-tmp/pts_all \
  --test-pt-dir /root/autodl-tmp/pts_all \
  --train-csv labels/train.csv \
  --val-csv labels/val.csv \
  --test-csv labels/test.csv \
  --validation-role-assignment labels/validation_roles_protocol_v2.json \
  --extra-test-csv labels/natural_subsets/test_branch_disagreement.csv \
  --extra-test-csv labels/natural_subsets/test_api_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_graph_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_manifest_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_competence_imbalance.csv \
  --extra-test-csv labels/natural_subsets/test_high_cross_modal_conflict.csv \
  --out-dir paper/outputs/maldozer_inspired
```

### MaMaDroid-inspired semantic Markov baseline

使用 `api_type_ids` 作为抽象 API 状态并建立转移矩阵。当前 PT 不含原始
package/family 名称，因此该方法只复现语义抽象与 Markov 建模思想。

```bash
python -m paper.baselines.mamadroid_inspired_markov \
  --train-pt-dir /root/autodl-tmp/pts_all \
  --val-pt-dir /root/autodl-tmp/pts_all \
  --test-pt-dir /root/autodl-tmp/pts_all \
  --train-csv labels/train.csv \
  --val-csv labels/val.csv \
  --test-csv labels/test.csv \
  --extra-test-csv labels/natural_subsets/test_branch_disagreement.csv \
  --extra-test-csv labels/natural_subsets/test_api_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_graph_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_manifest_only_wrong.csv \
  --extra-test-csv labels/natural_subsets/test_competence_imbalance.csv \
  --extra-test-csv labels/natural_subsets/test_high_cross_modal_conflict.csv \
  --out-dir paper/outputs/mamadroid_inspired
```

## B. 当前主方法身份

### Stage A：Joint 主专家与 atomic 辅助专家

- API、Graph、Manifest 三个 atomic 专家分别保留独立编码器和分类头。
- Joint 专家读取带 alive 掩码的三分支嵌入及 alive 位，是主分类器。
- Stage A 只使用干净训练样本，目标为
  `Joint CE + 0.25 × alive-masked atomic CE`。
- atomic CE 先在单样本的 alive atomic 专家间平均，再在有效样本间平均，
  避免缺失模态样本获得不成比例的损失权重。
- Stage A 检查点由Joint主专家在 `model_selection` 上的表现选择。

### I1：内容条件 competence 学习

- 对 API、Graph、Manifest、Joint 四个专家分别建立 competence head。
- 每个 head 只读取该专家当前嵌入与当前类别概率状态；`alive` 是硬可用性
  掩码。它不读取扰动类型、扰动强度、原始完整率、过滤前计数或其他专家
  特征。
- 连续监督目标为该专家对真实类别给出的概率
  \(q_e^\star=p_e(y)\)，即 TCP。目标、嵌入和专家概率均在 Stage B 与
  Stage A 边界处停止梯度。
- I1 目标由 TCP 回归和 atomic 专家样本内排序损失组成。训练池使用干净
  训练身份与每个身份的一种确定性单模态退化视图；退化辅助损失权重为
  0.25。每轮分别评价干净及三种固定退化验证源；先保留干净TCP损失位于
  最优值1%相对非劣带内的轮次，再最小化退化平均损失和最差来源损失。

### I2：Joint 锚定的单调 competence 融合

- alive atomic 专家的 late-fusion 权重由
  `relative_bias + positive_scale × log(q_m)` 产生，因此在其他条件不变时
  competence 越高，权重不会降低。
- Joint 与 atomic late fusion 的门控只依赖
  `log(q_late) - log(q_joint)`，斜率约束为正。
- I2 只更新小型路由参数，Stage A 专家和 I1 全部冻结。
- 每个候选与 Joint 都在同一 `model_selection` 干净身份上拟合各自的
  Macro-F1分类阈值，并把阈值固定应用于三种退化源。候选路由必须同时
  满足：干净性能不低于 Joint；每一种退化源都不低于 Joint；退化平均
  Macro-F1 严格优于 Joint。选中阈值随后直接锁定，不能二次拟合。否则
  部署安全回退并使用 Joint 自己的验证阈值：
  三模态完整时使用 Joint；Joint 不可用时对 alive atomic 专家均匀融合；
  全部不可用时输出类别均匀分布。

### I3：面向恶意漏报的 expected CRC

- 验证身份固定分为两个不相交角色：75% `model_selection`，25%
  `decision_calibration`。
- 前者用于 Stage A/Stage B 模型选择和普通分类阈值拟合；后者只用于 I3
  接受阈值校准。测试集不参与任何训练、选择或校准。
- 排序分数为 `malware_fn_probability_anchor`：已判为恶意的样本取1；
  已判为良性的样本取 \(1-p(\mathrm{malware})\)。分数越大表示越适合自动
  接受。
- CRC 使用有限样本修正
  \((N_{\mathrm{accepted\ FN}}+1)/(N_{\mathrm{malware}}+1)\le\alpha\)，
  在可行候选中最大化接受量。论文只能声明可交换性条件下的期望风险保证，
  不能表述为 \(1-\delta\) 高概率保证。

## C. 可信融合与多视图基线

所有对比共享 APK 模态、数据划分和可比训练预算，但按各自目标独立训练。

> **最终确认集要求**：当前方法迭代曾参考过现有 `labels/test.csv` 的汇总
> 结果，因此该集合只能作为 development test。正式论文必须另行锁定一份
> 从未查看的 confirmatory test，或使用外部数据集作最终确认；程序没有把
> test 用于反向传播并不能消除研究者已经依据 test 调整方法造成的适配。

- `tmc`：**TMC-style adapted**，保留逐视图/融合 Dirichlet 目标和
  Dempster-Shafer 组合。
- `ecml`：**ECML-style adapted**，保留平均 evidence 聚合、逐视图/融合
  evidence 目标与冲突一致性正则。
- `qmf_energy`：**QMF-Energy component baseline**，只实现 detached
  energy-weighted late fusion，不代表完整 QMF。
- `ours`：当前 competence-anchored 主方法。

固定融合规则对照包括 `dempster_rule_only`、
`cumulative_subjective_logic`、`log_pool` 与
`conflict_weighted_opinion`。它们是完整融合规则对比，不是 I2 原子消融。

```bash
python paper/run_trusted_fusion_baselines.py --method all
```

```bash
python paper/run_trusted_fusion_baselines.py --method all --dry-run
```

## D. 论文写作口径

- 使用 “Drebin-style”“MalDozer-inspired”“MaMaDroid-inspired”。
- 使用 “TMC-style adapted”“ECML-style adapted”，不得写成原论文严格
  复现。
- `qmf_energy` 只写作 “QMF-Energy component baseline”。
- 主方法的 I1 输出统一称为“预测competence”或“TCP competence”。
- 自然困难子集统一使用 `competence_imbalance`，并明确它依赖 I1 输出，
  只能作为 I2 下游压力诊断，不能循环证明 I1 正确。
- I3 统一写作“基于 `malware_fn_probability_anchor` 的expected CRC”。
