# 扩大训练覆盖的检测头对照

本轮扩大训练中的行人及场景覆盖，使用同一冻结骨干检测头方法，对三条新序列进行
冻结验证。相比上一轮微调，部分行人召回明显改善，但**精度、几何和 3s 恢复仍未
通过验收，不采纳为默认模型，不能上实车**。默认推理和跟踪配置保持不变，实验分支为 `codex/lidar-training-coverage`，
基于 `codex/lidar-head-adaptation`，通过独立草稿 PR 审查。

## 冻结分割与训练

分割和参数在读取新验证检测标签前固定，见
[实验配置](../configs/lidar_training_coverage.experimental.json)。训练序列为
0000/0001/0002/0003/0004/0005/0006/0007/0008/0009/0011/0012/0013/0014/0016/0017/0018，
共 17 条。已经查看过的 0009/0013/0017 纳入开发训练，不能再次作为未见测试。
新冻结验证为 0010/0015/0019，0020 保留给后续；本轮不读取 0020 数据或标签。

模型冻结后统一执行原检测头、上一轮冻结微调模型和本轮扩充数据模型的推理，再读取
验证标签评分。未根据验证成绩选择模型、轮次或门限。四条训练诊断回放固定为
0002/0009/0013/0017，它们的成绩不是交叉验证结果。

训练与验证之间的原始扫描 SHA256 无完全相同的扫描。该检查不证明采集路线独立、
邻近场景独立或速腾雷达上的泛化。原始数据只读，缺少的 0001 第 177–180 帧仍显式
保留为缺扫描；这条序列仅参与训练，没有进入本轮七条结果回放。

补充使用 GPS 平面位置检查路线接近程度：三个验证序列没有任何帧位于训练自车
位置 20m 内，最近训练位置距离的序列中位数分别约 3,647m、123m、1,260m。检查
忽略朝向和采集时间，仅描述空间接近度，不能证明现实目标或道路条件完全独立。

缓存共 3,317 帧，车辆 8,895 框、行人 4,503 框；按序列与 GT ID 统计，标注车辆实例
418 个、标注行人实例 92 个，上轮分别为 204 和 11。实例 ID 仅用于离线覆盖统计，
不进入训练特征或目标；跨序列可能出现同一现实物体，不宣称这些实例都是不同的人。
八条旧训练序列的特征、标签和源扫描校验后复用，其余九条重新提取。监督帧仍每隔
一帧采样并保留所有前向行人帧；帧选取使用离线标签，特征值来自真实点云。

模型仍冻结 PFN、SECOND、颈部、共享卷积、BatchNorm 统计与速度分支，只训练车辆和
行人检测头。沿用 12 轮、每序列每轮 128 次有放回采样、batch 8、AdamW 1e-4、
weight decay 0.01、回归权重 0.25、裁剪 10、种子 17；不做增强、早停或参数搜索。
输入仍为 64 倍强度、因果 5 帧融合、0.1 检测输出门限，跟踪仍为 0.7/0.1。

**相同的是每序列训练方法，训练总量同时增加。** 优化步数由 1,536 增至 3,264，不能
仅据对照把变化归因为数据覆盖这一单一因素。固定最终第 12 轮，损失从 1.675 降至
0.811，训练损失不作为泛化证据。

manifest SHA256 为 `57c3f48dda1bb632fe19e324edb6d8acd1f22edc20a6b7fa6c3415452d63d012`，
学习权重 SHA256 为 `948deb91608db724f00c979e1890e8aa712e105fa74987273c8bfa0ddaa859f1`。
分割 SHA256 为 `879a6cbd5876d89825a87550103d1ecb3198e555dc2ec39532a49f219a8f8376`。
数据、特征和学习权重保存在忽略目录，不上传 Git。分割、模型及运行输入均有哈希
检查，不能静默混用已有缓存。旧质量评分模型没有与新检测头叠加。

## 结果与边界

完整机器可读证据见 [结果 JSON](lidar-training-coverage-results.json)。

指标沿用前向 ±40°、50m 内 Car/Pedestrian 的实际观察轨迹，3D IoU ≥ 0.25；预测
保活输出单独报告。另做 IoU ≥ 0.5 检查，不覆盖原评分文件。
这不是 KITTI 官方指标，没有官方难度、2D DontCare 评测或后方真值。
[KITTI 官方跟踪说明](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)。

| 新验证序列 | 原检测头 P / R | 上轮微调 P / R | 本轮 P / R | 原检测头 / 上轮 / 本轮 ID 切换 |
| --- | --- | --- | --- | --- |
| 0010（294 帧） | 99.24% / 77.01% | 100% / 78.19% | 100% / 79.37% | 0 / 0 / 0 |
| 0015（376 帧） | 37.47% / 70.11% | 68.35% / 29.81% | 61.16% / 68.67% | 17 / 0 / 7 |
| 0019（1,059 帧） | 44.39% / 70.56% | 80.07% / 26.00% | 66.01% / 66.97% | 85 / 6 / 41 |

三条序列均未同时达到 P≥96%、R≥85%。与上一轮微调相比，本轮召回恢复，但 0015、
0019 的精确率下降；与原检测头相比，精确率和 IDF1 改善，但两条序列的召回略降。
0015 的总体 IDF1 从原检测头 44.35% 提高到 64.46%，0019 从 42.48% 提高到 52.28%；
这些结果有改善，也仍有明显身份错误，不能只看 ID 切换减少。

| 序列 | 原检测头行人 P / R | 上轮行人 P / R | 本轮行人 P / R |
| --- | --- | --- | --- |
| 0010（29 次 GT） | 100% / 34.48% | 未定义 / 0% | **未定义 / 0%** |
| 0015（689 次 GT） | 34.77% / 60.96% | 46.99% / 5.66% | 64.70% / 58.78% |
| 0019（6,024 次 GT） | 40.37% / 67.65% | 72.13% / 16.45% | 62.31% / 63.50% |

0010 本轮车辆 P/R 为 100%/84.17%，但两个标注行人实例全部漏检。0019 本轮车辆
P/R 为 93.19%/91.55%，不能用车辆成绩代替行人验收。四条训练回放也保留失败：
0013 总体召回从上轮 4.16% 升至 25.26%，仍低于原检测头 36.80%；0017 从 16.62%
升至 66.11%，但这是本轮训练内成绩，不能当作泛化证明。

更严格的 3D IoU≥0.5 下，原检测头 / 本轮 P/R 分别为：

| 序列 | 原检测头 P / R | 本轮 P / R |
| --- | --- | --- |
| 0010 | 83.04% / 64.44% | 100% / 79.37% |
| 0015 | 25.25% / 47.25% | 43.59% / 48.94% |
| 0019 | 34.21% / 54.42% | 41.18% / 41.93% |

0019 的严格几何召回明显退化，因此不能把宽松 IoU 下的改善解释为框几何已达标。

## 原始框、分数与身份恢复

新增离线分层检查，每个 GT 独立选最佳同类、同前向范围内原始框。该检查允许重复
使用一个候选，**不是一对一召回**。0010 本轮行人原始框可覆盖 18/29 个 GT，且均
达到 IoU 0.5，但没有一次覆盖框达到 0.7 新建门限，最终观察 0/29。这里有明确的
分数/新建损失，仍不能直接降低门限，因为其他序列有大量虚警。

0019 行人原始框在 IoU 0.25 下覆盖由原检测头 5,110/6,024 升至 5,787/6,024；在
IoU 0.5 下却从 3,972 降至 3,182。达到高分新建门限的覆盖从 1,886 降至 1,443，
最终观察 3,825/6,024。该结果说明几何及分数可靠性都需要改善。诊断只在模型冻结后
执行，没有根据这些数字再次修改模型、门限或训练标签。

第 50–79 帧人工隐藏全部检测持续 3s，以原检测头已有轨迹且返回有可匹配输入的
GT 为固定分母。新验证序列 0010、0015 均为三模型 1/1；0019 原检测头 2/2、上轮
微调和本轮均 1/2。本轮合计 **3/4**，原检测头 **4/4**，其中一个本轮目标未返回
有效观察。样本少，而且是人工中断，不能验收自然遮挡 3s 保持原 ID。

## 耗时、可视化与验证

RTX 4090、Torch 2.7.1+cu126，独立测量 0010 第 100–199 帧，预热 5 帧：端到端
中位 **72.92 ms**、P99 **82.55 ms**、**0/100 帧超过 100 ms**。包含读盘/源哈希、
自车补偿、体素化、原始及适配检测头、框转换和跟踪；不包含采集、ROS、序列化或显示。
该单段样本的时间预算通过，不能证明长时间运行稳定 10 Hz，也不能与上轮不同序列
耗时直接比较为提速。

7 条序列核对 644,770 个前向适配范围外及其他类别原始检测，完整几何、类别和分数
完全保留。该检查不保证后方检测精度或两侧 ID 数字相同。
本机 `outputs/kitti_training_coverage/review.html` 包含 14 个自然/中断对照案例、
13,000 个双侧回放帧和 3,250 个原始点云资产。JSON/资产静态检查通过，未作浏览器
渲染验证。`preview_0015.png`、`preview_0019.png` 展示预设第 120 帧，标注实际
TP/FP/MISS 和训练/验证角色；顶部数字是整个序列成绩，PNG 已作视觉检查。

干净暂存快照 wheel 完成 109 项测试：106 项通过、3 项因没有可选 PyTorch/Numba
而跳过；推理环境 26 项相关测试全部通过。核心 CLI 在没有 PyTorch 时可显示帮助，
许可证打包通过。冻结后补充的缓存/来源校验和显示修正没有改变模型数值计算。
其他未提交的 RadarScenes 修改不进入本轮快照或发布。

下一轮应在开发数据内按整序列多折生成未见该折的检测结果，再做分数可靠性校准及
行人几何修复，避免对单条失败序列降低门限。0010/0015/0019 从本轮起属于已查看
数据；0020 继续保留，不能在训练和参数修改中反复查看。必要时扩展其他独立标注
数据，完整检测训练与实机采集评估仍待完成。

## 复现

沿用 [独立推理环境和固定基础权重](lidar-surround.md)。首次运行使用新的输出目录；
数据只读，冻结模型存在时不覆盖。从仓库根目录运行：

```powershell
$dataset = 'C:/Projects/hx/Datasets/kitti'
$checkpoint = 'outputs/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
$plan = 'configs/lidar_training_coverage.experimental.json'
$run = 'outputs/kitti_coverage_reproduction'
$previous = 'outputs/kitti_head_adaptation'
$gpu = 'outputs/envs/lidar-detector'
conda run --no-capture-output -n radar-id-tracker python examples/audit_training_split.py --dataset-root $dataset --split-plan $plan --output "$run/split_input_audit.json"
conda run --no-capture-output -n radar-id-tracker python examples/audit_split_geography.py --dataset-root $dataset --split-plan $plan --run-root $run
conda run --no-capture-output --prefix $gpu python examples/cache_lidar_head_features.py --dataset-root $dataset --checkpoint $checkpoint --output-dir "$run/features" --split-plan $plan --split-input-audit "$run/split_input_audit.json" --reuse-features "$previous/features"
conda run --no-capture-output -n radar-id-tracker python examples/audit_lidar_heads.py --dataset-root $dataset --run-root $run --training-only
conda run --no-capture-output --prefix $gpu python examples/train_lidar_heads.py --features-dir "$run/features" --checkpoint $checkpoint --output-dir "$run/model"
conda run --no-capture-output --prefix $gpu python examples/benchmark_training_coverage.py --dataset-root $dataset --checkpoint $checkpoint --split-plan $plan --previous-run $previous --run-root $run --phase infer
conda run --no-capture-output -n radar-id-tracker python examples/benchmark_training_coverage.py --dataset-root $dataset --checkpoint $checkpoint --split-plan $plan --previous-run $previous --run-root $run --phase evaluate
conda run --no-capture-output -n radar-id-tracker python examples/score_lidar_heads_strict.py --dataset-root $dataset --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/diagnose_head_outputs.py --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/review_surround.py --dataset-root $dataset --run-root $run --comparison coverage
conda run --no-capture-output -n radar-id-tracker python examples/audit_lidar_heads.py --dataset-root $dataset --run-root $run
conda run --no-capture-output --prefix $gpu python examples/profile_surround.py --dataset-root $dataset --checkpoint $checkpoint --config "$run/evaluation_protocol.json" --sequence 0010 --head-model "$run/model/model.json" --output "$run/latency.json"
conda run --no-capture-output -n radar-id-tracker python examples/summarize_training_coverage.py --run-root $run --split-plan $plan --previous-run $previous --output "$run/results.json"
```

已有特征 manifest 保存上轮本机缓存路径，跨机器须重建上轮结果。
`--reuse-features` 可省略以重新提取全部 17 条训练序列。结果回放有明确的训练/验证角色。
