# 整序列三角色校准与 ID 确认

本轮分别训练检测头、拟合分数校准、评价完整序列，三种角色没有序列重叠。
校准＋中分确认提高整体召回，但精度和 3s 恢复退化，**不采用为默认或实车配置**。
这是已经查看过的开发数据上的外层评价，不是新的最终测试。0020 保持封存，
0010/0015/0019 本轮也不参与。默认推理和跟踪配置保持不变。

## 固定协议

配置见 [实验分割](../configs/lidar_sequence_calibration.experimental.json)。
三组分别为：

- A：0000/0003/0007/0014/0016/0018。
- B：0001/0004/0008/0011/0012/0013。
- C：0002/0005/0006/0009/0017。

三折依次为 A 训练/B 校准/C 评价、B 训练/C 校准/A 评价、C 训练/A 校准/B 评价。
校准序列不训练检测头，评价序列不参与检测训练、校准拟合或策略选择。
在模型最终轮次冻结和所有无标签点云推理完成后，才拟合校准器；所有校准器和
策略冻结后才进行本轮外层评价。没有选择最好的一折来发布部署模型。

分组先做无标签路线审计，将邻近的 0011/0012 放入同组。修正后三种角色之间
没有重复扫描，也没有距离参考自车位置 20m 内的查询帧。这不证明现实目标、天气、
道路或速腾传感器分布独立。原始数据只读；0001 缺少的 177–180 帧明确排除评分，
仍保留空检测输入和时间间隔。

检测头沿用冻结骨干/共享卷积/BatchNorm 统计/速度分支，固定 12 轮、每序列每轮
128 次采样、batch 8、AdamW 1e-4、权重衰减 0.01、回归权重 0.25、种子 17。
复用经哈希校验的原始点云特征。每个模型只训练 5/6 条序列，不能与上一轮全量
17 条序列模型直接比较为分数校准的收益。

## 分数与确认

校准采用每类一个正斜率 sigmoid(logit(score))，固定 L2=0.01、100 次优化上限，
各有预测的序列总权重相同；不重新平衡正负类别比例。校准目标是本地前向范围内、
3D IoU≥0.25 的同类一对一匹配，重复框作为负样本；只有未匹配且覆盖有效
Van/Person_sitting 忽略框的输出排除。没有官方 2D DontCare/难度处理，因此该
概率只针对此开发混合和目标定义，不能视为物理存在概率或身份概率。

校准器使用与检测模型训练分离的数据，符合
[概率校准的数据隔离原则](https://scikit-learn.org/stable/modules/calibration.html)。
它不能改变检测排序、增加缺失框或修复几何；缺少任一正/负结果的类别显式回退为
原分数并记录状态。在线只使用类别与检测分数，NumPy 推理；GT、帧/序列 ID 和
标注 ID 不进入数值计算。模型绑定基础权重、具体检测头和完整输入配置，拒绝混用。

固定四项对照：

1. 原检测头、原 0.7/0.1 跟踪。
2. 当前分折检测头、原 0.7/0.1 跟踪。
3. 同一分折检测头、校准分数、两次高分确认。
4. 同一分折检测头、校准分数、中分 0.5 候选/高分 0.7，连续三次中分或两次高分确认。

中分证据仍受 0.4s 窗口、0.2s 最大间隔和位置/尺寸门控约束。原始分数至少 0.7
仍是旧 ID 恢复的必要条件：恢复门限取 max(0.7, 校准后的原始 0.7)，并在浮点
sigmoid 饱和时对低于原始门限的分数作严格下界区分。位置、尺寸、轴向角度、
双向最优和歧义拒绝规则不变。该约束允许变得更严格，不保证恢复数量不退化。
策略按类别作用于跟踪，范围外检测值保持一致，但不保证后方 ID 数字一致。

## 评价与结果

完整数值及失败见 [结果 JSON](lidar-sequence-calibration-results.json)。自然序列
同时报告每序列、每类别、分折和合并计数，不使用车辆成绩掩盖行人漏检。
本地范围仍为前向 ±40°、50m 内 Car/Pedestrian，实际观察轨迹的 IoU≥0.25；
另报 IoU≥0.5，不覆盖原评分。不是 KITTI 官方指标，后方没有真值。

17 条完整外层开发序列共 5442 帧，5438 帧有扫描。合并计数如下；这些数字不是
每序列验收通过率，各序列和类别的失败都保留在结果 JSON 中。

| 固定方案 | 精度 | 召回 | IDF1 | ID 切换 | 行人精度/召回 | 行人 IoU≥0.5 召回 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原检测头 | 74.81% | 67.89% | 68.79% | 66 | 42.75% / 58.70% | 29.47% |
| 分折检测头原分数 | 92.65% | 69.52% | 76.65% | 67 | 79.62% / 42.58% | 19.30% |
| 校准＋两次高分确认 | 87.79% | 73.40% | 78.18% | 36 | 66.84% / 28.39% | 11.17% |
| 校准＋中分确认 | 85.02% | 79.13% | 81.03% | 34 | 64.86% / 49.88% | 26.62% |

相同分折模型的校准＋中分确认减少 FN，但 FP 从 1102 增至 2786。0017 的整体
精度从 66.04% 降至 33.87%，0013 的整体召回仅 12.16%；不能由汇总 IDF1 提高
推出稳定部署。中分策略有 47 项逐序列/类别/恢复或身份验收失败记录，未满足每序列
及有真值类别精度 96%、召回 85% 的既定要求。

3s 压力测试隐藏全部检测 30 帧，以原检测头的可评 GT 固定分母。正常序列从
50 帧开始，78 帧长的 0012 在 43 帧开始，按无标签帧数预先计算并保留完整 5 帧
返回窗口。候选没有预轨迹、没有返回轨迹、错误复活他人 ID 都保留为失败。
这不是自然物理遮挡验收。

固定 23 个可评身份中，原检测头恢复同一 ID 为 11/23，分折原分数为 9/23，
校准高分及中分均为 7/23。高分方案还出现 1 次错误保留他人身份，中分方案为 0。
严格原分数恢复门槛阻止低分复活，但不能阻止高分错误关联，恢复要求仍未通过。

独占 RTX 4090 的 0009 帧 100–199 测量中，中分校准方案端到端中位数 74.01ms、
P99 84.07ms，100 帧无超过 100ms 的情况；校准阶段中位数 0.77ms。
包含读取/哈希、位姿补偿、检测/解码和跟踪，不含 ROS、采集、显示及通信，不能
代替目标实机的长时间负载测试。1082432 个范围外原始检测数值完全一致，1152484
个框经校准后只改变分数；后方正确率仍没有证据。

几何误差分解按每个 GT 的最佳同类原始框统计中心 XY/Z、长宽高对数比和模 π
轴向角差，分别报告 IoU≥0.25 覆盖和 0.25≤IoU<0.5 子集。允许同一框重复覆盖
GT，不能把这种覆盖率当成一对一召回。本轮不修改几何训练；下一轮使用匹配的
额外训练对照来单独验证几何损失，避免把更多训练步数当作损失改进。

## 复现与展示

模型、特征、缓存和点云在 Git 忽略目录 `outputs/kitti_sequence_calibration_grouped`，
权重不随源码上传。运行时选择新的输出目录，冻结文件不会静默覆盖：

```powershell
$data = 'C:/Projects/hx/Datasets/kitti'
$plan = 'configs/lidar_sequence_calibration.experimental.json'
$run = 'outputs/kitti_sequence_calibration_grouped'
$checkpoint = 'outputs/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
$audit = 'outputs/kitti_training_coverage/split_input_audit.json'
conda run --no-capture-output -n radar-id-tracker python examples/audit_calibration_split.py --dataset-root $data --input-audit $audit --plan $plan --output "$run/split_role_audit.json"
conda run --no-capture-output -n radar-id-tracker python examples/prepare_calibration_folds.py --plan $plan --features outputs/kitti_training_coverage/features --run-root $run
foreach ($i in 0..2) {
  conda run --no-capture-output --prefix outputs/envs/lidar-detector python examples/train_lidar_heads.py --features-dir "$run/fold_$i/features" --checkpoint $checkpoint --output-dir "$run/fold_$i/model"
}
conda run --no-capture-output --prefix outputs/envs/lidar-detector python examples/infer_calibration_folds.py --dataset-root $data --checkpoint $checkpoint --plan $plan --run-root $run --input-audit $audit
conda run --no-capture-output -n radar-id-tracker python examples/fit_calibration_folds.py --dataset-root $data --plan $plan --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/evaluate_calibration_folds.py --dataset-root $data --plan $plan --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/diagnose_calibration_geometry.py --dataset-root $data --plan $plan --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/audit_calibration_outputs.py --plan $plan --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/prepare_calibration_clouds.py --dataset-root $data --plan $plan --run-root $run --input-audit $audit
conda run --no-capture-output -n radar-id-tracker python examples/review_calibration_folds.py --dataset-root $data --plan $plan --run-root $run
conda run --no-capture-output --prefix outputs/envs/lidar-detector python examples/profile_calibration_folds.py --dataset-root $data --checkpoint $checkpoint --plan $plan --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/summarize_calibration_folds.py --plan $plan --run-root $run --output docs/lidar-sequence-calibration-results.json
```

`review.html` 比较原检测头和候选，`review_heads.html` 比较同一分折检测头的原分数
和校准/中分策略；都保留所有自然和中断序列、TP/FP/MISS、临时候选、原始点云与
后方未评分提示。HTML 只作静态 JSON/资产验证，不声称浏览器渲染已验证。

干净暂存 wheel 的 116 项测试为 113 通过、3 项可选依赖跳过，核心无 PyTorch 时
CLI 帮助正常、许可证打包正常。原有未提交 RadarScenes 改动不进入本轮发布。
可选 GPU 环境的 26 项关联测试通过，包括检测头目标、推理适配、坐标和 JIT。
核心测试包含 7 项校准/门槛/领域绑定测试。两张固定帧 PNG 已进行视觉检查；
HTML 未作浏览器检查。
