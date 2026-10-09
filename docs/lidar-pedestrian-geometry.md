# 行人框几何：匹配训练预算的开发实验

本轮只训练行人的位置、高度、尺寸、方向回归分支。增加同样训练步数的原 L1 对照，
分别检验额外训练和新损失的效果；不使用上一轮分数校准器。新损失未优于匹配预算
的原 L1，严格行人召回和 IDF1 均退化，**不采用为默认或实车方案**。
完整数值及失败见 [结果 JSON](lidar-pedestrian-geometry-results.json)。

## 方法与数据隔离

沿用 [整序列三角色分组](lidar-sequence-calibration.md)。三折检测头分别只在 A/B/C
训练组上继续训练；外层评价组分别为 C/A/B，共 17 条完整序列。中间校准组本轮
不用来拟合、调参或挑选模型。这些开发序列已在上一轮查看，不能声称新的独立测试；
0020 未打开，0010/0015/0019 本轮也不使用。

每折以同一个已冻结检测头开始，两种方案均为固定 6 轮，每序列每轮 128 次采样、
batch 8、AdamW 5e-5、权重衰减 0.01、梯度裁剪 10、回归权重 0.25、种子 17。
训练样本/批次顺序、初始化及骨干特征相同，不早停，不选最好轮次/序列/折。
参数和父模型哈希见 [固定配置](../configs/lidar_pedestrian_geometry.experimental.json)。

- `continued_l1`：保留原始八个回归字段绝对误差之和，仅继续训练行人几何。
- `weighted_axis_huber`：前六字段使用 beta=0.1 的 SmoothL1，XY 偏移权重 2、
  高度中心权重 1、长宽高对数尺寸权重 2；方向对正/负目标向量取较小损失，
权重 0.5，并加 0.1 的单位范数约束。模 π 的轴向等价不代表识别行人朝向。

这是固定的一组加权/稳健/轴向损失组合，本轮不能将整体效果进一步归因到某一个
子项，也不根据评价数据再挑选权重。模型清单的原 `parameters` 描述父模型 12 轮，
新增 `geometry_training_parameters` 描述本轮 6 轮，两者共同记录训练履历。

两种损失用有效行人目标数归一化。SmoothL1 的定义可参照
[PyTorch 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)，
实际环境为 PyTorch 2.7.1+cu126，行为由本机测试校验。共享骨干、车辆全部分支、
行人分类/速度及 BatchNorm 运行统计不更新；导出前逐张量比较，任何越界改动直接
失败。分类张量冻结不保证保留的框完全相同，几何仍会影响 ROI、圆形 NMS 和关联。

六个检测头全部冻结后，先完成无标签全序列推理，再评分。新几何不能复用原有
校准器的质量目标/模型绑定，所以四项方案都用原始分数和相同 0.7/0.1 跟踪配置，
保留原两次高分确认、4s 身份记忆、位置/尺寸/歧义恢复门控。

## 评价

对照包含原检测头、上一轮分折原分数、继续训练原 L1、新损失。报告每序列/类别/
折及合并 P/R/IDF1/ID 切换，分别使用前向 ±40°、50m、Car/Pedestrian 范围的
IoU≥0.25 和 IoU≥0.5；不属于 KITTI 官方评分。几何诊断允许单个框覆盖多个 GT，
只说明最佳框覆盖及误差，不能当作一对一召回。后方显示检测而不评分。

3s 全检测中断使用上一轮原检测头固定的 23 个可评目标与相同中断窗口，保留没有
预轨迹/返回、错误恢复他人身份等失败。它不能替代自然遮挡。候选必须同时满足
每序列/支持类别 P≥96%、R≥85%，且不增加身份切换或降低恢复；汇总提升不能豁免
局部失败。这里不训练最终全量部署模型，也不合并到 main。

## 完整结果与决定

三折共 17 条完整开发序列，5442 帧，其中 5438 帧有原始扫描，缺失帧与上一轮一致。
自然序列合并计数如下；严格列为 IoU≥0.5，其他列为 IoU≥0.25。不能用合并平均
掩盖逐序列和类别失败，也不能将开发数据重复评价称作最终泛化验收。

| 固定方案 | 整体精度 | 整体召回 | 整体 IDF1 | ID 切换 | 行人精度/召回 | 行人严格召回 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原检测头 | 74.81% | 67.89% | 68.79% | 66 | 42.75% / 58.70% | 29.47% |
| 父分折原分数 | 92.65% | 69.52% | 76.65% | 67 | 79.62% / 42.58% | 19.30% |
| 匹配预算继续 L1 | 92.64% | 69.52% | 75.90% | 69 | 79.55% / 42.58% | 19.37% |
| 匹配预算加权轴向损失 | 92.57% | 69.49% | 75.83% | 68 | 79.17% / 42.43% | 18.99% |

行人原始最佳框的严格覆盖为：父模型 1817/4523（40.17%），继续 L1 为
1894/4523（41.87%），新损失为 1832/4523（40.50%）。前者允许重复用框，
且包含 raw score≥0.1 提案，不能替代确认轨迹严格召回 19.37%/18.99%。
仅继续训练的原 L1 改善了部分原始几何，却没有改善完整跟踪；新损失也没有显示
跨序列收益。0016 新损失行人 IDF1 从父模型 55.13% 降至 47.33%，行人 ID 切换
43→44；0013 整体召回仍只有 13.41%。这些退化全部保留，未挑选最好的一折。
候选共记录 43 项逐序列/类别/身份或恢复验收失败，未满足既定部署标准。

固定 23 个可评恢复目标中，原检测头同 ID 恢复 11/23；父分折、新 L1 对照及新损失
均为 9/23，错误保留他人身份均为 0。本轮未恢复到原检测头水平，不能满足 3s
遮挡后可靠保留同一 ID 的要求，也不能因零错误恢复就忽略未恢复的目标。

逐张量冻结审计覆盖六个模型，155211 个车辆原始检测及 1082432 个范围外原始
检测数值精确一致。该检查不证明后方正确率，也不要求全局整数 ID 编号相同。
同一 0009 帧 100–199 独占 RTX 4090 的新损失候选测量为中位数 74.67ms、P99
86.82ms，100 帧无超过 100ms；仍不含 ROS、采集、显示和通信，不是目标板卡或
实车长时间稳定性验收。

本轮保留原默认，发布方法、复现脚本和完整负结果。下一步应优先检查行人高分
提案的漏检/排序与跨数据源适配，增加有代表性的训练和外部验证数据；继续放宽
ID 确认门槛或按某一序列缩放框尺寸，现有结果不支持。

## 复现与展示

先按上一轮报告生成分折训练特征及父模型，在新输出目录冻结本轮协议。

```powershell
$data = 'C:/Projects/hx/Datasets/kitti'
$parent = 'outputs/kitti_sequence_calibration_grouped'
$run = 'outputs/kitti_pedestrian_geometry'
$checkpoint = 'outputs/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
$audit = 'outputs/kitti_training_coverage/split_input_audit.json'
$plan = 'configs/lidar_sequence_calibration.experimental.json'
conda run --no-capture-output -n radar-id-tracker python examples/prepare_geometry_experiment.py --config configs/lidar_pedestrian_geometry.experimental.json --parent-run $parent --run-root $run
foreach ($i in 0..2) {
  foreach ($variant in 'continued_l1','weighted_axis_huber') {
    conda run --no-capture-output --prefix outputs/envs/lidar-detector python examples/train_pedestrian_geometry.py --features-dir "$parent/fold_$i/features" --parent-model "$parent/fold_$i/model/model.json" --checkpoint $checkpoint --output-dir "$run/fold_$i/$variant/model" --variant $variant --experiment-protocol "$run/protocol.json"
    if ($LASTEXITCODE -ne 0) { throw 'Geometry training failed' }
  }
}
conda run --no-capture-output --prefix outputs/envs/lidar-detector python examples/infer_geometry_folds.py --dataset-root $data --checkpoint $checkpoint --parent-run $parent --run-root $run --input-audit $audit
conda run --no-capture-output --prefix outputs/envs/lidar-detector python examples/profile_geometry_folds.py --dataset-root $data --checkpoint $checkpoint --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/evaluate_geometry_folds.py --dataset-root $data --parent-run $parent --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/diagnose_calibration_geometry.py --dataset-root $data --plan $plan --run-root $run --geometry --parent-run $parent
conda run --no-capture-output -n radar-id-tracker python examples/audit_geometry_outputs.py --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/review_geometry_folds.py --parent-run $parent --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/summarize_geometry_folds.py --run-root $run --output docs/lidar-pedestrian-geometry-results.json
```

Windows 原始数据只读，5438 帧点云资产经父实验验证后用硬链接复用；输出和权重
在 Git 忽略目录，不上传原始数据或模型。`review.html` 为目录页，17 条序列的自然/
中断共 34 页，每页比较匹配预算的两个几何方案，保留 TP/FP/MISS、临时候选、
原始点云与稳定 ID。静态审计覆盖全部帧和资产，不声称浏览器渲染验证完成。

干净暂存 wheel 共 120 项测试：113 通过、7 项可选依赖跳过；GPU 环境的 30 项
相关测试全部通过，包括 4 项新损失测试。核心 CLI 不依赖 PyTorch，许可证打包
验证通过；其他未提交 RadarScenes 修改未进入发布。
固定 0016/0017 帧 120 的两张 PNG 已进行视觉检查，包含真实原始点云、3D 框、
稳定 ID、TP/FP/MISS 和后方未评分提示。
