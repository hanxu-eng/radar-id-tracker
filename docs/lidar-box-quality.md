# 点云框质量评分与跨序列验证

本轮增加可选的当前点云框质量评分器，三条新序列的车辆召回改善，但**整体部署验收仍未通过**。
默认配置保持不变。实验分支为 `codex/lidar-box-quality`，基于上一轮
`codex/lidar-input-validation`，通过独立草稿 PR 审核。

## 独立验证结果

两个版本使用相同 CenterPoint 权重、64 倍反射率输入、5 帧历史和 0.7/0.1 跟踪门限。
左侧为原检测分数，右侧只更换 Car/Pedestrian 的质量分数。指标为实际观察轨迹，
不包含仅预测输出。按当前雷达坐标前方 ±40°、50m 内框中心评分，3D IoU ≥ 0.25。
这不是 KITTI 官方跟踪指标，没有实现官方难度划分和 2D DontCare 评测。

| 序列 | 原分数精确率 / 召回率 | 质量评分精确率 / 召回率 | 原分数 / 质量评分 ID 切换 |
| --- | --- | --- | --- |
| 0003 | 94.09% / 68.71% | 93.17% / 93.17% | 0 / 0 |
| 0004 | 92.69% / 73.96% | 97.03% / 84.59% | 0 / 0 |
| 0008 | 87.05% / 81.82% | 92.88% / 94.83% | 0 / 1 |

这三条序列均未用于模型训练、超参数或门限选择。模型先冻结并记录 SHA256，再做验证。
没有任何一条同时达到总体精确率 96% / 召回率 85% 的现有门槛；0008 的 ID 切换还退化。
0004 的车辆 P/R 为 97.90%/92.23%，但行人 64 次有效 GT 全部漏检，还有 6 次未匹配输出。
0003、0008 范围内没有有效行人 GT，不能据其总体成绩证明行人泛化；0008 仍有 8 次行人
未匹配输出。这些是上述局部规则下的诊断，不能替代官方忽略区域处理或逐框人工确认。

完整机器可读证据见 [结果 JSON](lidar-box-quality-results.json)，包含所有自然与 3s
中断回放、分车型成绩、IDF1、固定恢复分母、模型校验和及延迟记录。

## 泛化检查与失败保留

已经查看过的 0000/0001/0002/0005/0006 全部归入训练开发集，每次用其中四条训练、
留出第五条整段回放。下表每条序列都使用未见该序列标签的模型，**没有使用全量模型对
自身训练序列打分**。各序列的样本总权重相等，没有随机拆分相邻帧。

| 留出序列 | 原分数 P / R | 留一质量评分 P / R | 原分数 / 质量评分 ID 切换 |
| --- | --- | --- | --- |
| 0000 | 70.09% / 95.98% | **56.59% / 94.78%** | 0 / 0 |
| 0001 | 90.74% / 71.99% | 88.40% / 86.60% | 2 / 0 |
| 0002 | 87.99% / 51.68% | 99.43% / 61.90% | 0 / 0 |
| 0005 | 98.55% / 50.68% | 98.22% / 68.62% | 0 / 0 |
| 0006 | 38.96% / 98.36% | 98.82% / 98.36% | 1 / 0 |

0006 的行人未匹配输出从 627 次降到 0，但该序列没有有效行人 GT，行人召回为未定义。
0000 的精确率下降保留在结果中；0001 行人召回从 31.78% 降到 18.69%；0002 行人轨迹
召回仍为 0/167。评分器无法产生缺失的检测框：0002 的原始检测中仅有 37 个行人框
能达到同类 GT 的 0.25 IoU，且这只是重叠数量，不是保证可得到的轨迹召回。

这些检查降低单序列调参风险，尚不足以证明速腾雷达上的泛化。0003/0004/0008 从本轮起
也已被查看，后续调参不能再次将它们称为未见的最终测试集。下一阶段应保留新的整段
序列，并改善车辆/行人的检测框召回与分类质量，避免继续按失败序列修补门限。

## 模型与在线接入

使用 [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
分别训练 Car/Pedestrian 小树：60 轮、每树最多 7 叶、深度 3、最小叶样本 40、
学习率 0.1、L2=1、固定随机种子 17，不早停、不搜索超参数、不调跟踪门限。
训练使用 scikit-learn 1.9.0；导出模型为约 71 KB 的有界 JSON 树。
**在线运行只需要 NumPy，不需要 scikit-learn，也不加载 pickle。**

21 个特征来自当前原始点云与检测框：原分数、距离、相对高度、尺寸、框内及邻域点数、
归一化点分布、上下区域比例和网格占用。框轴均值取绝对值，避免 yaw 相差 π 时符号
改变。没有反射率、绝对世界位置、yaw 数值、序列/帧号、GT ID、未来帧或 GT 框输入。
当前点云不截断采样；稀疏空间索引只加速框查询。检测器仍使用因果的 5 帧融合，质量
特征只看当前一帧。两者都使用实际自车位姿，缺帧不会重复使用过期检测。

训练标签在独立离线步骤生成，只取前向有效同类 GT；IoU≥0.25 为正，0.1–0.25 重叠
及有效忽略类别附近的框排除，投影到 DontCare/无有效 ID 区域的负样本也排除。该训练
排除措施不等于实现了官方 DontCare 评测。正样本共车辆 4,141、行人 171，行人覆盖不足。
其他类别分数和所有类别框几何保持原样，不修改神经检测权重、不新增框。

`--quality-model` 必须显式传入；其权重校验和、扫描数、周期、强度倍率和检测输出门限
必须与训练域一致，错配会报错，不会静默切换默认输入。原分数和质量分数同时保存以便
追溯。质量分数没有经过概率校准，不能解释为真实目标概率。

导出树在 145,218 个开发集特征上的计算与训练库相差最多 1.11e-16。另以新序列 0003
完整 144 帧直接从原始点云推理，和独立缓存特征回放比较：33,767 个检测的几何、类别、
分数最大差异为 0。模型 SHA256 为
`59d86914d14d476bf4d2e7bc3650152bf08e7d9cad7885d967f843d71974e544`。
冻结后补充的输入校验、输出防覆盖和延迟描述修正没有改变训练参数或数值计算；冻结
协议保留训练当时的源文件哈希，在线缓存另记录实际运行源文件哈希。

## 3s 恢复、性能与可视化

每条序列在第 50–79 帧人工隐藏所有检测。以原分数版本可恢复的 GT 为固定分母，8 条
序列只有 3 个有效目标（0003/0004/0005 各 1），两侧均恢复 3/3；其他序列没有有效
机会，不能写成 100%。这不是自然遮挡，样本数量也不足以验收真实 3s 同 ID 恢复。

RTX 4090、Torch 2.7.1+cu126 上，独立测量 0004 第 100–199 帧，预热 5 帧：全管线
中位 **88.63 ms**，P99 **104.09 ms**，**12/100 帧超过 100 ms**。包括读盘、源文件
校验、自车补偿、体素化、神经检测、质量评分、框转换和跟踪，不包含采集、ROS 通信、
序列化或显示。质量评分中位 21.83 ms，已包含在框转换/跟踪阶段内，不能重复相加。
即使平均速度够用，尾延迟仍未满足稳定 10 Hz 的要求。

本机 `outputs/kitti_box_quality/review.html` 包含 8 条序列、自然和中断回放、两侧对照、
稳定 ID 与原始点云叠加，共 2,245 个真实点云资产；缺少的 4 帧显式标记。仅显示坐标
量化到厘米，检测使用原始浮点数。科学 PNG `preview_0003.png` 与 `preview_0004.png`
标注 TP/FP/漏检/忽略框；顶部数字为整个序列成绩。PNG 已做视觉检查，HTML 完成 JSON
及资产静态检查，未做浏览器渲染验证。后方框仅供检查，KITTI 无后方真值；当前质量
模型仅接受前向标签训练，后方及速腾数据均未验证，不能据画面宣布后方检测达标。

本轮暂存文件与基线构建的干净 wheel 完成 98 项测试：97 项通过、1 项因核心环境没有
Numba 而跳过；独立推理环境的质量评分、输入、几何与 JIT 测试 19 项全部通过。
无 PyTorch 的 CLI 帮助、许可证打包及稀疏查询对暴力统计的边界检查也通过。原有未
提交的 RadarScenes 修改未加入快照或发布。

## 复现

沿用 [独立 GPU 环境与固定 CenterPoint 权重](lidar-surround.md)。训练还需独立的
Python 环境和 scikit-learn==1.9.0；本机已有 `base` 环境满足要求。训练依赖不加入
核心/推理环境。以下从仓库根目录使用新的输出目录，数据只读，权重和训练模型都不
上传 Git。训练命令拒绝覆盖已有冻结协议。

```powershell
$model = 'outputs/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
$envPath = 'outputs/envs/lidar-detector'
$run = 'outputs/kitti_box_quality_reproduce'
foreach ($seq in @('0000','0001','0002','0005','0006')) {
    conda run --no-capture-output --prefix $envPath python -m radar_id_tracker.lidar_raw --dataset-root C:/Projects/hx/Datasets/kitti --sequence $seq --checkpoint $model --inference-config configs/lidar_centerpoint_kitti_input.experimental.json --output-dir "$run/raw_cache/$seq"
    conda run --no-capture-output -n radar-id-tracker python examples/prepare_box_quality.py --dataset-root C:/Projects/hx/Datasets/kitti --cache-dir "$run/raw_cache/$seq" --output-dir "$run/features/$seq" --training-targets
}
conda run --no-capture-output -n base python examples/train_box_quality.py --features-root "$run/features" --output-dir "$run/models"
foreach ($seq in @('0003','0004','0008')) {
    conda run --no-capture-output --prefix $envPath python -m radar_id_tracker.lidar_raw --dataset-root C:/Projects/hx/Datasets/kitti --sequence $seq --checkpoint $model --inference-config configs/lidar_centerpoint_kitti_input.experimental.json --output-dir "$run/raw_cache/$seq"
}
conda run --no-capture-output -n radar-id-tracker python examples/benchmark_box_quality.py --dataset-root C:/Projects/hx/Datasets/kitti --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/review_surround.py --dataset-root C:/Projects/hx/Datasets/kitti --run-root $run --comparison quality
```

在线单序列（新的输出目录）可增加：

```powershell
conda run --no-capture-output --prefix $envPath python -m radar_id_tracker.lidar_raw --dataset-root C:/Projects/hx/Datasets/kitti --sequence 0003 --checkpoint $model --inference-config configs/lidar_centerpoint_kitti_input.experimental.json --quality-model "$run/models/full/model.json" --output-dir "$run/online_0003" --track-output-dir "$run/online_tracks_0003" --tracker-config configs/lidar_centerpoint.experimental.json
conda run --no-capture-output --prefix $envPath python examples/profile_surround.py --dataset-root C:/Projects/hx/Datasets/kitti --checkpoint $model --config "$run/models/protocol.json" --sequence 0004 --quality-model "$run/models/full/model.json" --output "$run/latency.json"
```

两侧未评分的原始框也完整保留，用于定位检测器自身的召回上限；不要将重评分后的
高分框召回与全部输入框召回混淆。下一轮优先改善行人候选框和训练域覆盖，再验证
身份稳定与尾延迟，最终仍需要速腾实机录制及后方标注。
