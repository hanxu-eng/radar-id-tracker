# 检测漏检归因与检测头微调实验

本轮完成漏检分层诊断、前向车辆/行人检测头微调，以及三条新序列的冻结验证。
**候选模型泛化失败，不采纳为默认方案，仍不能上实车。** 默认推理输入和跟踪配置均未
修改。实验分支为 `codex/lidar-head-adaptation`，基于 `codex/lidar-box-quality`，通过
独立草稿 PR 发布。完整证据见 [结果 JSON](lidar-head-adaptation-results.json)。

## 新序列结果

训练前固定开发序列 0000/0001/0002/0003/0004/0005/0006/0008，模型冻结后才评分
0009/0013/0017。没有随机拆分相邻帧，没有根据验证结果选择门限、训练轮次或模型。
八条开发序列的结果只属于训练回放诊断，**不是留出或交叉验证成绩**。

三个版本使用同一基础 CenterPoint 权重、因果 5 帧融合、64 倍强度输入及 0.7/0.1 跟踪
门限。原检测头为基线；上一轮质量评分使用已经冻结的全量模型，不重新训练；本轮
仅替换前向 Car/Pedestrian 检测头，不叠加旧质量评分器。指标为实际观察轨迹的 P/R，
前方 ±40°、50m 内 Car/Pedestrian，3D IoU ≥ 0.25；预测保活输出不计入观察召回。

| 新验证序列 | 原检测头 P / R | 上轮质量评分 P / R | 本轮微调 P / R | 原检测头 / 微调 ID 切换 |
| --- | --- | --- | --- | --- |
| 0009（803 帧） | 91.75% / 62.44% | 94.42% / 82.99% | 95.33% / 61.29% | 3 / 1 |
| 0013（340 帧） | 41.55% / 36.80% | 82.48% / 26.92% | **97.56% / 4.16%** | 1 / 0 |
| 0017（145 帧） | 24.28% / 56.78% | 54.76% / 23.53% | **81.25% / 16.62%** | 14 / 5 |

没有任何新序列同时达到精确率 96%、召回率 85% 的既有门槛。ID 切换减少同时伴随大量
漏检，不能单独解释为身份稳定性改善。0013 行人召回从 327/914（35.78%）降至 19/914
（2.08%）；0017 全部有效 GT 为行人，召回从 444/782 降至 130/782。0009 仅有一个行人
实例、28 次有效标注，两侧行人召回均为 0，车辆占优不能掩盖行人失败。

补充更严格的 3D IoU ≥ 0.5 评分，原检测头 / 微调总体 P/R 分别为：

| 序列 | 原检测头 P / R | 微调 P / R |
| --- | --- | --- |
| 0009 | 86.27% / 58.86% | 93.70% / 60.49% |
| 0013 | 23.59% / 20.89% | 87.80% / 3.74% |
| 0017 | 2.46% / 5.75% | 34.38% / 7.03% |

IoU 0.5 检查不覆盖原 IoU 0.25 结果，也没有用来重新选模型。这些都是局部诊断，
**不是 KITTI 官方跟踪指标**：未实现官方难度划分、2D DontCare 评测和后方真值。

## 漏检归因

逐个前向 GT 检查原始候选框、质量评分后继续/新建门限和实际观察输出，另外按距离、
框内当前点数和标注遮挡程度分组。GT、实例 ID 和这些诊断从不进入在线推理或跟踪。

| 已查看序列的行人 | 有效 GT 次数 | 原始同类框可覆盖 GT | 主要损失 |
| --- | --- | --- | --- |
| 0001 | 107 | 105 | 57 次低于质量继续门限，27 次低于新建门限，实际观察 20 次 |
| 0002 | 167 | 35 | 121 次无可用几何框、11 次仅其他类别框可覆盖；剩余 35 次低于继续门限 |
| 0004 | 64 | 16 | 48 次无可用原始框；11 次低于继续门限、4 次低于新建门限、1 次确认/关联损失 |

“可覆盖”指每个 GT 的最佳同类候选 3D IoU ≥ 0.25，允许多个 GT 使用同一个候选。
它不是一对一检测召回，也不保证最终跟踪可达到该数值。上一轮的 37 个匹配行人候选
框与这里的 35 个可覆盖 GT 是不同分母，候选中存在重复框。

0002 的 167 个 GT 中，149 个框内当前点数不少于 20，标注遮挡等级均为 0；原框主要
缺少附近同类中心或分类错误。0004 的 48 次原框失败中，14 次主要为垂直位置/高度，
18 次为中心/尺寸/朝向，16 次无附近同类中心。分层诊断说明，继续修改分数筛选不能
独立解决所有问题。细节与距离/点数/遮挡分组保存在结果 JSON。

## 微调方法与数据覆盖

冻结 PFN、SECOND、颈部及共享卷积，只训练克隆的车辆和行人任务头。BatchNorm 统计
冻结，速度分支冻结。监督为中心热力图、平面中心偏移、重力中心高度、对数尺寸和
sin/cos 朝向；中心高斯半径固定为 2。这是本仓库的有限适配实验，不是完整上游训练流程。
[CenterPoint 检测头参考](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/dense_heads/centerpoint_head.py)。

固定 12 轮，每条训练序列每轮等量有放回采样 128 帧，batch=8，AdamW 学习率 1e-4、
weight decay=0.01，回归损失权重 0.25，梯度裁剪 10，随机种子 17；不早停、不搜索参数、
不做数据增强、不按验证成绩选择轮次。训练损失由 2.964 降至 0.756，但损失下降不能
证明检测泛化。学习得到的置信度未经概率校准。

缓存 1,274 帧、float16 的共享特征。训练帧选取每隔一帧，并保留每个有前向行人的帧；
**帧选取使用离线标签，特征值仅来自点云**。目标共车辆 3,359 框、行人 354 框，但只
覆盖 204 个独立车辆、**11 个独立行人**。实例 ID 仅用于事后统计覆盖，不进入监督
张量或模型输入。少量行人的重复观测并没有提供足够的独立场景覆盖；新序列失败与
这一风险一致，但本轮无法单独确定是数据覆盖、冻结骨干还是训练方式造成。

前向有效忽略类别附近和保守投影到 DontCare/无有效 ID 矩形的区域不作负样本，随后
恢复有效正样本中心。该训练排除策略不等于实现官方忽略区域评测。因果点云融合只
使用当前/过去帧与 OXTS 自车补偿，不使用未来点云。

在线必须显式传入 `--head-model`。manifest 限制基础权重、输入域、float16 特征精度、
训练/验证序列分离与本地权重 SHA256；学习权重通过 `weights_only=True` 严格加载，
检查有限值，不接受旧质量评分与新头混用。没有参数时保持基础检测路径。
所有未适配类别以及前向范围外车辆/行人的原始检测均保留；11 条序列核对
**711,149 个检测实例的完整几何、类别和分数完全一致**。这项检查证明输出保留，
不证明后方检测精度；跟踪身份仍可能因前向输入变化而变化。

本机权重仅约 1.86 MB，保存在忽略目录，不上传数据、特征或学习权重。模型 manifest
SHA256 为 `996fe092f427b12483473fec981a56360a92ec192ff1a6aad06ac6736e510586`，
权重 SHA256 为 `5276c35d738851e0cd448c57181df64c4e9a8152d47517b91bd9ff6778be4cae`。
冻结后补充的 manifest 校验、CLI、诊断/显示和源哈希记录不改变训练数值、模型参数或
推理数学；冻结协议保留运行当时的源哈希。原缓存中 `features_label_free` 仅描述特征
值，新版缓存元数据明确拆分特征值与使用标签的帧选取。

## 3s 恢复、性能和查看结果

每条序列第 50–79 帧隐藏所有检测，持续 3 秒。这是人工全检测中断，不是自然遮挡。
以原检测头已有轨迹且恢复帧有可匹配输入的 GT 为固定分母：新序列 0009 双方 1/1，
0013 没有有效机会，0017 原检测头 2/4、微调 **0/4**。微调缺少中断前轨迹/返回检测也
计入失败，不缩小分母。全部 11 条序列共 8 次机会，原检测头 6/8、微调 4/8；样本少且
候选退化，不能宣布满足实车遮挡 3s 保持 ID 的要求。

RTX 4090、Torch 2.7.1+cu126，单独测量 0009 第 100–199 帧，预热 5 帧：端到端
中位 **88.39 ms**，P99 **102.74 ms**，**4/100 帧超过 100 ms**。包括读盘/扫描哈希、
自车补偿、体素化、原始与适配头检测/解码、坐标转换和跟踪；不包括采集、ROS、输出
序列化或显示。该序列不是上一轮延迟序列，不能直接据两组数字宣称提速。

本机 `outputs/kitti_head_adaptation/review.html` 提供 11 条序列的自然/中断双侧回放，
含原始点云、稳定 ID、候选框和离线真值，3,533 个真实点云资产；0001 缺少 177–180
四帧，显式标记并从评分中排除。共 22 个对照案例、14,148 个双侧回放帧。
HTML 完成 JSON/资产静态检查，未做浏览器渲染验证。

本轮暂存源码与基线构建的干净 wheel 完成 105 项测试：102 项通过，3 项因核心环境
没有 PyTorch/Numba 而跳过；独立推理环境的几何、JIT、原始输入、质量评分及新头
测试 26 项全部通过。无 PyTorch 的 CLI 帮助和许可证打包通过。未提交的 RadarScenes
修改不进入安装包快照或本轮发布。

`failure_0013.png` 标注 TP/FP/MISS，展示冻结验证 0013 自然回放第 76 帧：该帧是
微调相对基线额外漏检数最多时的最早帧，不是平均表现样本。候选漏检 5 个，基线
漏检 1 个；顶部 P/R 为整个序列成绩。PNG 已作视觉检查。

## 复现和下一步

沿用 [固定权重与独立 GPU 环境](lidar-surround.md)。训练使用已有 PyTorch 推理环境；
离线评分/审计使用核心 NumPy 环境，科学 PNG 使用含 matplotlib 的环境。以下命令从
仓库根目录运行；首次复现使用新的输出目录，缓存或冻结模型存在时不要覆盖：

```powershell
$dataset = 'C:/Projects/hx/Datasets/kitti'
$checkpoint = 'outputs/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
$run = 'outputs/kitti_head_reproduction'
$gpu = 'outputs/envs/lidar-detector'
conda run --no-capture-output -n radar-id-tracker python examples/diagnose_lidar_proposals.py --dataset-root $dataset --quality-run outputs/kitti_box_quality --output-dir outputs/kitti_proposal_diagnostics_reproduction
conda run --no-capture-output --prefix $gpu python examples/cache_lidar_head_features.py --dataset-root $dataset --checkpoint $checkpoint --output-dir "$run/features"
conda run --no-capture-output --prefix $gpu python examples/train_lidar_heads.py --features-dir "$run/features" --checkpoint $checkpoint --output-dir "$run/model"
conda run --no-capture-output --prefix $gpu python examples/benchmark_lidar_heads.py --dataset-root $dataset --checkpoint $checkpoint --run-root $run --previous-quality-run outputs/kitti_box_quality
conda run --no-capture-output -n radar-id-tracker python examples/compare_previous_quality.py --dataset-root $dataset --run-root $run --quality-model outputs/kitti_box_quality/models/full/model.json
conda run --no-capture-output -n radar-id-tracker python examples/score_lidar_heads_strict.py --dataset-root $dataset --run-root $run
conda run --no-capture-output -n radar-id-tracker python examples/review_surround.py --dataset-root $dataset --run-root $run --comparison heads
conda run --no-capture-output -n radar-id-tracker python examples/audit_lidar_heads.py --dataset-root $dataset --run-root $run
conda run --no-capture-output --prefix $gpu python examples/profile_surround.py --dataset-root $dataset --checkpoint $checkpoint --config "$run/evaluation_protocol.json" --sequence 0009 --head-model "$run/model/model.json" --output "$run/latency.json"
conda run --no-capture-output -n radar-id-tracker python examples/summarize_lidar_heads.py --run-root $run --diagnostic-root outputs/kitti_proposal_diagnostics_reproduction --output "$run/results.json"
```

上一轮特征 manifest 保存本机原检测缓存路径，跨机器复现时须重建上一轮缓存和 manifest。
完整回放、训练特征和点云资产较大，均保存在 `outputs/`。结果 JSON 包含所有训练回放
失败项，例如 0004 微调行人召回仍为 0；0002 行人召回升至 120/167 是训练内成绩，
不能当作独立改进证据。

下一轮应先扩大独立行人实例和场景覆盖，再验证标准训练/增强及骨干适配，保留新整段
序列进行冻结评估。检查公开预训练数据与跟踪序列的场景重叠后才能声称独立验证。
0009/0013/0017 本轮已被查看，后续训练和参数修改后不能再次称为未见最终测试。
默认方案不变，后续优先改善真实框召回和几何，再校准分数及 3s 恢复。
