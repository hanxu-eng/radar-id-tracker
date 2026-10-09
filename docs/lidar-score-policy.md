# LiDAR 分数策略与新 ID 确认：实现和验收

2026-10-09。功能实现为显式启用的实验选项，**参数验收未通过，不替换旧默认方案**。
本轮没有接入全周点云检测器；后方检测能力没有增加。

## 实现行为

`ClassScorePolicy` 为每个类别分别配置延续、新候选、高分确认和旧身份恢复门槛。
未配置的类别沿用 `high_confidence/low_confidence`，未配置策略时中分建 ID 路径关闭。
`LidarTrackerConfig.from_dict()` 支持 JSON 配置；CLI、KITTI replay 和 benchmark 都支持
`--config`，并拒绝同时指定会造成歧义的标量门槛参数。分数仍是检测分数，不是身份概率。

- 已确认轨迹先匹配高分，再匹配符合延续门槛的检测；休眠恢复保持独立门槛、双向最优
  和歧义拒绝。降低候选门槛不自动放宽旧身份恢复。
- 待确认轨迹优先匹配高分检测。两次高分观测可在原时间窗内确认；中分路径要求连续
  三次达到候选门槛且通过类别、位置和尺寸门控。所有观测位于 0.4 秒时间窗内，
  相邻观测间隔不超过 0.2 秒，按输入时间戳计算。标准 10 Hz 连续输入的确认延迟约 0.2 秒。
- 空帧、未匹配或过大的时间间隔重置连续中分证据。候选有总生存时限，确认历史有上限；
  一次高分加一次中分不能走“两次高分”快速确认。
- 实验配置最多保留 64 个候选，总轨迹上限仍为 256。容量不足时拒绝新候选，不移除
  已确认轨迹。新候选按分数排序，抑制与本帧已接收同类框 3D IoU ≥ 0.85 的重复候选。
  相同位置上的不同物体仍可能不可辨认，几何去重不是身份识别保证。
- 待确认框独立输出为 `tentative`，没有正式 `track_id`。正式输出新增
  `confirmation_path` 和 `confirmation_delay_s`。页面可勾选“待确认候选”，显示临时候选
  C 编号，不能将其当作稳定身份。
- 按类别记录分数过滤、匹配、候选建立/过期、确认和容量丢弃。评价器将未匹配真值区分为
  无匹配输入、分数过滤、待确认及匹配/过期等剩余原因。这是基于类别和 3D IoU 的离线
  诊断，不是逐点或人工遮挡标注。目标标签只进入独立评价与离线叠图。

## 固定验证流程

协议为 [`configs/kitti_score_validation.json`](../configs/kitti_score_validation.json)。
仅用 0000 选择参数，0001、0005 为已查看过的回归序列；0007、0010、0011 为在本轮
参数冻结后首次评分的留出序列。检测文本来自固定 AB3DMOT 提交
`61f3bd72574093e367916c757b4747ca445f978c` 的 PointRCNN 预测。
原始 KITTI 目录只读，预测文件、下载清单和大体积结果都在 Git 忽略目录内。

每个类别在有限候选门槛中，选择召回最高且类别精度下降不超过 2 个百分点、ID 切换
不增加的方案，召回相同时优先较高门槛；冻结完整配置后再评分其他序列。
冻结配置 SHA-256：`d06c495026cd111893086487aa6c883b7089b38210d38ea5cf226d6a85f8b136`。

选择结果为：Car 候选 0.95、延续 0.8；Pedestrian 候选/延续 0.6；Cyclist 候选 0.98、
延续 0.8。所有类别的高分和旧身份恢复门槛均为 0.98。
**这组配置仅作为可复现的失败实验示例，不推荐部署。**

逐类别精度约束没有保证组合后的总体精度：0000 行人精度本身只有 12.5%，降至 11.1%
仍能通过上述类别约束，但额外行人误报使总体精度显著下降。持续性错误检测也能通过
三帧确认。最后的总体验收会拒绝该配置，因此不会自动将局部选择结果提升为默认值。

自然序列结果如下；数值为本地类别一致、直立 3D IoU ≥ 0.25 的诊断，**不是官方 KITTI 分数**。
每帧目标出现分别计数，不能将其视为独立物理目标样本。

| 序列 | 用途 | 精度：旧→实验 | 召回：旧→实验 | IDF1：旧→实验 | ID 切换：旧→实验 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0000 | 参数选择 | 68.35%→62.04% | 93.79%→95.94% | 79.07%→75.35% | 0→0 |
| 0001 | 回归 | 87.05%→84.14% | 83.75%→85.68% | 85.22%→84.76% | 2→2 |
| 0005 | 回归 | 98.04%→95.31% | 60.18%→61.81% | 74.58%→74.99% | 0→0 |
| 0007 | 留出 | 91.94%→88.21% | 90.24%→91.74% | 91.08%→89.90% | 0→1 |
| 0010 | 留出 | 99.59%→98.23% | 75.43%→77.28% | 85.84%→86.51% | 0→0 |
| 0011 | 留出 | 98.34%→95.02% | 69.13%→70.44% | 78.10%→77.91% | 4→1 |

3 秒中断测试沿用原来的 27 个位置及两个保留时长共同的 **26 次固定可评机会**。
候选策略没有建立中断前轨迹也计失败，不从分母中移除；新增加的可评目标另列。
同 ID 恢复为 21/26→23/26；两者各有一次误接其他旧身份，发生在同一案例。
这属于人工全检测中断，不是自然遮挡恢复准确率。

验收结论：0005 召回仅提升 1.63 个百分点，未达到 10 个百分点；精度低于 96%；
0000、0001 总体精度下降超过 2 个百分点；0007 新增一次 ID 切换。固定恢复数量和
不新增错误旧身份恢复两项通过。配置总体被拒绝，`recommended=false`。
可移植摘要和各类别详细数据见 [`score-policy-results.json`](score-policy-results.json)。

## 工程验证

提交内容导出到独立目录后，构建 wheel 并从 wheel 加载模块，75 项发布范围内测试通过；
二维和三维 CLI 示例均通过。整个本机工作区为 79 项测试，额外四项属于未发布的
RadarScenes 实验，未用于证明此 PR 的测试范围。未启用新策略时，898 帧自然序列的
原输出字段、数值和 ID 与保存的基线逐帧一致。HTML/JS 通过语法检查，六序列的点云
资源和位姿已核对；静态点云 PNG 已渲染并检查。HTML 的浏览器交互未作运行验证。

## 复现

仓库根目录、本机已解压 KITTI 的 PowerShell 命令：

```powershell
conda run --no-capture-output -n radar-id-tracker python examples/download_kitti_predictions.py --sequences 0000 0001 0005 0007 0010 0011
conda run --no-capture-output -n radar-id-tracker python examples/benchmark_kitti.py --dataset-root C:\Projects\hx\Datasets\kitti --detections-root datasets/kitti_tracking/detections --stress-grid --output-dir outputs/kitti_reference
conda run --no-capture-output -n radar-id-tracker python examples/benchmark_score_policy.py --dataset-root C:\Projects\hx\Datasets\kitti --detections-root datasets/kitti_tracking/detections --stress-reference outputs/kitti_reference/stress_grid.json --stress-grid
conda run --no-capture-output -n radar-id-tracker python -m radar_id_tracker.pointcloud_review --dataset-root C:\Projects\hx\Datasets\kitti --output-dir outputs/kitti_score_policy --sequences 0000 0001 0005 0007 0010 0011
```

替换 `--dataset-root` 即可使用其他机器的解压目录。原版 tracking 标定无需转换。
输出包括 `protocol.json`、`tuning.json`、`frozen_config.json`、`benchmark.json`、
`acceptance.json`、`fixed_stress.json` 和 `review.html`；页面左侧为旧策略，右侧为实验策略，
两侧均保留身份 4 秒。当前点云资源有 2361 帧、282,568,426 个原始点；0001 的 177–180 帧
仍缺失。页面同时包含固定恢复失败案例。

仅验证配置接口或自己的检测输入：

```powershell
conda run --no-capture-output -n radar-id-tracker lidar-id-track examples/lidar_boxes.jsonl --config configs/lidar_medium_confirmation.example.json --output outputs/medium_example.jsonl
```

此示例配置必须显式指定，不会自动加载。配置中的门槛只适用于此次未校准的 PointRCNN
分数，不能直接迁移到其他检测器。后续应优先更换或改善检测输入，并加入总体精度及误报
约束；继续降低门槛不能解决未提供后方检测框的问题。
