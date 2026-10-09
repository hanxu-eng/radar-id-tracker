# 车载 10 Hz 激光雷达：实现与数据集测试

2026-09-30，本机已实现独立的 `LidarTracker`、KITTI 数据转换、真值隔离评分、
3 秒检测中断压力测试与可交互的俯视对比页面。此次验证的是 **3D 检测框之后的跟踪**。
没有运行原始点云检测器，也没有完成速腾设备或车辆全链路验收。

2026-10-09 新增按类别分数策略与中分候选确认；参数验收未通过，旧默认方案继续保留。
六序列实测、固定分母压力测试和复现命令见 [分数策略验收报告](lidar-score-policy.md)。

## 本轮实现

- 输入三维中心、三维尺寸、框方向、类别和检测分数；不需要毫米波多普勒。
- 六维 `x/y/z/vx/vy/vz` 匀速 Kalman 滤波，连续白加速度过程噪声、Joseph 协方差更新。
- 类别、尺寸、位置和高度门控，分阶段全局 Hungarian 匹配；低分检测只延续活跃身份。
- 两次高分观测确认新 ID，停止的目标仍可跟踪，不依赖“必须运动”。
- 未观测最多发布 0.5 秒预测，并输出 `predicted_only`、最近观测时间、漏检帧数、
  3×3 位置协方差和衰减后的检测分数。
- 身份从最近一次观测起保留 4 秒，覆盖 30 帧无检测后的约 3.1 秒返回间隔；
  超过预测发布时限只保留内部状态，避免把长时间外推当作持续可见目标。
- 休眠恢复需要双方互为最佳候选并通过歧义检查；模糊匹配被拒绝。已修复
  “最佳候选被拒绝后，较差候选接走检测”的错误恢复路径。
- ID 在单个实例运行期间递增、不复用；历史缓存和轨迹数有上限。时间戳严格递增，
  非有限值、错误检测格式被拒绝。文件模式失败时保留原输出。

尺寸和框方向采用最近观测，不作为外观重识别特征。`confidence` 反映检测分数及
预测衰减，**不是同一身份的概率**。3 秒保留具备功能支持，但此次测试未达到全场景可靠恢复。

## 数据来源及测试边界

数据位于 `datasets/kitti_tracking`，原始结果在 `outputs/`，两者均被 Git 忽略。

### 本机完整数据集（2026-10-09）

已接入 `C:\Projects\hx\Datasets\kitti`。目录检查发现训练集 21 个序列、8004 个
点云文件，测试集 29 个序列、11095 个点云文件；所有序列的标定和 OXTS 均可解析，
所有点云文件长度均为非零的 16 字节整数倍。训练集 0001 的 177–180 帧在原始
压缩包内也不存在，不能通过重新解压补齐。测试集没有公开目标标注，离线定量评价
使用训练集。检查清单保存于 `outputs/kitti_local/inventory.json`。

适配器同时支持原版 tracking 标定的 `R_rect`、`Tr_velo_cam`、`Tr_imu_velo`（无冒号）
和转换版文件的字段名。`--detections-root` 指向含 Car/Pedestrian/Cyclist 子目录的
外部预测结果，不要求向原始数据集复制文件。当前已有预测仅覆盖 0000、0001、0005；
其余序列需要补齐独立检测器预测，不能拿目标标注充当检测输入。

在仓库根目录使用本机完整数据集重跑跟踪、独立评价和原始点云叠图：

```powershell
conda run --no-capture-output -n radar-id-tracker python examples/benchmark_kitti.py --dataset-root C:\Projects\hx\Datasets\kitti --detections-root datasets/kitti_tracking/detections --output-dir outputs/kitti_local --sequences 0000 0001 0005
conda run --no-capture-output -n radar-id-tracker python -m radar_id_tracker.pointcloud_review --dataset-root C:\Projects\hx\Datasets\kitti --output-dir outputs/kitti_local --sequences 0000 0001 0005
```

结果页面为 `outputs/kitti_local/review.html`，此前的 `outputs/kitti_final` 保留。
完整数据集接入扩大了可用数据范围；以上重跑仍使用相同三段检测预测，不能视为新的
全量性能验证，也没有运行原始点云检测器。

- [KITTI Tracking 官方数据](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)：
  0000、0001、0005，分别 154、447、297 帧，共 898 帧；使用标定、GPS/IMU OXTS 和独立目标标注。
- [AB3DMOT 官方仓库](https://github.com/xinshuoweng/AB3DMOT)：使用其发布的 PointRCNN
  Car/Pedestrian/Cyclist 检测文件，固定提交 `61f3bd72574093e367916c757b4747ca445f978c`。
  格式见 [KITTI 检测说明](https://github.com/xinshuoweng/AB3DMOT/blob/master/docs/KITTI.md)。
- `manifest.json` 记录下载 URL、文件大小和 SHA-256。未下载或执行外部模型代码。
  KITTI 数据按其 [CC BY-NC-SA 3.0 许可](https://www.cvlibs.net/datasets/kitti/)使用，
  不继承本项目代码的 MIT 许可。

在线输入只读取检测、标定和车辆位姿，不读取目标标签或目标 ID。
删除或破坏标签文件的回归测试确认跟踪输出字节一致。标签只进入独立评价器及离线叠图。
这里的“真值隔离”指目标标注隔离；车辆位姿仍来自数据集提供的 GPS/IMU，尚未测试
实车定位噪声、定位漂移或坐标重置。

KITTI 框为整流相机系下的底部中心，转换为三维几何中心，再经外参和逐帧 OXTS
变换到序列固定、z 朝上的局部米制坐标。评估采用水平旋转的直立框，未评估严重横滚和俯仰。
这组 tracking 文件没有实测采集时间戳，明确使用标称 0.1 秒周期；不会把它称为实际时间戳。
发布的检测分数通过 sigmoid 映射到 `[0,1]`，没有做概率校准。

## 实测结果

分数门槛只在 0000 上比较：高/低分门槛由 `0.6/0.2` 改为 `0.98/0.8` 后，
观测精度从 38.4% 提升到 68.3%，召回从 97.1% 降为 93.8%。参数随后固定，
用于 0001、0005 和全部中断测试。这不是新检测器的独立泛化评估，也不保证其他检测模型
使用相同分数门槛；通用 `LidarTracker` 的默认门槛仍为 `0.6/0.2`。

以下为自然检测输入、4 秒保留方案的本地诊断。类别一致且 upright 3D IoU ≥ 0.25
才能匹配；逐帧优先最大有效匹配数，再最大重叠，IDF1 使用全序列身份对应。

| 序列 | 帧数 | 观测精度 | 观测召回 | 观测 IDF1 | ID 切换 | 含预测输出精度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0000 | 154 | 68.3% | 93.8% | 79.1% | 0 | 63.5% |
| 0001 | 447 | 87.0% | 83.7% | 85.2% | 2 | 72.0% |
| 0005 | 297 | 98.0% | 60.2% | 74.6% | 0 | 83.9% |

高阈值不能掩盖问题：0000 行人观测精度仅 12.5%；0001 行人召回仅 37.5%；
0005 自行车的观测召回为 0。检查全部原始检测框后，0005 自行车即使不做分数筛选，
在 IoU 0.25 下也只覆盖 39/139 次真值出现。高分门槛又进一步损失召回。
需要继续改善上游检测及按类别校准门槛，不能依靠延长跟踪记忆补齐。

门槛敏感性同样记录在 `benchmark.json`：IoU 0.5 时三个序列观测 IDF1 分别为
78.3%、84.3%、73.8%；IoU 0.7 时为 52.8%、77.1%、63.3%。本地评价不处理
KITTI 的二维 DontCare 区域，未套用官方难度筛选；只用三维 Van/Person_sitting 框
忽略未匹配预测。**这些不是 KITTI 官方 HOTA/MOTA 或排行榜结果**。

### 3 秒恢复压力测试

两个方案除保留时长 1.5 秒/4 秒外完全一致。主对比在帧 50 起隐藏连续 30 帧全部检测，
三个序列各有 1 次可评恢复机会，分别由 0/1 改为 1/1。为避免只展示这三个成功样本，
还在每个序列从帧 20 起、间隔 30 帧固定设置中断，保留至少 5 帧返回窗口，共 27 个位置。

可评机会要求中断前 0.5 秒有观测轨迹匹配真值，且返回后 0.5 秒内存在能匹配
该真值的高分**输入检测**。返回后没有输出轨迹也计失败，不能从分母中移除。
使用两个方案共同的可评机会：26 次，涉及 14 个不同物理目标；同一目标可能反复测试，
不是 26 个独立目标样本。

| 方案 | 恢复原 ID | 换 ID | 返回检测存在但无轨迹匹配 | 换 ID 中误接其他旧身份 |
| --- | ---: | ---: | ---: | ---: |
| 保留 1.5 秒 | 0/26 | 24 | 2 | 0 |
| 保留 4 秒 | 21/26（80.8%） | 4 | 1 | 1 |

未进入分母的目标，例如车辆已驶出标注视野或返回没有强检测，另行记录，不能算恢复成功。
这属于**人工全检测中断**，不是经过标注的自然遮挡测试，也不是同类多车遮挡的可靠性证明。

已发现的失败案例都进入 `review.html` 的下拉框：0001 中断起始帧 20、110、290、350、380。
其中帧 380 的案例是目标转弯后预测偏移，相似车辆的旧身份被错误恢复。
门控与拒绝歧义能减少错误，却不能凭位置和框尺寸确保遮挡后唯一身份。
后续应优先在这些失败用例上增加多帧恢复确认、可用的点云外观/几何特征或多假设关联，
并继续保留失败统计。单纯放宽距离门槛会增加错接风险。

本机自然回放的**跟踪器计算** P99 约 0.4–1.0 ms；这不包含点云接收、检测网络、
位姿计算、文件转换或 ROS 通信，不能作为全链路 10 Hz 验收依据。

## 本机复现与审查

```powershell
cd C:\Projects\hx\code\radar-id-tracker
conda run --no-capture-output -n radar-id-tracker python examples/download_kitti_tracking.py
conda run --no-capture-output -n radar-id-tracker python examples/benchmark_kitti.py --stress-grid --output-dir outputs/kitti_final
conda run --no-capture-output -n radar-id-tracker python -m unittest discover -s tests -v
```

直接用浏览器打开 `outputs/kitti_final/review.html`，选择自然检测、主中断或多位置失败案例，
点击“跳到中断结束”。左侧 1.5 秒保留，右侧 4 秒保留；黄色为离线真值，轨迹颜色按 ID，
虚线为短时预测。轨迹尾迹保留 2 秒。网格每格 10 米，视角随本车平移。
可切换俯视和三维斜视、旋转方向、调整视野，三维框标注 ID、类别和预测状态。
页面不需要第三方 CDN，点云资源位于同目录 `pointclouds/`，不能只复制 HTML。
框显示数字保留 4 位小数，原始 JSONL 和评分保留全精度。
已验证内嵌数据、帧对齐和 JavaScript 语法；本次工具无法打开本地 HTML 做浏览器截图验收。

主要结果：

- `outputs/kitti_final/benchmark.json`：自然检测与主中断的完整结果，以及 IoU 敏感性。
- `outputs/kitti_final/stress_grid.json`：27 个中断位置、共同分母、逐目标恢复和错误身份统计。
- 每个运行目录下 `detections.jsonl`、`tracks.jsonl`、`summary.json`、`evaluation.json`、
  `evaluation_frames.jsonl` 可逐帧追查。原有 `outputs/` 内容保留。
- `outputs/kitti_tuning/benchmark.json`：0000 单序列的门槛比较。

加入点云可视化后共 58 项自动测试通过，覆盖 3 秒移动/停止目标恢复、过期 ID、交叉和启停、
类别/尺寸/高度拒绝、歧义与错误接走回归、预测衰减/协方差、长期缓存上限、标定变换、
KITTI 空帧、标签隔离、评分已知答案、时间对齐与 CLI 输出保护。

## 原始点云及标注（新增）

```powershell
conda run --no-capture-output -n radar-id-tracker python examples/download_kitti_pointclouds.py
conda run --no-capture-output -n radar-id-tracker python -m radar_id_tracker.pointcloud_review
```

下载器通过 HTTP Range 读取官方约 35 GB ZIP 的目录及选定序列数据，只下载本轮
894 个实际存在的点云帧，约 1.58 GiB。每帧校验 ZIP CRC，并记录 SHA-256 和源归档 ETag，
支持已下载文件校验后跳过。原始 float32 `x/y/z/reflectance` 文件位于
`datasets/kitti_tracking/training/velodyne/`，不修改、不替换原始点。
官方归档没有 0001 序列的 177–180 帧，页面显示缺帧提示，不能沿用上一帧或补造点云。

894 帧共 105,754,399 点，此次打包没有无效点、坐标溢出或反射强度裁剪。
可视化采用与检测框相同的标定、逐帧 OXTS 和固定世界轴，浏览器显示坐标量化到 1 cm；
这是显示编码，原始文件和评分保持原精度。默认采样预览约 3 万点，可勾选“全量点”
在 WebGL 模式绘制所有打包点；无 WebGL 时降级为最多 3 万点的兼容显示，页面会提示。
点大小、视野、俯仰和方向可调，按帧加载本地资源，内存缓存最多 6 帧。

点云可按相对雷达高度、反射强度或跟踪 ID 着色。ID 着色使用已观测跟踪框的几何包含关系，
最多给前 24 个框内的点着色；所有跟踪框均绘制。它不是实例分割或点级真值。
真值框单独切换，仅用于离线审查；预测框虚线显示“预测”。框与尾迹绘制在点云上方，
便于看清身份，因此显示遮挡关系不能当作物理遮挡标注。
人工检测中断期间仍显示实际点云，帮助核对检测中断与目标离开的区别。

`outputs/kitti_final/pointcloud_preview.png` 提供原始点云三维标注预览：0000 帧 81，
短时方案的自行车由 ID 2 换为 ID 8，4 秒方案保持 ID 2；匹配对应独立真值 ID 1。
这张导出图使用区域裁剪和采样，完整逐帧点云见网页。
可通过 `examples/render_pointcloud_preview.py` 重生成，额外需要 Matplotlib；
核心跟踪和网页生成仍只依赖 NumPy。
静态图已目视检查，网页的内嵌脚本、资源覆盖和坐标对齐已验证；尚未完成浏览器 WebGL 截图验收。

## 接入接口（不依赖 ROS）

```python
from radar_id_tracker import BoxDetection, LidarTracker, LidarTrackerConfig

tracker = LidarTracker(LidarTrackerConfig(retention_seconds=4.0))
# 调用前，上游已经把本帧检测框转换到同一固定米制坐标系，z 朝上。
tracks = tracker.update(timestamp_s=0.0, detections=[
    BoxDetection(x=10., y=2., z=1., length=4.2, width=1.8, height=1.6,
                 yaw=0., confidence=.99, label='Car')
])
```

`yaw` 为弧度，框长轴沿该方向，框方向按模 π 比较，不能直接当作运动方向。
空帧也调用 `update(time, [])`。实机使用实际采集时间，不能照搬数据集标称周期。
里程计重置、时间回退或重启需要新建实例并由外部流编号区分 ID 命名空间。

普通 JSONL 示例与命令：

```powershell
conda run --no-capture-output -n radar-id-tracker lidar-id-track examples/lidar_boxes.jsonl --output outputs/lidar_example_tracks.jsonl
```

自行转换的输入只应包含检测字段，不能包含真值 ID。对于本轮 KITTI 文件，使用
`--high-confidence 0.98 --low-confidence 0.8` 可与数据集脚本保持一致；其他检测器需自行校准。
命令行也可通过 `python -m radar_id_tracker.lidar_cli` 调用。

ROS 接入之前，还需补上原始点云检测、逐点时间与去畸变、真实自车定位及坐标重置处理，
并在实际定位噪声、目标交叉、遮挡和丢帧数据上重复这套评分。硬件和系统选择本轮暂不展开。
