# RadarScenes 实际帧率接入与首轮测试

这条链路直接读取 RadarScenes 官方格式中的一条序列、一部雷达的每次扫描，按真实采集时间运行跟踪器；**不做 10 Hz 重采样，也不合并相邻扫描**。官方说明每部雷达的平均测量周期约 60 ms，但实际间隔以 `scenes.json` 的微秒时间戳为准。

这是可运行的接入和诊断基线，不是经过训练和精度验收的目标检测器。脚本使用单帧 `x_seq/y_seq/vr_compensated` 做简单空间与多普勒连通聚类，完全不读取逐点 `track_id` 或 `label_id` 来生成检测。常数置信度仅为接口占位，不代表校准过的检测概率。[官方结构说明](https://radar-scenes.com/dataset/structure/)、[官方数据下载](https://zenodo.org/records/4559821)、[官方读取工具](https://github.com/oleschum/radar_scenes)。

## 1. 核对解压结果

按 RadarScenes 官方包的目录结构，找到某条序列，例如：

```text
RadarScenes/
├── sensors.json
├── sequences.json
└── data/
    └── sequence_137/
        ├── scenes.json
        ├── radar_data.h5
        └── camera/
            └── *.jpg
```

有些文档把安装文件写作 `sensor.json`；脚本会从 `scenes.json` 的父目录向上查找 `sensors.json` 或 `sensor.json`，也可用 `--sensors` 明确指定。不要把 11.1 GB 数据集拷入本 Git 仓库。数据集许可是 CC BY-NC-SA 4.0，不可商用。

## 2. 在已经创建的 Conda 环境中安装可选读取依赖

```bash
conda activate radar-id-tracker
conda install -c conda-forge h5py
python -m pip install radar_scenes --no-deps
python -c "from radar_scenes.sequence import Sequence; import h5py; print('RadarScenes reader OK')"
```

`--no-deps` 只安装本测试所需的官方读取 API，不安装它的图形查看器依赖。本仓库本体不需要 `h5py` 或 `radar_scenes`；只有运行这条数据集接入时才需要。

## 3. 先跑前 200 次真实扫描

将下面 `--scenes` 改为您电脑上的绝对路径。`--sensor-id` 取 1、2、3 或 4；若该序列没有指定传感器，脚本会列出可用编号。

```bash
python -m radar_id_tracker.radarscenes_adapter \
  --scenes "/path/to/RadarScenes/data/sequence_137/scenes.json" \
  --sensor-id 1 \
  --output-dir "outputs/sequence_137_sensor1" \
  --limit 200
```

Windows PowerShell 可以写成单行：

```powershell
python -m radar_id_tracker.radarscenes_adapter --scenes "D:\RadarScenes\data\sequence_137\scenes.json" --sensor-id 1 --output-dir "outputs\sequence_137_sensor1" --limit 200
```

如果传感器外参文件不在数据集根目录，加上 `--sensors "/path/to/RadarScenes/sensors.json"`。如安装后提示找不到 `radar_id_tracker.radarscenes_adapter`，先在仓库根目录执行 `python -m pip install -e . --no-deps`。

默认在输出目录写三个文件：

- `detections.jsonl`：逐次扫描的候选目标，可再次用 `radar-id-track` 回放。
- `tracks.jsonl`：逐次扫描的公开 `track_id`，首帧没有 ID 是正常的。
- `summary.json`：读取与输出计数、真实中位扫描间隔、处理时间及“非精度验收”标记。

### 查看稳定 ID 的俯视角回放

在原命令末尾加 `--visualize`，即可额外生成浏览器可打开的俯视角回放。这里使用您实际扫描间隔控制播放，不把 RadarScenes 强行变成 10 Hz，也不需要安装 OpenCV、Matplotlib、FFmpeg 或浏览器服务器。

```powershell
python -m radar_id_tracker.radarscenes_adapter --scenes "C:\Projects\hx\Datasets\RadarScenes\data\sequence_137\scenes.json" --sensor-id 1 --output-dir "outputs\sequence_137_sensor1" --limit 200 --visualize
```

完成后，在文件管理器中双击 `outputs\sequence_137_sensor1\visualization.html`，或把它拖进浏览器。`frames\frame_000000.svg` 等逐帧图像必须与 HTML 保持原有相对目录；若复制结果到另一台电脑，请一起复制整个输出目录。页面可播放、暂停、逐帧和拖动进度条；1× 播放按扫描时间戳之间的实际间隔调度，浏览器后台限速可能使显示稍慢，但**跟踪计算仍始终使用原始时间戳**。

图中灰点是当前雷达原始点、橙方块是无标签聚类中心、彩色圆及 `ID n` 是跟踪器公开的编号；空心虚线圆与 `(P)` 表示这一帧仅靠预测、没有重新观测到该轨迹。彩色线是该 ID 最近 2 秒的路径，预测段用虚线。黑点是当前雷达位置。坐标使用 RadarScenes 固定序列系：雷达始终处于画面中心，视窗随车平移，但**不随车旋转**；默认可见范围为当前位置四周各 60 m。超出画面的点或轨迹不会绘出，但不会影响跟踪数据。

视距和轨迹尾迹可调整，例如 `--vis-range-m 100 --trail-seconds 3`。`--visualize` 会额外写入每帧一张 SVG，长序列要预留磁盘空间；先用 `--limit 200` 检查，再决定是否处理全序列。反复写同一个输出目录时，HTML 仅引用本次生成的帧，但旧的多余 SVG 可能留在 `frames` 中；需要整洁归档时使用新的输出目录。该可视化并非 MP4，也没有拿官方 `track_id` 绘制或验证真值。

### 可选：同步并排查看相机

如果 `sequence_137/camera/` 已下载，在原命令末尾同时加 `--visualize --with-camera`：

```powershell
python -m radar_id_tracker.radarscenes_adapter --scenes "C:\Projects\hx\Datasets\RadarScenes\data\sequence_137\scenes.json" --sensor-id 1 --output-dir "outputs\sequence_137_sensor1_camera" --limit 200 --visualize --with-camera
```

播放器左边是雷达俯视图，右边是该扫描在 `scenes.json` 中关联的**时间最近**相机图；共同使用雷达时间轴。若文件名能解析为微秒时间戳，右侧还显示“相机时间 − 雷达时间”，`summary.json` 记录偏差的中位绝对值和最大绝对值。所需图像会复制到输出目录的 `camera/`，重复使用的相机图只复制一次；复制整个输出目录即可离线查看。若缺少某张图，脚本会在写结果前报出其路径；此时先确认相机文件已经解压，或者去掉 `--with-camera` 继续只看雷达。

这是**并排场景参考**，不是相机与雷达的几何融合。RadarScenes 的相机是记录用，官方 `sensors.json` 仅描述雷达安装位置与角度；没有经本项目验证的相机内外参，就不能把雷达 `ID` 精确投到图像像素上。四部雷达朝向各异，视野也不必与记录相机重合；右边看不到左边的目标，不一定是雷达错误。[官方数据结构](https://radar-scenes.com/dataset/structure/)、[传感器布局](https://radar-scenes.com/dataset/sensors/)。

日志和 `summary.json` 中，`frames > 0`、`radar_points > 0` 且两份 JSONL 行数均等于 `frames`，就表示**文件读取和帧级接入成功**。`cluster_detections > 0` 且 `unique_published_ids > 0` 才表示在这段数据上也实际生成了目标和 ID。测试区间若没有足够运动目标，后两个值可能为零，并不必然是读取失败。`median_interval_s` 来自真实选中扫描，不能由标称频率硬填。

确认短序列能跑后，去掉 `--limit 200` 处理完整序列。每条序列和每部雷达用不同的 `--output-dir`，因为跟踪 ID 从 1 重新编号；再次运行同一目录会覆盖本次结果文件。

## 4. 默认聚类参数与诊断

| 参数 | 默认 | 用途 |
| --- | --- | --- |
| `--min-abs-vr` | `0.5` m/s | 仅保留补偿后径向速度绝对值达到阈值的点。 |
| `--spatial-eps` | `2.0` m | 单次扫描内两点可连通的最大平面距离。 |
| `--doppler-eps` | `1.5` m/s | 两点可连通的最大径向速度差。 |
| `--min-points` | `2` | 每个候选目标最少点数；稀疏时可暂设为 `1`。 |
| `--confidence` | `0.75` | 占位分数，**未校准**；保证通过跟踪器默认高分门槛。 |
| `--doppler-sign` | `as-is` | 使用 `vr_compensated`；若实测符号与跟踪器约定相反可选 `invert`，或先选 `off` 禁用多普勒关联。 |

如 `frames` 正常但 `cluster_detections=0`，先检查该传感器这 200 帧是否有运动目标，再尝试 `--min-points 1 --min-abs-vr 0.2`。这只用于确认管线，不是提高精度的结论；降低阈值会增加静态杂波和虚警。官方也提醒：单靠多普勒阈值不能可靠地区分真实运动物体。[标注说明](https://radar-scenes.com/dataset/labeling/)。

如有候选目标但 ID 经常跳变，检查聚类是否把一个物体拆成多团或把相邻物体合并；再检查多普勒符号、传感器位置与里程计。脚本的 `median_cluster_and_track_ms` 只计聚类和跟踪，不含文件读取/写入；不能当作端到端 60 ms 时限的验收值。

## 5. 字段和评估边界

| RadarScenes 字段 | 处理方式 |
| --- | --- |
| `scene.timestamp` | 微秒；相对首个选中扫描转为秒，保留真实间隔。 |
| `scene.sensor_id` | 只取 `--sensor-id` 指定雷达，不把四部异步雷达当成同一帧。 |
| `radar_data.x_seq/y_seq` | 已在序列固定坐标系；聚类后取中心。 |
| `radar_data.vr_compensated` | 自车运动补偿后的径向速度；聚类取中位数。 |
| `scene.odometry_data.x_seq/y_seq/yaw_seq`、`sensors.json` 的 `x/y` | 计算该帧雷达在序列坐标系中的 `sensor_xy`。 |
| `radar_data.track_id/label_id` | **不进入检测和跟踪**；将来单独实现真值匹配与指标评估。 |

官方逐点 ID 在目标遮挡或停止超过约 500 ms 后可能重新分配；不能直接把不同时间的同一物理车当作必然相同的真值 ID。首轮接入只回答“是否正确读取并跑通真实帧”，**不回答** IDF1、HOTA、检测精度或生产实时性是否达标。

## 6. 如何判断 ID 跟踪效果

当前 `summary.json` 中的 `unique_published_ids`、`published_track_outputs` 和处理时间只能说明**有输出且跑得动**；它们不能说明 `ID 1` 是否跟对了同一辆车。相机并排图可以帮助判断道路场景，但它不具有真值标注资格。建议先在播放器中逐帧审查十字路口、两车交叉、短时漏检、远距离稀疏点和转弯片段：记录目标是否始终保留同一预测 ID、是否一个目标产生多个 ID、两目标是否被合成一个、无目标处是否长期存在彩色轨迹。`(P)` 连续很多帧且车已离开视野时，可能是假轨；一出现遮挡就换号，可能是关联或保活问题。

可复现的量化验收应使用**未参与检测和跟踪**的官方逐点 `track_id/label_id` 作离线真值：逐帧把同一官方 ID 的动态点组成真值对象，固定中心/范围的定义与一对一匹配门限，再统计检测召回、虚警、ID switches、轨迹断裂，并计算 IDF1/HOTA。要同时报告评估序列、传感器、目标距离分层、匹配门限和是否计入仅预测轨迹；否则分数不可比较。[TrackEval](https://github.com/JonathonLuiten/TrackEval) 提供 IDF1/HOTA 参考实现，但**本仓库尚未实现 RadarScenes 到该评测格式的转换，不能把当前摘要当成这些指标**。官方标注只覆盖运动目标，且遮挡或停止超过约 500 ms 可能重新赋新真值 ID，因此这些事件需要单独标记，避免把真值定义变化误判为跟踪器换号。[官方标注规则](https://radar-scenes.com/dataset/labeling/)。

若审查发现问题，先定位是“单帧聚类没有产生正确候选”，还是“候选正确但关联换号”；不要同时改聚类阈值和跟踪门控。当前单帧空间＋多普勒聚类是管线基线，官方也明确指出仅按多普勒阈值不能可靠分离所有动态目标。改正时先保存当前命令和输出为对照，再只调整一个主要变量并重跑同一序列片段；比较漏检、虚警和换号是否共同改善。
