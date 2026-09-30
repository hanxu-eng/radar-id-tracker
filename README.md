# Radar ID Tracker：10 Hz 雷达动态目标稳定编号

本仓库是一个在线多目标跟踪器。每收到一帧雷达**检测结果**，它预测已有目标的位置，用位置与多普勒速度进行匹配，并输出持续使用的 `track_id`。标称输入频率为 10 Hz，即约每 0.1 秒处理一帧；程序无需等待凑齐 10 帧。

核心方法：直行/加速双模型 IMM 滤波、位置与多普勒门控、分级 Hungarian 匹配、轨迹确认、短时漏检预测和休眠恢复。跟踪器在 CPU 上运行。RTX 4080 可以运行上游检测模型，但仓库**没有**训练好的 CenterPoint、TensorRT 引擎或真实雷达数据；也没有经过真实场景的精度与全链路延迟验收。

## 1. 在另一台电脑上准备环境

以下安装步骤面向 Linux、macOS、Windows；建议使用 Python 3.11 的 Conda 环境。本项目只需 NumPy，不要求 CUDA、PyTorch 或 NVIDIA 驱动就能运行示例跟踪器。当前自动化测试运行于 Linux，其他系统请按下文在目标电脑上执行测试确认。

先准备：

- [Git](https://git-scm.com/downloads)；用 `git --version` 检查。
- Conda。尚未安装时可按 [Conda 官方安装指南](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)安装 Miniforge 或 Miniconda；Windows 请使用安装后的 Miniforge/Anaconda Prompt，Linux/macOS 使用终端。
- 可以访问 GitHub 与 conda-forge 的网络。公开仓库克隆无需 GitHub 登录。

以下命令均在终端执行。Windows 建议使用 PowerShell 或 Miniforge Prompt；路径分隔符由系统处理。

### 步骤 1：克隆仓库

```bash
git clone https://github.com/hanxu-eng/radar-id-tracker.git
cd radar-id-tracker
```

没有 Git 时也可在 GitHub 仓库页面选择 **Code → Download ZIP**，解压后进入含有 `environment.yml` 和 `pyproject.toml` 的目录。

### 步骤 2：创建并激活 Conda 环境

```bash
conda env create -f environment.yml
conda activate radar-id-tracker
python --version
```

预期 Python 为 3.11.x。`environment.yml` 从 conda-forge 安装 Python、pip、NumPy；[Conda 官方说明](https://docs.conda.io/projects/conda/en/stable/commands/env/create.html)也使用 `conda env create -f environment.yml` 创建命名环境。

如果 `conda activate` 提示未初始化 shell，先在当前 shell 执行 `conda init`，关闭并重新打开终端，再运行 `conda activate radar-id-tracker`。Windows 也可以直接打开 Miniforge Prompt。

### 步骤 3：安装本仓库

确认终端所在目录仍为 `radar-id-tracker`，然后运行：

```bash
python -m pip install -e . --no-deps
python -c "import radar_id_tracker, numpy; print(radar_id_tracker.__file__); print(numpy.__version__)"
```

`-e` 表示开发模式安装，代码仍指向当前克隆目录。`--no-deps` 表示使用 Conda 已安装的 NumPy。如果第二条命令报 `ModuleNotFoundError`，先运行 `conda activate radar-id-tracker`，再在仓库根目录重试安装。

### 步骤 4：运行自动测试

```bash
python -m unittest discover -s tests -v
```

预期最后出现 `OK`。测试包含目标交叉、短时漏检、休眠恢复、低分虚警、时间戳变化以及匈牙利分配。GitHub Actions 也在 Python 3.10 和 3.13 上运行相同测试。

### 步骤 5：运行自带的 10 Hz 示例

在仓库根目录运行：

```bash
python -m radar_id_tracker examples/detections.jsonl
```

也可以使用安装后的命令行入口，并把结果保存为文件：

```bash
radar-id-track examples/detections.jsonl --output tracks.jsonl
```

示例共有四帧，时间戳分别为 0.0、0.1、0.2、0.3 秒。第一帧建立暂定轨迹，第二帧输出 `track_id: 1`；第三帧没有检测，仍输出同一 ID，并标记 `predicted_only: true`；第四帧重新检测到目标，恢复正常观测输出。`tracks.jsonl` 每行对应输入的一帧。

如需在另一终端或日后重新运行，先进入克隆目录并激活环境：

```bash
cd radar-id-tracker
conda activate radar-id-tracker
python -m radar_id_tracker examples/detections.jsonl
```

处理自己录制的文件时，将示例路径换成实际文件路径，例如：

```bash
radar-id-track my_detections.jsonl --output my_tracks.jsonl
```

命令行适合离线回放；实时接入设备时，在采集回调或处理循环中按时间顺序调用下文的 Python API。

## 2. 接入自己的雷达数据

### 输入是什么

命令行输入是 JSONL 文本：**每行一个雷达帧**。当前程序接收目标检测列表或聚类后的目标中心，不直接接收原始 ADC、Range-Doppler 热图或未聚类点云。若设备已经输出目标列表，先将其转换成下面的格式；若只有点云，应先运行自己的聚类/检测器。

```json
{"timestamp_s":0.0,"sensor_xy":[0.0,0.0],"detections":[{"x":10.0,"y":2.0,"confidence":0.91,"radial_velocity":3.0,"length":4.2,"width":1.8}]}
{"timestamp_s":0.1,"sensor_xy":[0.0,0.0],"detections":[{"x":10.3,"y":2.0,"confidence":0.88,"radial_velocity":3.0,"length":4.2,"width":1.8}]}
{"timestamp_s":0.2,"sensor_xy":[0.0,0.0],"detections":[]}
```

必填字段：

| 字段 | 含义 |
| --- | --- |
| `timestamp_s` | 实际采集时间，单位秒；必须严格递增，可为小数。不能用帧编号代替。 |
| `detections` | 本帧检测列表；无检测时填空数组 `[]`，仍需送入跟踪器。 |
| `x`, `y` | 每个检测目标的中心位置，单位米。 |
| `confidence` | 每个检测目标的置信度，范围 `[0, 1]`。 |

帧级可选字段 `sensor_xy` 表示本帧雷达在同一固定坐标系中的 `[x, y]`（米）；省略时默认 `[0, 0]`。如果雷达在移动，必须提供其真实位置。

可选的检测字段：`radial_velocity`（m/s）、`z`（m）、`length`/`width`/`height`（m）、`yaw`（弧度）。未提供多普勒时仍能运行，但目标交叉时更易换 ID。`z`、尺寸和航向沿用最近一次观测；当前滤波器只跟踪地面平面的 `x/y/vx/vy`。神经网络检测器可提供目标框中心和置信度，但不可把二维速度模长直接当作径向速度。

### 坐标与多普勒约定

`x/y` 与 `sensor_xy` 必须属于**同一个固定米制坐标系**，例如车辆里程计坐标系。车载雷达的自车坐标每帧会移动，应先使用里程计/定位把检测中心变换到固定坐标系，再输入跟踪器。坐标系重置后，重新创建一个跟踪器实例，并在外部记录新的流/运行编号。

`radial_velocity` 使用“目标远离雷达为正”的约定，且必须已补偿雷达自身运动。若设备输出的是目标相对雷达的径向速度 `v_raw`，视线单位向量为 `u`，雷达在固定坐标系中的速度为 `v_sensor`，则转换为 `v_compensated = v_raw + dot(u, v_sensor)`。先核对设备驱动的正负号定义；符号反了会导致关联失败。

输出同样是每帧一行 JSONL。每行包含 `timestamp_s` 和 `tracks` 数组；每个轨迹含 `track_id`、`x/y`、`vx/vy`、`predicted_only`、`missed_frames`、置信度及最近观测的可选尺寸字段。`tracks: []` 表示这一帧没有可发布的已确认轨迹。

### 置信度与 ID 的默认行为

- 高分检测：`confidence >= 0.5`，可新建暂定轨迹。
- 低分检测：`0.2 <= confidence < 0.5`，可延续已确认轨迹，不能创建新 ID。
- 新目标在 0.3 秒内累计两次高分命中后发布正式 ID，因此通常从第二个有效观测帧开始输出。
- 短时漏检期间最多保留 0.5 秒的预测输出，`predicted_only` 为 `true`。
- 再保留 1.0 秒休眠恢复窗口；休眠时不输出，重新匹配成功后恢复原 ID。
- 单个跟踪器实例中的 ID 从 1 开始递增，运行期间不复用。重启进程后会重新从 1 开始。

这些是配置初值，需用实际雷达数据调参。查看 [TrackerConfig](src/radar_id_tracker/tracker.py) 可调整门控阈值、确认时间和保活时间。

## 3. 在 Python 程序中调用

```python
from radar_id_tracker import Detection, RadarTracker, TrackerConfig

tracker = RadarTracker(TrackerConfig(lost_seconds=0.5, dormant_seconds=1.0))

# 每收到一帧调用一次；没有目标时也传入空列表。
frames = [
    (0.0, [Detection(x=10.0, y=2.0, confidence=0.9, radial_velocity=3.0)]),
    (0.1, [Detection(x=10.3, y=2.0, confidence=0.9, radial_velocity=3.0)]),
    (0.2, []),
]
for timestamp_s, detections in frames:
    tracks = tracker.update(timestamp_s, detections, sensor_xy=(0.0, 0.0))
    for track in tracks:
        print(timestamp_s, track.track_id, track.x, track.y, track.predicted_only)
```

为每路独立雷达数据流创建一个 `RadarTracker`。如果需要跨进程重启也不重复的全局 ID，请把设备 ID 和运行批次与 `track_id` 组合保存。

## 4. 推荐的公开雷达数据集与接入

**首选 RadarScenes**：它公开雷达点云、真实采集时间、序列坐标、已补偿径向速度以及逐点目标 ID，最适合先查验位置/多普勒关联与换 ID 情况。不过它不是原生 10 Hz：官方给出的单传感器平均扫描周期约 60 ms。不要把四台雷达的异步扫描拼成“一台 10 Hz 雷达”，也不要把帧号乘 0.1 当作真实时间。[官方数据下载（Zenodo）](https://zenodo.org/records/4559821)、[字段说明](https://radar-scenes.com/dataset/structure/)、[传感器频率](https://radar-scenes.com/dataset/sensors/)、[官方 Python 工具](https://github.com/oleschum/radar_scenes)。数据约 11.1 GB，许可为 CC BY-NC-SA 4.0，**不可用于商业用途**。

数据下载完后，按 [RadarScenes 实际帧率接入与首轮测试](docs/radarscenes.md) 直接运行本仓库的无标签逐帧聚类基线。它能生成 `detections.jsonl`、`tracks.jsonl` 和 `summary.json`；加 `--visualize` 还能生成可直接在浏览器打开的 `visualization.html` 和逐帧 SVG，查看原始点、聚类中心、稳定 ID、预测态与轨迹尾迹。这便于先验证真实数据的读取、时间戳、传感器位置和 ID 输出；它不是已训练检测器，也不能据其输出宣称真实场景精度达标。

| 数据集 | 为什么推荐 | 接入时的限制 |
| --- | --- | --- |
| [RadarScenes](https://zenodo.org/records/4559821) | 逐点类别/目标 ID、`x_seq/y_seq`、`vr_compensated` 和里程计；适合先做雷达专用跟踪实验。 | 点云不是目标检测框；须先聚类或检测。逐点真值 ID 只用于离线评估。 |
| [View of Delft（VoD）](https://github.com/tudelft-iv/view-of-delft-dataset) | 单个前向 3+1D 雷达约 13 Hz，配套标定、里程计和跨帧 3D 框 ID，适合接近实际车载 4D 雷达的验证。 | 须按[官网流程申请访问](https://viewofdelft-dataset.tudelft.nl/)；仅限符合许可条件的非商业学术研究。原始发布的框**没有**跟踪 ID，须另取[带 ID 标注](https://github.com/tudelft-iv/view-of-delft-dataset/blob/main/docs/ANNOTATION.md)。 |
| [nuScenes / mini](https://www.nuscenes.org/nuscenes) | 五部雷达、多传感器标定及跨帧 `instance_token`，适合检查多传感器坐标链与数据管线；[官方教程含 mini 下载](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html)。 | 跟踪真值与官方评测按 **2 Hz 关键帧**，不适合作为 10 Hz 每帧 ID 稳定性的主要验收集。中间 radar sweeps 没有逐帧框真值。 |

### 通用接入流程

1. **按序列处理**：每个 recording/scene 单独建立跟踪器，不跨序列继承 ID。每个雷达测量使用原始采集时间，不能把多帧点云融合后仍当作单帧输入。
2. **产生检测结果**：雷达自带目标列表、点云聚类或检测模型输出每帧目标中心和 `[0,1]` 置信度。真值框可作“理想检测输入”仅调试关联器，不能据此宣称真实雷达检测加跟踪性能。禁止把数据集的 `track_id`、`instance_token` 输入跟踪器。
3. **统一坐标**：用外参和里程计把目标中心、雷达自身位置转换到同一个序列固定米制坐标系。若有径向速度，统一“远离雷达为正”并补偿自车运动；确认之前可以先省略 `radial_velocity`，不能把速度模长直接填入。
4. **输出每帧数据**：生成第 2 节的 JSONL；没有检测也保留空帧。按时间顺序运行 `radar-id-track`。预测 ID 与真值 ID 分开保存，最后再做 IDSW、IDF1/HOTA 和漏检恢复率评估。

### CSV 桥接示例（适用于上述数据集的检测器输出）

若你的预处理/检测器已经输出 CSV，可参考 [examples/detections.csv](examples/detections.csv)。一行一个目标，同一帧的多目标共用 `timestamp_s`；无目标时也保留一行时间戳，而 `x,y,confidence` 留空。字段如下：

```text
timestamp_s,sensor_x,sensor_y,x,y,confidence,radial_velocity,length,width
0.0,0.0,0.0,10.0,2.0,0.91,3.0,4.2,1.8
0.1,0.0,0.0,10.3,2.0,0.88,3.0,4.2,1.8
0.2,0.0,0.0,,,,,,
```

其中 `timestamp_s,x,y,confidence` 是列名必需项，空帧的数据单元可留空；`sensor_x/sensor_y` 必须成对出现，省略时固定为零；`radial_velocity,z,length,width,height,yaw` 均可选。CSV 必须只含**一条序列**且按时间排序。转换脚本会拒绝额外列（包括真值 `track_id`），避免把标签泄漏进跟踪输入；输入有误时不会覆盖已有输出文件。

```bash
python examples/csv_to_jsonl.py examples/detections.csv converted.jsonl
radar-id-track converted.jsonl --output converted_tracks.jsonl
```

对于 RadarScenes，如使用[官方 `Sequence.from_json` 接口](https://github.com/oleschum/radar_scenes/blob/master/radar_scenes/sequence.py)，可在现有 Conda 环境中按需安装读取依赖（不安装图形查看器的依赖）：

```bash
conda install -c conda-forge h5py
python -m pip install radar_scenes --no-deps
```

然后读取 `data/sequence_XXX/scenes.json`，对该序列实际包含的雷达分别用 `sequence.scenes(sensor_id=1)` 等迭代。每个 `scene.timestamp` 是微秒，写 CSV 时除以 `1_000_000`；目标中心使用同帧检测器/聚类结果的序列坐标（可参考逐点 `x_seq/y_seq`），多普勒可参考逐点 `vr_compensated` 汇聚。聚类器若没有置信度，需要单独校准评分，不能直接拿真值标签伪造置信度；也**不要**用逐点真值 `track_id` 来聚类形成正式评测输入。`sensor_x/sensor_y` 应由该雷达的安装外参与该帧里程计姿态计算；只在雷达固定不动时才可省略。每个传感器/序列单独导出一个 CSV；若计划融合四雷达，先完成严格的时间同步和多雷达去重，再输入单个跟踪器。官方只对运动物体做目标标注，且在遮挡或停止超过约 500 ms 后可能给同一物体新 ID，计算恢复率时必须遵循其标注协议。[标注说明](https://radar-scenes.com/dataset/labeling/)。

VoD 的 `label_2` 框位于相机坐标系，不能直接把其中的 x/y 填给本跟踪器；要先用标定转换到雷达/车体坐标，再结合里程计转到固定序列坐标。带 ID 的标注与检测输入应分开保存。nuScenes 同样要使用各雷达 `sample_data` 的实际时间、外参和 ego pose；2 Hz `sample_annotation.instance_token` 仅作关键帧真值，不得插值为“每帧真值”后宣称 10 Hz 跟踪精度。

## 5. 算法与部署边界

每帧依次执行：按实际时间差预测现有轨迹；用位置创新距离、最大残差、多普勒残差和可选尺寸变化进行门控；按“已确认轨迹配高分检测 → 未配对的已确认轨迹配低分检测 → 休眠轨迹严格匹配高分检测 → 暂定轨迹配高分检测”四级顺序做确定性 Hungarian 分配；最后新建暂定轨迹或执行漏检保活。IMM 包含低加速度的匀速模型和匀加速模型。

单张 RTX 4080 只在接入上游 GPU 检测器时有用；本仓库本身是检测结果到稳定 ID 的 CPU 跟踪模块，不包含训练、点云预处理、外参标定、检测器权重或 TensorRT 引擎。实际部署建议采集、检测、跟踪分阶段异步运行，以采集时间戳排序，并限制等待队列长度，避免处理过期帧。接入接口为上文 JSONL 格式或 `Detection` 对象。

当前实现有边界：只滤波地面二维运动，未解决多个物体长时间合并为一个检测的身份歧义；缺少多普勒或多个物体多普勒接近时，交叉场景仍可能换 ID。Python 实现可作原型和工程基线，最终是否满足吞吐要求需按实际目标数和硬件实测。

## 6. 性能与验收

用合成检测运行 CPU 跟踪器基准：

```bash
python examples/benchmark.py
```

输出 `median_ms` 与 `p99_ms`。这是**跟踪模块**在当前电脑上的耗时，不包含雷达采集、点云预处理、检测模型、进程间传输和显示。对于 RTX 4080 系统，可将训练好的检测模型部署在 GPU，将其结果按上述接口输入跟踪器；本仓库不会自动下载或训练检测模型。

真实部署前应在录制数据上分别统计 IDSW、IDF1、HOTA/AssA、漏检后原 ID 恢复率，以及端到端 P50/P95/P99 延迟。重点覆盖目标交叉、并行、急转、1～5 帧漏检、远距离弱目标和虚警。示例测试通过并不等于达到真实场景的指标。

## 7. 常见问题

| 现象 | 检查方法 |
| --- | --- |
| `conda: command not found` | 安装 Conda 后重新打开终端；Windows 使用 Miniforge/Anaconda Prompt。 |
| `EnvironmentNameNotFound` | 在仓库根目录重新运行 `conda env create -f environment.yml`，再激活。 |
| `ModuleNotFoundError: radar_id_tracker` | 先激活环境，在仓库根目录运行 `python -m pip install -e . --no-deps`。 |
| 第一帧 `tracks` 为空 | 正常；默认需要两个有效观测才分配正式 ID。 |
| 一直没有 ID | 检查置信度是否达到 0.5、时间戳是否递增、目标位置单位是否为米，以及门控条件。 |
| 报错 `frame timestamps must strictly increase` | 按采集时间排序，每帧只调用一次；不要重复发送同一时间戳。 |
| 目标交叉时 ID 改变 | 检查多普勒正负号、自车速度补偿、坐标系转换与检测质量，再根据录制数据调门控。 |
| 运行速度达不到 10 Hz | 用基准脚本定位跟踪耗时，并分别测采集、检测和传输；减少目标数或使用编译语言实现。 |

## 8. 仓库结构

```text
environment.yml                 Conda 环境
src/radar_id_tracker/            跟踪器、IMM、匹配和命令行
examples/detections.jsonl        四帧输入示例
examples/detections.csv          CSV 桥接输入示例
examples/csv_to_jsonl.py         检测器 CSV 转 JSONL
examples/benchmark.py            合成数据 CPU 跟踪耗时
tests/                           单元与场景测试
```

## 许可证

MIT，见 [LICENSE](LICENSE)。
