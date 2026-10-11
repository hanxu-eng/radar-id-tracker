# Radar ID Tracker：10 Hz 雷达／激光雷达稳定编号

目标工作环境为 **Ubuntu 24.04 LTS x86_64 + NVIDIA 独立显卡**。车载速腾激光雷达
主路径是“原始点云 → 3D 检测 → 固定坐标系变换 → 稳定 ID 跟踪”。下面给出从
新电脑克隆、安装、生成基线到完整序列评价的步骤。Ubuntu 的系统 Python 保持不变，
检测程序使用独立 Python 3.11 环境。ARM/Jetson 需要另外适配依赖，不套用此配方。

**当前状态：接口与 CPU 测试已实现，官方模型权重可下载；Ubuntu GPU 编译、真实
检测精度和全链路 10 Hz 尚待目标电脑验收。当前不能据此认定可以上实车。**

发布前从 Git 暂存区导出的干净版本执行了 134 项 CPU 测试：127 项通过，7 项因
可选依赖缺失而跳过；安装脚本通过 Bash 语法检查。仓库另有 Ubuntu 24.04 /
Python 3.11 自动检查，覆盖安装、测试及评测命令入口，不运行 GPU 模型。

## Ubuntu 24.04 环境与版本

| 组件 | 本轮配置 | 作用 |
| --- | --- | --- |
| Ubuntu | 24.04 LTS / x86_64 | 目标运行系统 |
| Python | 3.11，独立 Conda 环境 | 避开系统 Python 3.12 与选定 spconv wheel 的差异 |
| CUDA Toolkit | 12.6 | 编译 OpenPCDet CUDA 算子，必须有 `nvcc` |
| PyTorch / torchvision | 2.7.1 cu126 / 0.22.1 cu126 | 检测网络运行 |
| NumPy | 1.26.4 | 固定旧检测代码使用的数组接口 |
| spconv | spconv-cu120 2.3.6 | 使用已发布的 Python 3.11 Linux 预编译包 |
| OpenPCDet | `233f849829b6ac19afb8af8837a0246890908755` | 固定官方模型实现 |
| 检测模型 | SECOND-IoU、PV-RCNN | 官方 KITTI Car/Pedestrian/Cyclist 权重 |

CUDA 12.6 官方支持 Ubuntu 24.04；spconv 官方允许同一 CUDA 主版本的不同次版本
组合，所以使用 cu120 spconv 和 cu126 torch。但此组合仍需要下面的实际算子和模型
检查，安装成功不能代替运行验收。来源：[CUDA 安装指南](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html)、
[spconv 安装说明](https://github.com/traveller59/spconv#install)、
[PyTorch 官方版本](https://pytorch.org/get-started/previous-versions/)。

### 1. 克隆本轮分支并安装系统工具

```bash
git clone --branch codex/ubuntu2404-openpcdet https://github.com/hanxu-eng/radar-id-tracker.git
cd radar-id-tracker
sudo apt update
sudo apt install -y build-essential git curl ca-certificates libgl1 libglib2.0-0t64
nvidia-smi
```

`nvidia-smi` 应能显示显卡与驱动。驱动安装按 Ubuntu/NVIDIA 的硬件推荐完成，
必要时重启电脑后再检查。不要将 `nvidia-smi` 显示的 CUDA 支持版本当作已安装
Toolkit 的版本；编译工具以 `nvcc --version` 为准。

未安装 CUDA Toolkit 12.6 时，使用 NVIDIA 的 Ubuntu 24.04 软件源：

```bash
curl -fL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/radar-cuda-keyring.deb
sudo dpkg -i /tmp/radar-cuda-keyring.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
```

使用 `cuda-toolkit-12-6` 安装编译工具；显卡驱动独立管理。网络受限时可以先按
NVIDIA 官方指南安装 Toolkit，再继续下面的项目安装。

### 2. 创建独立环境与安装检测器

先安装 [Miniforge/Conda](https://github.com/conda-forge/miniforge)。在仓库根目录：

```bash
conda env create -f environment-openpcdet-ubuntu2404.yml
conda activate radar-openpcdet-ubuntu2404
python --version
bash scripts/setup_openpcdet_ubuntu2404.sh
```

脚本检查 Ubuntu、架构、Python 与 Toolkit，安装固定直接依赖，下载固定版本
OpenPCDet 及两套官方权重，编译 CUDA 扩展，执行运行检查并保存依赖快照。脚本
不调用 sudo，不更改系统 Python，也不读取 KITTI 标签。默认编译当前可见显卡
架构，使用两线程；显存不影响编译内存消耗，内存不足时可设置 `MAX_JOBS=1`。

```bash
MAX_JOBS=1 bash scripts/setup_openpcdet_ubuntu2404.sh
python -m unittest discover -s tests -v
```

运行检查结果在 `outputs/ubuntu2404/runtime_check.json`，环境快照在
`outputs/ubuntu2404/pip-freeze.txt`。只有检查 JSON 中 `passed: true` 才进入数据集
推理。检查包括两个模型的完整权重加载、CUDA IoU/NMS、点落框判断和稀疏卷积，
尚不代表数据集检测精度通过。需要用一个真实扫描再验证完整网络前向：

```bash
python examples/check_openpcdet_runtime.py --scan /data/kitti/training/velodyne/0000/000000.bin --output outputs/ubuntu2404/real_scan_check.json
```

### 3. 准备 KITTI Tracking 数据

数据可以放在任意位置；后续例子使用 `/data/kitti`，请改为实际目录。至少应有：

```text
/data/kitti/
  training/
    velodyne/0000/000000.bin
    calib/0000.txt
    oxts/0000.txt
    label_02/0000.txt
```

需要 Tracking 的训练点云、标定、OXTS 和训练标签。标签仅用于最后的离线评价；
推理与跟踪不读取它。点云和 OXTS 必须保留完整帧编号，缺失扫描保留为空检测，
不要通过重新编号消除缺帧。官方入口：[KITTI Tracking](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)。

### 4. 在 Ubuntu 生成可搬迁的 CenterPoint 对照基线

新克隆不会包含 Windows 电脑的 `outputs/`。下面生成自己的基线，不依赖
`C:\Projects\...` 或 WSL `/mnt/c`：

```bash
python examples/download_centerpoint.py
python examples/prepare_centerpoint_baseline.py \
  --dataset-root /data/kitti \
  --checkpoint outputs/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth \
  --output-dir outputs/ubuntu2404/centerpoint_baseline
```

固定五帧因果扫描、强度乘 64、0.1 秒标称周期和 0.1 检测门槛。它复现当前
CenterPoint 对照配置，不表示其精度合格。生成的 `cache_manifest.json` 使用相对于
清单目录的路径，复制整个基线目录即可迁移。禁止让评价标签或真值 ID 进入基线。

如果复制了旧电脑的整个 `outputs/`，可以使用旧清单，并为评测提供
`--cache-root /实际仓库/outputs` 重定位旧绝对路径；原始输入、缓存和源代码校验
仍需通过。推荐新环境重新生成基线，以保留实际运行时信息。

### 5. 固定方案、推理、评价

所有命令在激活的检测环境、仓库根目录执行：

```bash
python examples/benchmark_openpcdet.py --stage prepare \
  --dataset-root /data/kitti \
  --baseline-manifest outputs/ubuntu2404/centerpoint_baseline/cache_manifest.json \
  --output-dir outputs/ubuntu2404/comparison

python examples/benchmark_openpcdet.py --stage infer \
  --dataset-root /data/kitti \
  --baseline-manifest outputs/ubuntu2404/centerpoint_baseline/cache_manifest.json \
  --output-dir outputs/ubuntu2404/comparison

python examples/benchmark_openpcdet.py --stage evaluate \
  --dataset-root /data/kitti \
  --baseline-manifest outputs/ubuntu2404/centerpoint_baseline/cache_manifest.json \
  --output-dir outputs/ubuntu2404/comparison --with-pointclouds
```

配置在 `configs/openpcdet_comparison.experimental.json`：固定 17 条完整开发序列，
0020 不打开；两个官方模型与 CenterPoint 使用同一跟踪配置，单独比较检测 PR 和
固定门槛的跟踪效果。没有按序列调整阈值。推理与评价是独立阶段。

准备阶段只清点、校验并冻结协议。推理阶段可继续使用完整且校验通过的缓存；
只有 `.part` 的目录需要先保留排错证据，再另选目录重跑。源代码、配置或模型发生
变化时选择新的实验目录；已生成的评价目录不自动覆盖。

| 输出 | 内容 |
| --- | --- |
| `protocol.json` / `preparation.json` | 冻结配置、来源、输入清单及准备状态 |
| `raw_cache/` | 原始点云检测结果，包含位姿、帧号、分数与校验值 |
| `benchmark.json` | 完整序列、自然回放及三秒中断的逐项结果 |
| `detection_curves.json` | 类别检测 PR/AP 与高精度条件下的召回 |
| `observed_counts.json` | IoU 0.25/0.5 的类别精度、召回、IDF1 与 ID 切换 |
| `fixed_recovery.json` | 固定基线目标分母的三秒同 ID 恢复 |
| `review.html` + `pointclouds/` | 双栏原始点云、框、ID 与离线真值叠图 |

打开 `outputs/ubuntu2404/comparison/review.html`，场景名区分 SECOND-IoU 和
PV-RCNN。保留同目录 `pointclouds/`；这些输出和权重都不上传 Git。

### 6. 实机接入与验收边界

当前 KITTI 官方模型是前向检测对照，不能用来证明车后方的召回。KITTI Object
预训练与 Tracking 场景是否重叠仍需审计，本地 AP/IDF1 也不是 KITTI 官方成绩。
实车下一步需要速腾实测数据、真实采集时间戳、点云去畸变、定位噪声与坐标重置
验证，并测量检测到追踪的全链路 P99 延迟、误检、自然遮挡和错误身份恢复。

ROS 接入应读取当帧点云、当帧位姿和真实时间戳，将框变换到同一固定坐标系后
调用 `LidarTracker.update(timestamp_s, detections)`。当前仓库尚未提供完整 ROS 节点。
ROS 的系统 Python 与检测 Conda Python 分别管理，不能直接把不同 Python 版本的
二进制 `rclpy` 放进检测环境。

常见排错：

- `nvcc` 找不到或版本不符：检查 `CUDA_HOME`、PATH 和 Toolkit 12.6，不能只装
  PyTorch wheel。升级驱动不等于安装了 CUDA 编译工具。
- `CUDA unavailable`：检查 `nvidia-smi` 和当前 torch wheel；先解决驱动/GPU 可见性。
- `av2`、`SharedArray`、`spconv` 缺失：在激活的环境重新执行安装脚本，不使用系统 pip。
- CUDA 扩展编译失败：查看最早的编译错误和 GCC/nvcc/torch 版本，内存不足时减小
  `MAX_JOBS`，不要以跳过算子或丢弃权重张量来绕过错误。
- 协议校验失败：核对数据、代码、模型和缓存来源，使用新的实验目录保留旧证据。
- GPU 检查成功但召回差：查看检测 PR、几何误差和确认阶段损失，再决定迭代位置。

### 7. 在 Ubuntu 拉取与推送

公开克隆不需要登录；推送需要在 Ubuntu 上配置自己的 GitHub SSH 或凭据，本机
Windows 的凭据不会随仓库迁移。保持实验分支和提交范围清楚：

```bash
git status
git pull --ff-only
git switch -c codex/my-iteration
# 修改、运行检查后，逐个添加相关文件
git add README.md path/to/changed_file.py
git diff --cached
git commit -m "Describe the change and validation"
git push -u origin HEAD
```

工作区有未保存改动时先检查并保存自己的修改再拉取。数据集、学习权重、特征缓存、
视频及运行输出保持在 Git 忽略目录，避免意外发布。

## 既有实验与其他平台说明

新增 [官方 OpenPCDet 检测器接入与评测准备](docs/lidar-official-detectors.md)：
SECOND-IoU / PV-RCNN 官方权重已下载，固定完整序列对照协议；真实模型推理与
性能评价待 Linux CUDA 算子环境准备完成后执行。

车载速腾激光雷达使用新增的 **3D 检测框跟踪器 `LidarTracker`**，输入固定坐标系下的
中心、类别、尺寸和方向，不需要多普勒。已实现车辆位姿补偿、短时预测、4 秒身份保留、
保守恢复及独立 KITTI 数据集评分。复现命令、实测结果和已发现的问题见
[激光雷达数据集测试](docs/lidar-kitti.md)。

```powershell
conda run --no-capture-output -n radar-id-tracker python examples/download_kitti_tracking.py
conda run --no-capture-output -n radar-id-tracker python examples/benchmark_kitti.py --stress-grid --output-dir outputs/kitti_final
```

结果在 `outputs/kitti_final/review.html`。上面的回放使用公开 PointRCNN 激光检测框。
新增 [原始点云全周 CenterPoint 推理实验](docs/lidar-surround.md) 已完成数据集回归与本机
延迟测量，但精度未通过验收；实机时间同步仍待验证。以下原有 `RadarTracker` 文档说明
二维毫米波雷达接口；激光雷达请按上面的专门文档接入。

后续 [输入适配与跨序列泛化检查](docs/lidar-input-validation.md) 已完成三个开发序列的
留一检查和两条冻结验证序列。候选在 0006 出现明显精度退化，继续保留默认；报告和
点云对照包含失败案例，不按单条序列继续调门限。

新增原始点云可视化：894 帧真实点云与三维框、ID、类别和预测状态同步叠加；
支持俯视／三维斜视、旋转、缩放和全量点显示。准备命令如下，生成后打开同一个 `review.html`：

```powershell
conda run --no-capture-output -n radar-id-tracker python examples/download_kitti_pointclouds.py
conda run --no-capture-output -n radar-id-tracker python -m radar_id_tracker.pointcloud_review
```

点云按帧加载，请保留 HTML 同目录下的 `pointclouds/` 文件夹。

本机完整 KITTI Tracking 数据集已接入 `C:\Projects\hx\Datasets\kitti`，
对应重测与点云叠图在 `outputs/kitti_local/review.html`；外部数据地址和检测预测目录
可分别配置，命令见 [本机完整数据集说明](docs/lidar-kitti.md#本机完整数据集2026-10-09)。

按类别分数门槛与连续中分检测确认现已支持显式 `--config`，待确认框独立显示，旧身份
恢复门槛独立控制。[实现、复现与验收报告](docs/lidar-score-policy.md)记录六序列对比和
固定恢复测试。实验参数未通过总体精度、召回及 ID 切换验收，**旧默认策略保留**。

本机 Git 同步与 Conda 运行方式见 [本机工作流程](docs/local-workflow.md)。
新增 [整序列分数校准实验](docs/lidar-sequence-calibration.md)：检测训练、校准、评价
三种角色隔离，17 条完整开发序列复验。中分确认提升召回，但精度及 3s 恢复退化，
仍保留默认；原始点云对照和逐类别失败均已记录。
随后 [行人框几何实验](docs/lidar-pedestrian-geometry.md) 使用同预算原 L1 对照，
新损失未改善严格行人召回及 3s ID 恢复；完整失败和分序列点云展示已保留。
`outputs/` 首轮效果审查、新实验档和多序列复验结果见
[2026-09-30 迭代报告](docs/iteration-2026-09-30.md)。

本仓库是一个在线多目标跟踪器。每收到一帧雷达**检测结果**，它预测已有目标的位置，用位置与多普勒速度进行匹配，并输出持续使用的 `track_id`。标称输入频率为 10 Hz，即约每 0.1 秒处理一帧；程序无需等待凑齐 10 帧。

二维雷达核心方法：直行/加速双模型 IMM 滤波、位置与多普勒门控、分级 Hungarian 匹配、轨迹确认、短时漏检预测和休眠恢复。跟踪器在 CPU 上运行。RTX 4080 可以运行上游检测模型，仓库尚未包含训练好的 CenterPoint 或 TensorRT 引擎。

## 1. 在另一台电脑上准备环境

以下安装步骤面向 Linux、macOS、Windows；建议使用 Python 3.11 的 Conda 环境。本项目只需 NumPy，不要求 CUDA、PyTorch 或 NVIDIA 驱动就能运行示例跟踪器。自动化测试已在 Linux CI 和本机 Windows Conda 环境运行，其他目标电脑仍需按下文测试确认。

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

数据下载完后，按 [RadarScenes 实际帧率接入与首轮测试](docs/radarscenes.md) 直接运行本仓库的无标签逐帧聚类基线。它能生成 `detections.jsonl`、`tracks.jsonl` 和 `summary.json`；加 `--visualize` 还能生成可直接在浏览器打开的 `visualization.html` 和逐帧 SVG，查看原始点、聚类中心、稳定 ID、预测态与轨迹尾迹。如果已下载 `camera/`，再加 `--with-camera` 可并排显示时间最近的相机图。这便于先验证真实数据的读取、时间戳、传感器位置和 ID 输出；它不是已训练检测器，也不能据其输出宣称真实场景精度达标。

对已有 RadarScenes 回放结果可运行独立的 `python -m radar_id_tracker.radarscenes_evaluate --scenes /path/to/scenes.json --run-dir outputs/sequence_137_sensor1_vis`，生成候选检测、已观测轨迹与全部公开轨迹的分项评分及逐帧错误清单。评分才读取官方真值，**不会反向影响检测或跟踪**；它是固定中心距离门限的诊断，不是官方榜单指标。命令、口径与解释见上述文档。

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

核心模块处理检测结果到稳定 ID，另有可选的原始激光点云 CenterPoint 推理路径，见 [全周检测实验与复现](docs/lidar-surround.md)。[点云框质量评分](docs/lidar-box-quality.md) 改善部分车辆召回；[首轮检测头微调](docs/lidar-head-adaptation.md) 出现严重行人漏检。最新的 [扩大训练覆盖实验](docs/lidar-training-coverage.md) 增至 92 个标注行人实例，在三条新序列恢复部分召回，但精度、严格几何及 3s ID 恢复仍未通过，候选未采纳、默认配置不变。推理依赖、权重下载与核心跟踪环境分开；质量评分训练使用独立依赖，在线只增加 NumPy 模块。仓库提供可选的冻结骨干检测头微调实验，完整检测器训练、外参标定及 TensorRT 引擎仍需补充。实际部署建议采集、检测、跟踪分阶段异步运行，以采集时间戳排序，并限制等待队列长度，避免处理过期帧。接入接口为上文 JSONL 格式或 `Detection` 对象。

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

跟踪核心采用 MIT，见 [LICENSE](LICENSE)。CenterPoint 适配代码遵循上游 Apache-2.0，见 [第三方说明](THIRD_PARTY_NOTICES.md)；权重与数据分别遵循提供方条件。
