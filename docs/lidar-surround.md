# 原始点云全周检测：2026-10-09 实验

本轮已经实现原始 `.bin` 点云 → 全周 CenterPoint 检测框 → 车辆运动补偿 → 稳定 ID
跟踪 → 离线评估和可视化。**功能已跑通，精度验收失败，不替换现有默认方案，不建议上实车。**
这是固定官方权重的 Windows/PyTorch 推理适配，不是新的训练模型。适配源码和许可证见
[第三方说明](../THIRD_PARTY_NOTICES.md)。机器结果的可移植摘要见
[surround-results.json](surround-results.json)。

## 实现与验证

采用 OpenMMLab 的 `CenterPoint pillar02 + SECOND + SECFPN + circle NMS` nuScenes 权重。
权重下载地址、SHA256 固定，加载使用 `weights_only=True`、`strict=True`，414 个权重张量均加载，
不存在随机补全层。推理不读取 KITTI `label_02`；标注只由独立评分器读取。

用固定提交 `fe25f7a51d36e3702f961e198894580d83c4387b` 的上游类做数值审计，
以普通 PyTorch 工厂替代 MMCV 注册器：真实扫描上的 PFN、三层骨干、颈部、共享卷积和
六个任务头所有张量的最大绝对差均为 0；框解码也通过逐项对照。该检查不等于安装完整
MMCV CUDA 框架后的官方基准复现。CPU JIT 后端保持原始点与体素的首次出现顺序，不启用
fast-math；随机点、边界、容量截断、真实扫描与 NumPy/reference 体素结果逐项一致。
三条序列全部 898 个时间帧再次使用 JIT 推理，与 NumPy 版本的 261,952 个检测框的
类别、中心、尺寸、方向、分数逐项完全相同，最大数值差为 0。

输入转换为模型传感器约定 `(-y_KITTI, x_KITTI, z_KITTI)`，输出长度沿 yaw，使用重心
而不是底面中心。当前帧加最多四个过去扫描，跨度 0.4s，只用已经发生的 OXTS 位姿，
不读取未来点云；扫描缺失时显式空输入、不重复上次检测。KITTI 是 10Hz，训练模型使用
nuScenes 多扫描，扫描密度、传感器和场景仍有域差异。没有足够历史帧时直接使用已有帧，
不伪造额外扫描。KITTI 时间是名义帧间隔，不是真实采集时间，也未做逐点去畸变。

反射率乘 255 是当前域转换假设，可用 `--intensity-scale` 改变；它不是针对速腾雷达的
已验证标定。模型输出 heatmap sigmoid 只做一次，不能再经过旧 PointRCNN logit 的 sigmoid
转换，也不能沿用其 0.98 门限。Bicycle/Motorcycle 不强制映射成 KITTI 的 Cyclist。

## 效果

共处理 0000/0001/0005 的 898 个时间帧、894 个真实扫描。0001 的 177–180 帧原始包缺失，
新旧方案在定量评估中都排除。使用同一范围：当前 LiDAR 前方 ±40°、平面距离 50m 内，
Car/Pedestrian，局部 3D IoU ≥0.25。GT 和预测按相同框中心规则筛选。此范围是明确的共同
ROI，不是相机像素可见性判断；未实现 KITTI 2D DontCare 和官方难度划分，因此不是官方成绩。
**后方不计入精确率/召回率，也不把无标注的后方框当作误检。**

仅在 0000 上扫描高分门限 0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9/0.95，低分续接门限 0.1。
所有门限都未达到 96% 精确率要求。0.9/0.95 没有有效输出，精确率为未定义，不能算通过。
为展示可运行实验，按 0000 最大 F1 固定 0.7 后才评估 0001/0005；没有用留出序列重新选门限。

| 序列 | 用途 | 原方案 P / R | 全周实验 P / R | 原方案 / 实验 IDF1 |
| --- | --- | --- | --- | --- |
| 0000 | 门限选择 | 57.29% / 91.57% | 66.43% / 73.90% | 70.48% / 69.96% |
| 0001 | 留出验证 | 88.42% / 88.35% | 90.11% / 53.63% | 88.22% / 65.72% |
| 0005 | 留出验证 | 99.72% / 89.04% | 99.63% / 33.37% | 94.08% / 50.00% |

这个表与此前“所有有效 GT、三个类别”的报告分母不同，不能直接跨报告比较。
低门限输入框的召回在 0001/0005 分别为 95.62%/90.54%，但大量虚警迫使新 ID 创建门限
升高，跟踪召回随之损失。仅降低门限无法同时满足精准度和召回，需改变检测评分质量。
其中 0001 行人观察轨迹的召回只有约 7.48%，是明显弱项。

3s 人工全检测中断的共同 ROI 中，按旧方案固定两次可恢复机会作为分母：旧方案 2/2，
实验方案 0/2，原因是实验方案未在中断前形成对应轨迹。这不是自然遮挡验证；不能用
实验方案自己的零个可评目标宣称恢复成功，也不能拿此小样本证明真实 3s 保 ID 能力。

可视化现在包含原始点云、所有类别的前后检测框、正式 ID、候选与预测框。0001 第 120 帧
有 20 个后方显示轨迹（原方案一个后方预测轨迹），展示覆盖变化，不代表 20 个经真值
确认的正确检出。框内点着色只是几何叠加，不是语义分割。

## 耗时

RTX 4090、PyTorch 2.7.1+cu126、NumPy 1.26.4；在 0001 第 100–199 帧单独复测完整链路，
除去 4 个缺失扫描，96 帧，启动预热 5 帧不计入。包含读盘、源文件 SHA256、历史运动补偿、
体素化、GPU 推理、解码、坐标转换和跟踪，不包含 ROS 采集/通信/显示。

| 实现 | 总延迟中位数 | P99 | 超过 100ms |
| --- | --- | --- | --- |
| NumPy 体素化 | 227.28ms | 319.54ms | 96/96 |
| CPU JIT 体素化 | 71.93ms | 105.29ms | 3/96 |

NumPy 初始测试同时有点云资产读盘工作；JIT 最终探针独立运行，因此不能把此表看作严格的
仅一个变量微基准。JIT 版更接近 10Hz，但尾延迟仍未达标，且仅为这台桌面 GPU 的结果。
模型与体素/解码阶段中位数约 31.19ms，历史帧补偿约 29.39ms，框转换与跟踪约 9.96ms。

## 复现

以下命令在仓库根目录运行。原始 KITTI 目录只读；模型、环境和输出放到忽略目录。
不要修改核心跟踪环境，不设置 `KMP_DUPLICATE_LIB_OK` 绕过 DLL 冲突。

```powershell
conda env create --prefix ./outputs/envs/lidar-detector -f environment-lidar.yml
conda activate ./outputs/envs/lidar-detector
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e . --no-deps
python examples/download_centerpoint.py --references
$checkpoint = 'outputs/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
python examples/verify_centerpoint_port.py --numba --reference-root outputs/models/centerpoint/reference --checkpoint $checkpoint --scan C:/Projects/hx/Datasets/kitti/training/velodyne/0000/000000.bin --output outputs/port-audit.json
foreach ($sequence in @('0000','0001','0005')) {
  python -m radar_id_tracker.lidar_raw --dataset-root C:/Projects/hx/Datasets/kitti --sequence $sequence --output-dir "outputs/surround_cache/$sequence" --checkpoint $checkpoint
}
python examples/download_kitti_predictions.py --sequences 0000 0001 0005
python examples/benchmark_surround.py --dataset-root C:/Projects/hx/Datasets/kitti --detections-root datasets/kitti_tracking/detections --cache-root outputs/surround_cache --output-dir outputs/surround_report
python examples/review_surround.py --dataset-root C:/Projects/hx/Datasets/kitti --run-root outputs/surround_report
python examples/profile_surround.py --dataset-root C:/Projects/hx/Datasets/kitti --checkpoint $checkpoint --config outputs/surround_report/protocol.json --output outputs/surround_report/latency.json
python -m unittest discover -s tests
```

只运行新检测器和跟踪器，不进行旧方案对比时，可直接：

```powershell
python -m radar_id_tracker.lidar_raw --dataset-root C:/Projects/hx/Datasets/kitti --sequence 0005 --output-dir outputs/raw_cache_0005 --checkpoint $checkpoint --track-output-dir outputs/raw_tracks_0005 --tracker-config configs/lidar_centerpoint.experimental.json
```

上述配置是未通过验收的实验门限，不会自动加载；检测输出在 cache，带正式 ID 的输出在
tracking 目录，两者不得相同。

`review.html` 与 `pointclouds/` 保持在同一目录，直接本地打开即可播放。生成静态对比图需要
单独安装 Matplotlib（不影响核心或推理）：

```powershell
python examples/render_surround_preview.py --dataset-root C:/Projects/hx/Datasets/kitti --run-root outputs/surround_report --sequence 0001 --frame 120 --output outputs/surround_report/surround_preview.png
```

本机已经有独立环境，用 `conda run --no-capture-output --prefix
C:/Projects/hx/code/radar-id-tracker/outputs/envs/lidar-detector python ...` 即可，避免直接调用
未经激活的 Conda Python。命令默认 CPU JIT 体素化；用 `--voxel-backend numpy` 可回到
NumPy 参考路径；CPU 推理需明确 `--device cpu`，不能自动降级却报告 GPU 耗时。

## 接下来需要的数据与工作

先用有完整 360° 框和身份标注的数据，检验同一模型在原训练域的检测/跟踪表现及后方精度，
再确定是否更换更强的现成权重或做目标域微调。当前 KITTI 可继续前方回归，不能补足后方
真值。速腾实车之前还需要采样密度/反射率适配、实际时间戳和逐点去畸变，以及真实遮挡、
密集交会的保 ID 验证。性能侧继续减少历史点云搬运，并加入有界队列、过期帧处理和尾延迟
监测；这些与 ROS 包装分别验收。
