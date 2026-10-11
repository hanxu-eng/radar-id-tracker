# 官方 KITTI 检测器对照：接入与评测准备

本轮按用户选择只完成接入与评测准备，不安装 WSL/Ubuntu，不执行真实模型
GPU 推理、正式性能评价或实车验收。默认检测器与跟踪配置没有替换。

后续目标环境已明确为原生 Ubuntu 24.04 x86_64。完整安装与运行步骤见根目录
README 的 Ubuntu 24.04 部分；下面的 Windows/WSL 命令只保留为本次准备记录。
新增独立环境配方、安装脚本和实际 GPU 检查；本次发布的 CPU 代码检查与目标
Ubuntu GPU 编译/精度验收分别报告。

## 模型与实际下载状态

官方代码在 `outputs/vendor/OpenPCDet`，固定提交
`233f849829b6ac19afb8af8837a0246890908755`。适配器调用原始网络、体素化和
后处理，没有重新实现稀疏卷积、ROI 池化或学习骨干。源文件和权重校验值写入
`outputs/models/openpcdet/assets.json`；这些大文件不提交 Git。

| 模型 | 当前状态 | 实际文件大小 | 本轮用途 |
| --- | --- | ---: | --- |
| SECOND-IoU | 官方链接下载成功 | 47,853,439 字节 | 主对照 |
| Part-A2-Free | 官方链接无法下载 | — | 保留失败记录与接入支持 |
| PV-RCNN | 官方链接下载成功 | 52,574,589 字节 | 替代的两阶段检测器对照 |

SECOND-IoU SHA256：`bf839864e7f46e004ef57c722c1f8260173e26aecf07303f180519f4f6b80a91`。

PV-RCNN SHA256：`08552932026855bcc10f66dbc45409f139d3208fd0095c2aab1f459f055fe463`。

校验值在首次从官方链接下载后本地计算，用于后续变更检测；它们不是官方签名。
下载不等于网络已成功初始化或性能验收通过。Part-A2-Free 失败响应保留在清单里，
没有使用未知第三方权重替代。

官方来源：[模型库](https://github.com/open-mmlab/OpenPCDet#model-zoo)、
[安装说明](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/INSTALL.md)、
[原始点云 Demo](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/demo.py)。

## 输入与接口约定

- 使用官方 DemoDataset 的单帧 `Nx4` 输入：KITTI LiDAR x 前、y 左、z 上，反射
  强度不放大。不套用 CenterPoint 的坐标旋转、多帧累积或强度乘 64。
- 保留官方配置的前向范围 `[0,-40,-3,70.4,40,1]`、体素大小、体素上限、分数
  筛选和 NMS。Demo 路径没有实现 KITTI Dataset 的相机 FOV 裁剪；本轮明确记录
  这一差异，本地也未下载图像。此对照不是官方 KITTI 成绩复现。
- 类别索引 1/2/3 映射 Car/Pedestrian/Cyclist。框采用几何中心；长宽高、yaw 和
  分数保留官方含义，不补半个高度，也不再次做 sigmoid。
- 框按外参与当帧 OXTS 变换到现有 ENU 跟踪坐标。推理不读取目标标签和未来
  扫描；缺失扫描输出空检测，不复用上一帧框。
- 检查源代码、权重、三类配置和模型状态张量；缺失张量将阻止运行，不采用官方
  Demo 的宽松权重加载。严格模型加载仍待 Linux 环境实际验证。
- 缓存兼容现有回放、前向 ROI 评价和点云可视化。torch/pcdet 仅在创建检测器时
  导入，不增加普通跟踪器安装的运行依赖。

## 固定的评价方案

`configs/openpcdet_comparison.experimental.json` 固定此前已审查的 17 条完整开发
序列。0010/0015/0019 不加入此次对照，0020 继续保留。官方模型与 CenterPoint
基线使用同一跟踪配置：高分 0.7、低分 0.1、身份保留 4 秒，以及现有候选容量和
重复建轨抑制设置。此轮不调整阈值或重新训练。

不同模型的分数尺度不同，先比较检测 PR 曲线、AP、精度≥96% 时最大召回，再
解释固定跟踪门槛下的损失。曲线受官方初始分数截断及 NMS 限制，不声称保存了
全部候选或完成概率校准。

使用一致的 Car/Pedestrian、50 米、前向左右 40° ROI，报告 IoU 0.25 和 0.5
的检测/观察结果、按类别 IDF1 与 ID 切换。30 帧中断作为三秒恢复测试；优先从
帧 50 开始，短序列固定使用 `min(50, 帧数-30-5)`，保留至少两帧中断前数据和
五帧返回窗口。恢复分母来自本轮同配置 CenterPoint 基线，候选没有中断前轨迹
也计失败；它不自动等于此前模型/策略实验的 23 个目标。

KITTI Object 预训练与 Tracking 场景是否重叠尚未审计，因此结果只能作为开发
对照，不能证明独立泛化、后方检测、自然遮挡恢复或速腾实车精度。评价不采用
KITTI 官方难度筛选与完整二维 DontCare 规则，不是官方 AP_R40/HOTA/MOTA。

## 本机准备与复现

独立下载工具环境为 `outputs/envs/openpcdet-tools`，包含 gdown 5.2.0、NumPy
1.26.4。已有检测环境依赖未修改。下载、清单校验和数据清点可在 Windows 运行：

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
& outputs/envs/openpcdet-tools/Scripts/python.exe examples/prepare_openpcdet.py --download --models second_iou pv_rcnn
& outputs/envs/lidar-detector/python.exe examples/benchmark_openpcdet.py --stage prepare --dataset-root C:/Projects/hx/Datasets/kitti --baseline-manifest outputs/kitti_sequence_calibration_grouped/cache_manifest.json --output-dir outputs/kitti_official_detector_comparison
```

`protocol.json` 冻结配置、当前工作区推理/评价源文件、基线缓存、外参与位姿、
扫描帧清单和中断位置。`preparation.json` 明确记录未运行模型和性能评价。准备
过程不读取标签或保留序列。冻结协议有变化时必须使用新的实验目录。

本次清点为 17 条序列、5,442 帧，其中 5,438 帧有原始点云；0001 缺少四帧扫描。
0012 总长 78 帧，其固定中断从帧 43 开始。准备报告和冻结协议位于
`outputs/kitti_official_detector_comparison/`。

本机有 RTX 4090 和现有 PyTorch CUDA 运行时，但没有可用 Linux/WSL 环境、
CUDA 编译工具链或已编译的 OpenPCDet 算子。用户本轮选择暂不安装。以后运行
模型，应先准备独立 Linux CUDA 环境，安装与 torch/CUDA 匹配的 spconv，并按
官方说明编译算子。固定版本的数据集注册还会导入 `av2`，应同时核对该依赖。
这些系统及运行时步骤本轮未执行。

环境验证通过后的待执行命令：

```bash
# WSL 示例，前提是已经安装官方运行时并完成 CUDA 算子构建。
cd /mnt/c/Projects/hx/code/radar-id-tracker
export PYTHONPATH="$PWD/src"
python examples/benchmark_openpcdet.py --stage infer --dataset-root /mnt/c/Projects/hx/Datasets/kitti --baseline-manifest outputs/kitti_sequence_calibration_grouped/cache_manifest.json --output-dir outputs/kitti_official_detector_comparison
python examples/benchmark_openpcdet.py --stage evaluate --dataset-root /mnt/c/Projects/hx/Datasets/kitti --baseline-manifest outputs/kitti_sequence_calibration_grouped/cache_manifest.json --output-dir outputs/kitti_official_detector_comparison --with-pointclouds
```

校验通过的完整推理缓存可继续使用，只有 `.part` 的中断输出不能视为完成。评价
前需要两个模型在全部 17 条序列的完整缓存，并核对与基线的原始扫描 SHA256。
目标标签只在明确的 `evaluate` 阶段读取。

评价将生成 `benchmark.json`、`detection_curves.json`、`observed_counts.json`、
`fixed_recovery.json` 和双栏 `review.html`；`--with-pointclouds` 加入原始点云。
这些评价阶段的文件目前都尚未生成。

## 验证范围

新增七项测试通过，覆盖几何中心和坐标的已知答案、类别与分数约定、非法输出
拒绝、标签隔离、单帧输入与缺失扫描、权重变更拒绝、保留序列和冻结协议，以及
人工缓存的完整离线评价/恢复/双模型可视化生成。人工缓存测试不执行神经网络，
不代表真实模型准确率。

另有 29 项已有原始缓存、KITTI 适配器和 LiDAR 跟踪器回归测试通过。真实模型
初始化、CUDA 算子、推理延迟、行人检测收益和浏览器视觉验收仍待后续完成。
