# 输入适配与跨序列泛化检查：2026-10-09

本轮实现了独立输入配置、按序列分组的参数选择、留一序列检查、冻结后的独立验证，
以及带原始点云和误检／漏检标注的对照图。**输入适配候选的泛化验收失败，保留旧默认，
不建议上实车。** 数据摘要见 [lidar-input-results.json](lidar-input-results.json)。

## 避免针对单个序列调参

最初在 0000 上对照反射率倍率 1/16/64/128/255，以及两种单扫描设置，共七组输入。
乘 16 在该序列上的 F1 最好。收到跨序列要求后，停止单序列发布路径；其阶段性结果
仅留在本机 `outputs/kitti_surround_iteration2`，不作为最终参数选择依据。

改用 0000/0001/0005 三个已查看序列作为开发集，对五种倍率进行完整的 3×5 对照；
各组保持 5 帧扫描、同一官方权重、同一 0.7 新 ID 门限和 0.1 续接门限。
在线检测与跟踪不读取标注；开发参数选择由独立离线评分完成。没有按序列或 GT ID
设置例外，没有再用验证序列选择分数门限。

选择先要求**每条**训练序列精确率 ≥96%、召回 ≥85%；没有组合满足。
失败后仅为复现实验，优先最大化最差序列 F1，再比较各序列等权平均 F1；长序列不能
通过大量样本掩盖短序列失败。三次留一序列都选择倍率 64，最终也选择 64、5 帧。
留一时仅用另外两条序列选择参数，留出序列只评分。这是参数敏感性检查，样本很少，
不能代替独立验证或证明跨传感器泛化。

在读取 0002/0006 验证结果之前，将配置、选择过程、开发结果校验和、实现校验和写入
`outputs/kitti_surround_generalization/protocol.json`。0002/0006 未参与本轮选择；它们的
结果不回流到参数搜索。以后这两条序列应作为已查看回归数据，不能重复称作新盲测。

## 冻结配置的结果

五序列共 1,401 个时间帧、1,397 个真实扫描；0001 的 177–180 帧缺失，两侧均排除。
范围与上一轮一致：当前 LiDAR 前方 ±40°、50m 内 Car/Pedestrian、3D IoU ≥0.25。
不是相机可见性判断，未实现 2D DontCare／难度划分，也不是
[KITTI 官方的 2D 跟踪指标](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)。
后方框没有用于精确率和召回率计算。

表中的“原输入”是上一轮 CenterPoint 的倍率 255、5 帧；两侧跟踪策略相同，
**不是** PointRCNN 与 CenterPoint 的比较。

| 序列 | 用途 | 原输入 P / R | 候选 P / R | 原输入 / 候选 ID 切换 |
| --- | --- | --- | --- | --- |
| 0000 | 开发 | 66.43% / 73.90% | 70.09% / 95.98% | 0 / 0 |
| 0001 | 开发 | 90.11% / 53.63% | 90.74% / 71.99% | 0 / 2 |
| 0005 | 开发 | 99.63% / 33.37% | 98.55% / 50.68% | 0 / 0 |
| 0002 | 独立验证 | 82.13% / 50.26% | 87.99% / 51.68% | 0 / 0 |
| 0006 | 独立验证 | 88.46% / 80.80% | **38.96% / 98.36%** | 0 / 1 |

0006 明显退化，不能用总体召回提高宣布成功。其中候选有 627 次行人输出未匹配有效
前方 GT；该序列在评分范围内没有有效行人 GT，不能把行人召回写成 100%。另一方面，
0002 有 167 次有效行人 GT，但候选观察轨迹一次也未匹配；统一升高行人门限会进一步
恶化已经存在的行人漏检，不能据单条失败序列这么修补。所有数字都是上述局部评分规则
下的结果，不代表这些框都经过人工或官方 DontCare 规则确认。

0001 候选行人召回从上一轮约 7.48% 提高到 31.78%，仍远低于要求。0006 候选的 Car
精确率／召回为 93.13%／98.36%，总体精度下降主要来自行人类别。

3s 全检测中断保留每条序列第 50–79 帧的测试。以原输入 CenterPoint 自己的可恢复目标
为固定分母时，没有有效机会，不能宣称恢复率通过。补充沿用上一轮 PointRCNN 的两个
已固定目标作为开发集参考：候选恢复 1/2，上一轮原始全周输入为 0/2，PointRCNN 为 2/2。
这个参考只展示进展，样本很小，且是人工全检测中断，不能证明真实遮挡 3s 的保 ID 能力。

## 本轮改动

- `RawLidarConfig` 将扫描数、帧周期、反射率倍率、检测输出门限和体素后端保存为显式
  JSON。配置与命令行输入参数禁止混用，避免无声覆盖；未传配置时仍沿用上一轮默认。
- 在线模块先检查参数，再加载 GPU；拒绝未来扫描、重复或逆序历史、非法强度／周期，
  保持原始点云不变。缓存记录实际输入设置与校验和。
- 延迟工具读取冻结输入配置；缺帧时按真实帧跨度保留仍有效历史，和正式推理一致。
- 点云对照保持全周显示。PNG 可开启前方 TP／FP／MISS／忽略框标注，数字为整个序列
  的观察轨迹成绩，后方和范围外仍明确不评分。HTML 包含五条序列、自然回放和中断回放，
  共 1,397 个真实点云资产；HTML 完成静态数据与资产检查，未做浏览器渲染验证。

`configs/lidar_centerpoint_kitti_input.experimental.json` 仅用于复现**已失败**的 64 倍率
实验，不自动加载，也不是速腾反射率标定。本轮查验的一个官方 nuScenes 示例点云强度
范围为 0–255、中位数 12；一个样例不能定义整个数据域的分布，更不能证明固定倍率
适用于任何雷达。原始示例来自固定提交的
[OpenMMLab 示例目录](https://github.com/open-mmlab/mmdetection3d/tree/fe25f7a51d36e3702f961e198894580d83c4387b/demo/data/nuscenes)，
源文件 SHA256 为 `5f8f9b1b199ceff7d41cd319021a7a7b02dcd44d41f622a9e65a6a4a6be3cbdb`。

从本轮暂存文件和基线创建干净源码快照，构建并安装 wheel 后执行 92 项测试：91 项通过、
1 项因核心环境没有 Numba 而跳过；独立推理环境的 JIT／输入配置／点云边界测试
17 项全部通过。无 PyTorch 的 CLI 帮助和许可证打包检查也通过。原有未提交的二维
RadarScenes 修改未加入本轮源码快照或发布。

## 复现

使用上一轮建立的独立 GPU 环境，在仓库根目录执行；原始数据保持只读。完整运行需要
新的输出目录；已有合法缓存可复用。第一条命令可先不传 `--reuse-ablation`，它会重新
推理开发集；本机已有 0000 缓存时可传最初的 `input_ablation.json` 复用。

```powershell
$model = 'outputs/models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
$envPath = 'outputs/envs/lidar-detector'
$run = 'outputs/kitti_surround_generalization_replay'
conda run --no-capture-output --prefix $envPath python examples/ablate_surround_inputs.py --dataset-root C:/Projects/hx/Datasets/kitti --checkpoint $model --output-dir $run --sequences 0000 0001 0005 --baseline-cache-root outputs/kitti_surround_cache_numba
conda run --no-capture-output --prefix $envPath python examples/benchmark_surround_inputs.py --dataset-root C:/Projects/hx/Datasets/kitti --checkpoint $model --ablation "$run/input_ablation.json" --baseline-cache-root outputs/kitti_surround_cache_numba --output-dir $run --validation-sequences 0002 0006
conda run --no-capture-output -n radar-id-tracker python examples/review_surround.py --dataset-root C:/Projects/hx/Datasets/kitti --run-root $run --comparison input
conda run --no-capture-output -n radar-id-tracker python examples/summarize_surround_inputs.py --run-root $run --output "$run/results.json"
conda run --no-capture-output -n base python examples/render_surround_preview.py --dataset-root C:/Projects/hx/Datasets/kitti --run-root $run --sequence 0006 --frame 120 --comparison input --show-errors --output "$run/failure_0006.png"
```

仅复现单条冻结实验时，在线命令增加
`--inference-config configs/lidar_centerpoint_kitti_input.experimental.json`；与既有
`--tracker-config configs/lidar_centerpoint.experimental.json` 分开指定。

## 下一步取向

本轮没有更改检测模型权重。固定倍率只改善部分输入域差异，无法同时解决分类虚警、
行人召回和身份稳定。下一步应使用覆盖多场景／不同点密度的训练与校准数据改善检测
本身，按完整序列分组训练／验证，另保留尚未查看的序列做最终测试；避免按序列定门限。
车辆和行人分别验收，持续保留所有失败案例。完整后方准确率还需要 360° 标注数据，
KITTI 前方评分和本机点云画面不能代替；跨速腾雷达的泛化仍需对应传感器验证。
