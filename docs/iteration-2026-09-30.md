# 2026-09-30：RadarScenes 效果审查与第一轮迭代

本机原始输出中的 `sequence_137` 前 200 帧只有 3 个真值身份，但发布了
15 个预测身份。问题包括同一目标被拆成多团、单点扫描被丢弃、杂波被确认成轨迹。
`sequence_1` 前 200 帧全部标为静态点，原先没有公开 ID 属于合理结果；
该完整序列后面才出现动态目标。

## 改动

保留原始默认 `baseline`，新增显式 `--profile stable` 实验档：

| 项目 | baseline | stable |
| --- | --- | --- |
| 空间连通距离 | 2 m | 3 m |
| 新 ID 确认 | 0.3 s 内两次高分观测 | 0.3 s 内三次高分观测 |
| 少于 `min_points=2` 的团 | 丢弃 | 0.35 分弱候选，只能延续已确认轨迹 |
| 多点候选分数 | 0.75 | 0.75 |
| 最小绝对径向速度、多普勒距离 | 0.5 m/s、1.5 m/s | 相同 |
| 跟踪位置噪声 | 0.6 m | 相同 |

不读取标签来检测、跟踪或选择当前帧候选。真值只由独立评分阶段读取。
新增 `birth_eligible_detections` 单独评分可创建 ID 的候选；`detections`
仍评分全部候选，包括弱点。弱点会使候选总虚警上升，不能把两种口径混用。
所有分数仍是本项目的中位中心距离诊断，不能当作官方榜单或生产精度。

## 原有片段的对照

相同 `sequence_137`、sensor 1、前 200 次扫描、2 m 距离门限：

| 指标 | baseline | stable |
| --- | ---: | ---: |
| 高分候选 F1 | 52.1% | 58.5% |
| 已观测轨迹 F1 | 56.1% | 68.8% |
| 已观测轨迹 IDF1 | 48.5% | 68.0% |
| 已观测虚警 / 漏检 | 58 / 47 | 50 / 28 |
| 换号 / 断续 | 11 / 23 | 2 / 6 |
| 公开身份数 | 15 | 10 |
| 含纯预测的公开轨迹 IDF1 | 42.0% | 51.0% |

门限敏感性：1 m 的已观测 IDF1 为 34.3% → 54.4%；
4 m 为 61.9% → 75.2%。结论不只来自放宽评估距离。
包含纯预测时精度仍低，不能据本轮改善宣布问题全部解决。

## 扩展复验与退步

固定上述实验档后复验，没有按每个序列另外选参数。

| 数据 / 雷达 / 帧数 | 已观测 F1 原 → 新 | IDF1 原 → 新 | 换号 原 → 新 | 虚警 原 → 新 |
| --- | ---: | ---: | ---: | ---: |
| sequence_137 / 1 / 全 1836 | 55.3% → 70.2% | 46.8% → 66.9% | 43 → 4 | 232 → 184 |
| sequence_1 / 1 / 全 1162 | 48.6% → 57.7% | 41.8% → 54.4% | 33 → 11 | 186 → 140 |
| sequence_107 / 1 / 前 500 | 43.6% → 47.1% | 29.9% → 36.6% | 42 → 31 | 561 → 565 |
| sequence_111 / 1 / 全 423 | 0% → 0% | 0% → 0% | 0 → 0 | 672 → 871 |
| sequence_122 / 1 / 全 226 | 50.0% → 45.2% | 50.0% → 45.2% | 0 → 0 | 0 → 0 |
| sequence_137 / 3 / 前 500 | 48.5% → 55.6% | 42.0% → 51.2% | 46 → 19 | 503 → 375 |

sequence_107、111、122 在官方划分中属于 validation；这里只进行了有限的复验，
不是完整验证集验收。sequence_111 只有 1 次真值对象观测，现有无标签聚类仍会把
大量杂波确认成轨迹；弱候选延续还加剧了虚警。sequence_122 的确认等待导致召回下降。
因此 **stable 保持可选，默认不替换 baseline**。

后续优先处理杂波/多径候选的运动一致性或接入可训练检测器，再评估近邻目标合并风险；
盲目降低速度阈值或让单点直接创建 ID 已在本轮试验中增加虚警。

## 在本机复现

生成旧版、迭代版和同步对照页，保留之前的输出目录：

```powershell
conda run --no-capture-output -n radar-id-tracker python examples/compare_radarscenes.py --scenes "C:\Projects\hx\Datasets\RadarScenes\data\sequence_137\scenes.json" --sensor-id 1 --output-dir outputs/sequence_137_sensor1_iteration --limit 200 --visualize --with-camera
```

打开 `outputs/sequence_137_sensor1_iteration/comparison.html`。
两边共享同一时间轴、滑条和播放按钮，黑框为离线真值中位中心；可开关真值。
相机为最近时间图，只作参考。原始回放页和逐帧评分分别保存在 `baseline/`、`stable/`；
`comparison.json` 还包含 1 m、4 m 门限敏感性。复制整个目录即可离线审查。

只生成迭代版本：

```powershell
conda run --no-capture-output -n radar-id-tracker python -m radar_id_tracker.radarscenes_adapter --scenes "C:\Projects\hx\Datasets\RadarScenes\data\sequence_137\scenes.json" --sensor-id 1 --output-dir outputs/sequence_137_sensor1_stable --profile stable --limit 200 --visualize --with-camera
```

完整/多序列复验可对上表序列分别运行 `compare_radarscenes.py`，完整序列省略
`--limit`，另存目录；无需生成图像时省略 `--visualize --with-camera`。
本轮扩展复验原始报告在本机 `outputs/iteration_validation/`。
输出被 Git 忽略，代码和复现方法保存在仓库中。

本机自动测试包含弱点延续、不创建新 ID、标签变更不影响结果和评分分项检查。
