# 本机开发与 Git 同步

2026-09-30 已核对本机仓库：`C:\Projects\hx\code\radar-id-tracker`，远程为
`https://github.com/hanxu-eng/radar-id-tracker.git`，`main` 跟踪 `origin/main`。
已通过 Git Credential Manager 登录，`git fetch origin` 与
`git push --dry-run origin main` 成功。提交身份在本仓库内按既有提交作者配置；
`pull.ff=only` 防止拉取时自动生成合并提交。未推送本次代码修改。

在 PowerShell 中：

```powershell
cd C:\Projects\hx\code\radar-id-tracker
git status
git pull --ff-only
conda activate radar-id-tracker
python -m unittest discover -s tests -v
```

本机已有 `radar-id-tracker` 环境和可编辑安装，修改 `src/` 后立即生效。
未激活环境的终端可运行：

```powershell
conda run --no-capture-output -n radar-id-tracker python -m unittest discover -s tests -v
```

不要在 `base` 环境直接运行项目，也不要只调用其他环境的 `python.exe`
而跳过激活：本机实验发现后者会让 NumPy 线性求解无法正常结束，
正确激活或 `conda run` 后测试通过。

提交前先查看差异，选择要提交的源码、测试和文档，随后执行：

```powershell
git diff
git diff --check
git add src tests docs examples .gitignore README.md pyproject.toml
git commit -m "描述本次修改"
git push origin main
```

先确认没有其他尚未保存的修改再拉取；快进失败时应查看分支差异。
`outputs/` 和根目录示例 `tracks.jsonl` 已加入忽略列表，保留本机结果且避免将
大量 SVG、相机图和实验输出带入 Git。`datasets/` 同样被忽略；本轮小型 KITTI
测试数据保存在 `datasets/kitti_tracking`，原 RadarScenes 数据位于仓库外。

2026-10-09 的分数策略迭代使用 `codex/lidar-score-confirmation` 功能分支，经 PR 审核再
合并到 `main`。已有 LiDAR 基础和本轮策略分别提交，其他二维 RadarScenes 实验仍在
工作区，不进入本轮发布。最新完整 KITTI 原始目录为 `C:\Projects\hx\Datasets\kitti`，
原始数据只读，生成文件保留在仓库忽略目录中。

效果审查与复现命令见 [本轮迭代报告](iteration-2026-09-30.md)。

2026-10-09 的下一阶段使用 `codex/lidar-surround-detector`，以分数策略功能分支为基线，
通过独立草稿 PR 审核；没有合并到 `main`。推理环境在项目忽略目录
`outputs/envs/lidar-detector`，模型和全部输出也在 `outputs/`。本机 Git Credential Manager
继续用于 fetch/push。该实验验收失败，详见 [全周检测报告](lidar-surround.md)。

后续跨序列输入验证使用 `codex/lidar-input-validation`，基于全周检测功能分支，仍通过
草稿 PR 审核，不合并主分支。实验输出在 `outputs/kitti_surround_generalization`，配置
与结果见 [泛化检查报告](lidar-input-validation.md)。单序列阶段结果不是最终参数依据。

当前迭代使用 `codex/lidar-box-quality`，基于输入验证分支，以独立草稿 PR 发布。
增加当前点云框质量评分、整序列五折训练与三条新序列验证；默认配置不变，实验仍未
通过部署验收。结果与复现命令见 [框质量报告](lidar-box-quality.md)，本机输出在
`outputs/kitti_box_quality`。其他未提交 RadarScenes 改动继续留在工作区。
