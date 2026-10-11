The inference architecture and decoding in `centerpoint_model.py` and the
associated geometric preprocessing are adapted from OpenMMLab MMDetection3D,
Copyright (c) OpenMMLab. All rights reserved, licensed under Apache License 2.0.
The original license is reproduced in `docs/LICENSE-OpenMMLab.txt`.

Reference commit: `fe25f7a51d36e3702f961e198894580d83c4387b` in
https://github.com/open-mmlab/mmdetection3d . References include
PillarFeatureNet/PFNLayer, SECOND, SECONDFPN, SeparateHead/CenterHead,
CenterPointBBoxCoder and circle_nms. The local port removes framework registries,
training code and compiled operators, fixes one checkpoint architecture, and uses
NumPy hard voxelization. It retains checkpoint parameter names and loads strictly.
The optional front-head adapter and training examples reuse that fixed model's
head architecture. Front-scoped Gaussian heatmap and center/size/rotation targets
follow the CenterPoint representation; the local offline target generation and
frozen-backbone training loop are independent additions, not an upstream training
recipe or benchmark reproduction. No learned model files are shipped in Git.
It is an experimental port, not an official OpenMMLab runtime or verified benchmark
reproduction. Tensor parity checks are documented in `docs/lidar-surround.md`.

Model weights are downloaded separately from OpenMMLab with a fixed SHA256;
they are not included in the repository. Dataset licenses and access conditions
remain those of their respective providers.

The optional `openpcdet_adapter.py` bridge invokes the original OpenPCDet
runtime from https://github.com/open-mmlab/OpenPCDet, pinned to commit
`233f849829b6ac19afb8af8837a0246890908755`. OpenPCDet is licensed under
Apache License 2.0. Its source checkout, original LICENSE and separately
downloaded model-zoo weights are stored under ignored `outputs/` paths, not
distributed as this package's source. No OpenPCDet neural architecture or
compiled operator implementation is copied into the bridge. Dataset and
weight access conditions remain those of their providers. See
`docs/lidar-official-detectors.md` for provenance and the unvalidated runtime.
