# 论文对齐版 MRI→FDG-PET 潜空间桥接方法

本目录是第四章对应的独立代码版本。原始代码来源为 `原项目源码`，原目录保留不变。

## 主入口

- `v9_bridge/train_exp037G_v9_bridge.py`
- `v9_bridge/t1mri_fdgpet_exp037G_v9_bridge.yaml`

## 论文对齐口径

- Patch AE/DVAE 先行预训练；最终桥接训练中冻结 AE。
- BSFE 先行训练；最终桥接训练中冻结 BSFE，仅提供多尺度结构条件。
- 桥接总时间步为 `T=10`。
- 训练阶段随机采样时间位置并构造带噪桥接状态，模型预测 PET latent endpoint。
- 推理阶段从 MRI 条件端点开始执行 10 次确定性时间步更新，再经 PET AE 解码。
- 最终训练目标以代码实际生效的 latent endpoint、图像、Parc98、条件排序、临床语义和全局一致性损失为准。

## 运行前提

需要在包含数据、AE/BSFE checkpoint 及必要依赖的 GPU 环境中运行；本目录不包含大规模数据和权重。
