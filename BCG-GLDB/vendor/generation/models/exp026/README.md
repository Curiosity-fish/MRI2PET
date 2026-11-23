# Exp026: Patch-DVAE + MRI-ControlNet + Table + Global Bi-Mamba

该目录是一个独立的七阶段 MRI→PET 实验，不依赖 `exp025` 中缺失的 `bridge/engine/losses` 公共模块。整体采用“局部 patch 生成主导、全局 Mamba 信息辅助”的路线。

## 数据流

1. PET 被切成 `4×4×4=64` 个不重叠 patch；默认单块 `40×48×40`，每次仅处理 8 块。
2. Patch-DVAE 将每块 PET 压缩为 `4×10×12×10` latent。
3. MRI 先按完整体积编码为高/中/低三尺度结构特征，再裁成与 64 个 PET patch 对齐的特征；MRI prior head 先学习三尺度 PET patch-latent 先验。
4. MRI 结构特征通过零初始化的 `1×1×1` 卷积注入局部 U-Net。缺失感知表格 tokenizer 与 64 个 MRI patch token 双向交互，产生 subject token、可靠度和 64 个 PET patch-latent prior。
5. Bi-Mamba 只处理 64 个 patch 摘要 token，执行三种空间扫描顺序及正反向建模；表格 token 与 patch token 在 Mamba 前后双向交互。
6. 局部 U-Net 每次最多处理 8 个 `[4,10,12,10]` latent patch，内部仅两次下采样：`10×12×10 → 5×6×5 → 3×3×3`；训练时每个 chunk 立即反向，不再保留全部 64 个局部网络计算图。
7. 所有 `q_sample`、噪声预测和 `predict_clean` 都在 `[B,64,4,10,12,10]` patch-latent 上完成；只在图像重建监督时解码 patch 并拼接为 full PET。默认开启 activation checkpointing，以反向重计算降低 3D 中间激活占用。

## 七阶段

| 阶段 | 训练模块 | 主要监督 |
|---|---|---|
| 1 `patch_vae` | DVAE 编码器+解码器 | patch PET 重建、图像梯度、KL |
| 2 `mri_patch_prior` | MRI 多尺度编码器、MRI prior heads | 高/中/低三尺度 PET patch-latent、梯度 |
| 3 `local_control` | MRI 编码器、MRI prior heads、局部 U-Net、MRI 零卷积 | patch 噪声、clean patch-latent、MRI prior 保持损失 |
| 4 `table_mri_prior` | 表格 tokenizer、Table-MRI 双向交互、patch residual head | 64 个 PET patch-latent prior 与梯度 |
| 5 `global_mamba` | Bi-Mamba、全局-局部融合门 | patch 噪声和 clean patch-latent |
| 6 `joint_finetune` | MRI、MRI prior、表格、Mamba、局部 U-Net | patch 噪声、latent、prior、full PET、seam |
| 7 `decoder_calibration` | 仅 DVAE decoder | 真实 latent 为主、预测 latent 为辅 |

阶段6冻结 DVAE 编码器和解码器参数，但 decoder 前向不使用 `torch.no_grad()`，因此 full-PET loss 会穿过冻结 decoder 更新去噪网络。阶段7冻结所有上游模块，只校准 decoder。

阶段1不再同时计算数学上等价的 patch L1 与 full-volume L1。阶段6的 full-PET L1 利用非重叠等尺寸 patch 的等价性逐 chunk 解码计算；seam loss 先在无模型计算图的完整 patch-latent 预测上求边界梯度，再通过第二遍 chunk 重算把该梯度传回网络，因此无需同时保留 64 个局部 U-Net 图。

## 配置与运行

七个阶段的 epoch 均在 `configs/t1mri_fdgpet_exp026.yaml` 的 `stages` 下独立设置；每个阶段还可以单独设置 `enabled`、`learning_rate`、`resume` 和 `loss` 权重。

```powershell
python -m exp026.train_exp026 --config exp026/configs/t1mri_fdgpet_exp026.yaml
```

只运行某个阶段时：

```powershell
python -m exp026.train_exp026 --config exp026/configs/t1mri_fdgpet_exp026.yaml --stage global_mamba
```

单独启动后期阶段时，应通过该阶段的 `resume` 或 `experiment.initial_checkpoint` 加载前序权重。每个阶段保存：

```text
exp026_outputs/t1mri_fdgpet/<stage>/checkpoints/last.pt
exp026_outputs/t1mri_fdgpet/<stage>/checkpoints/best.pt
exp026_outputs/t1mri_fdgpet/<stage>/best_metric.json
```

连续运行七个阶段时，每个阶段结束后会自动重新加载该阶段的 `best.pt`，再进入下一阶段。因此阶段间始终承接验证最优权重。验证噪声由 `training.validation_seed` 固定；存在验证集时只使用验证损失选择 best，且最后一个 epoch 一定执行验证。`last.pt` 仅用于当前阶段意外中断后的 `resume`；如果希望单独启动某个后续阶段，应将上一阶段的 `best.pt` 配置为 `experiment.initial_checkpoint`，并保持当前阶段的 `resume: null`。

本次七阶段模型新增了 `mri_prior_heads`，并将表格先验从整幅 latent 改为 64 个 patch prior，因此旧六阶段版本保存的 checkpoint 与当前模型结构不兼容。

## 依赖说明

生产配置要求安装 `mamba-ssm`。代码不会在依赖缺失时自动用注意力代替 Mamba。仅做 CPU/形状调试时，可显式设置：

```yaml
model:
  global_mamba:
    backend: debug_gru
    allow_test_backend: true
```

`debug_gru` 不是正式实验后端，不能用于最终对比结果。
