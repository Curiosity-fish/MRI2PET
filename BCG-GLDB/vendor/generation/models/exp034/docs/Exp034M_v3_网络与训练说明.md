# Exp034M_v3 网络与训练说明

## 1. 实验定位

Exp034M_v3 是 MRI 到 FDG-PET 的 latent bridge。流程为：

```text
MRI -> MRI patch AE encoder -> MRI latent x1 -> U-Net bridge
    -> PET latent z -> PET patch AE decoder -> PET
```

正式配置：

```text
configs/t1mri_fdgpet_exp034m_v3_joint_full.yaml
experiments/Exp034M_v3_Joint_Table_BSFE_Mamba_Decoder_ROI_full/
```

数据使用 ADNI exp020 全量配对数据和 seed42 划分；表格来自
`datasets/ADNI_NACC_Table.csv)，MRI/表格访视日期窗口为 90 天。

## 2. MRI/PET Patch AE

### 2.1 作用和形状

AE 将 1 通道 MRI 或 PET 压缩为 4 通道 latent。原图大小为
`160 x 192 x 160)，patch 大小为 `40 x 48 x 40`。每个 patch
经过两次下采样，latent patch 大小为 `10 x 12 x 10`。

因此：

```text
latent patch: [4, 10, 12, 10]
full latent:  [4, 40, 48, 40]
```

### 2.2 AE 架构

当前 AE 参数：

```text
输入通道: 1
latent 通道: 4
base channels: 16
GroupNorm groups: 8
dropout: 0
```

编码器：

```text
1 -> 16 -> 32 -> 64 -> 4 latent
```

解码器：

```text
4 latent -> 64 -> 32 -> 16 -> 1
```

每个分辨率包含两个残差卷积块；编码器使用两次 3D 下采样，
解码器使用两次 3D 上采样。

AE checkpoint 来源：

```text
PET AE : experiments/Exp033J_AE_patch_pet_full/checkpoints/best.pt
MRI AE : experiments/Exp034B_AE_patch_mri_full/checkpoints/best.pt
```

AE 训练目标是 patch reconstruction 和 full reconstruction。对应
VAE 配置中的 KL 最大权重为 0.001，并在 20 个 epoch 内 warm-up。
M_v3 不重新训练 MRI AE encoder，而是直接加载已有权重。

M_v3 中：

```text
MRI AE encoder: 冻结
PET AE encoder: 不用于 bridge 输入
PET AE decoder: 加载已有权重，并从第 1 轮参与联合训练
```

## 3. Bridge U-Net 主干

### 3.1 输入和时间信息

输入是 MRI AE latent：

```text
x1: [B, 4, 40, 48, 40]
```

时间步先做正弦时间编码，再由 MLP 映射为 4 通道并直接加到
`x1)。当前单步 bridge 中训练代码固定使用 `t=1)。

### 3.2 通道和分辨率

M_v3 当前配置为：

```text
64 -> 128 -> 256 -> 512 -> 512 -> 256 -> 128 -> 64 -> 4
```

三次下采样后的分辨率为：

```text
40 x 48 x 40, 64 channels
20 x 24 x 20, 128 channels
10 x 12 x 10, 256 channels
5  x 6  x 5,  512 channels
```

Stem 是一个 `4 -> 64` 的 ResBlock3D。每个 encoder 下采样层为：

```text
ResBlock3D -> ResBlock3D -> GlobalSelfAttention3D -> Conv3D(stride=2)
```

中间层为两个 `512 -> 512` 的 ResBlock3D。每个 decoder 上采样层为：

```text
ConvTranspose3D -> skip concat -> ResBlock3D -> ResBlock3D
-> GlobalSelfAttention3D
```

输出头：

```text
GroupNorm -> SiLU -> Conv3D(64 -> 4)
```

**M_v3 的最深通道是 512，不是 1024。** 1024 是后续
Exp034N01 独立分支加入的瓶颈配置，不属于 M_v3。

### 3.3 全局 3D 注意力

每个 encoder 和 decoder stage 都在第二个残差块后使用
`GlobalSelfAttention3D)。实现为：

```text
GroupNorm -> 1x1 Conv 得到 Q/K/V
-> memory-efficient multi-head attention -> 1x1 Conv
```

注意力头数为 4，使用 xFormers memory-efficient attention，避免显式
构造完整的 voxel-by-voxel 注意力矩阵。

## 4. 表格信息模块

表格经过 missing-aware tokenizer 转为 128 维 token。每个特征保留
数值信息和缺失信息。

核心模块 `BidirectionalTableStage` 做两个方向的交互：

```text
image tokens <- table tokens
table tokens <- updated image tokens
```

处理过程：

1. 当前 image feature 自适应池化到固定网格并转成 image tokens。
2. image query 读取 table keys/values。
3. 得到 image delta，并更新 image tokens。
4. table query 再读取更新后的 image tokens。
5. image delta 上采样回当前分辨率。
6. 通过可学习 gate 做残差注入。

残差形式为：

```text
image_out = image + lambda_scale
            * sigmoid(spatial_gate + time_gate) * image_delta
```

M_v3 从第 1 轮启用表格模块，主要放在前两个 encoder 高分辨率和前两个
decoder 高分辨率阶段。投影和 gate 使用零初始化，使训练初期接近原始 U-Net。

## 5. BSFE 模块

### 5.1 权重来源

BSFE 和 PET head 加载：

```text
experiments/Exp011b_bsfe_t1mri_fdgpet_fold0/checkpoints/fold0_best.pt
```

checkpoint 包含 `bsfe` 和 `head)。M_v3 从第 1 轮启用 BSFE 路径，
并允许 BSFE/head 一起更新。

### 5.2 结构

```text
base channels: 24
channel multipliers: [2, 4, 8, 16]
pyramid channels: [48, 96, 192, 384]
resblocks per stage: 3
attention stage: deepest stage 3
attention heads: 4
attention head dim: 32
tokens: 32
token dim: 128
global dim: 256
dropout: 0.1
```

BSFE 从原始 MRI 提取多尺度 pyramid、global feature 和 token feature。
pyramid 通过 1x1 zero-conv 投影到 U-Net 对应通道，注入 encoder、
bottleneck 和 decoder。

BSFE head 根据 pyramid/global/token 生成 PET structural prediction，
用于 BSFE 引导损失。

## 6. Mamba 全局/局部融合

Mamba 位于 U-Net bottleneck：

```text
512 feature -> 1x1 Conv -> 128-dim tokens
-> AxialBiMambaBlock -> 1x1 Conv -> 512 context
```

参数：

```text
d_state: 8
d_conv: 3
expand: 1
bidirectional: true
active axes: x, y, z
line chunk: 256
gradient checkpointing: true
```

Mamba context 在 bottleneck 处与 image feature gated fusion，并投影到
两个高分辨率 decoder stage 再次 gated fusion。相关投影使用零初始化。

## 7. PET decoder

U-Net 输出 PET latent：

```text
z: [B, 4, 40, 48, 40]
```

全图 latent 按 patch grid 切分，经 PET patch AE decoder 解码后再 stitch：

```text
PET latent -> latent patches -> patch decoder -> PET patches -> stitch
```

PET decoder 从 Exp033J checkpoint 初始化，并从第 1 轮加入优化器。
MRI AE encoder 始终冻结。

## 8. 损失函数

M_v3 从第 1 轮同时启用所有损失。

### 8.1 图像 L1

```text
L_image = L1(PET_recon, PET_target)
```

### 8.2 表格辅助损失

latent 经过 TablePredictor 预测表格特征：

```text
L_table = masked SmoothL1(table_pred, table_target)
```

缺失特征由 table mask 排除，总权重为 0.05。

### 8.3 BSFE 引导损失

```text
L_bsfe = L1(PET_recon, BSFE_PET_prediction)
```

权重为 0.02。

### 8.4 全局和局部结构损失

全局项比较平均池化后的 PET：

```text
L_global = L1(avg_pool(PET_recon), avg_pool(PET_target))
```

局部项比较三个空间方向的一阶差分：

```text
L_local = 1/3 * sum_d L1(diff_d(PET_recon), diff_d(PET_target))
```

权重分别为 0.10 和 0.05。

### 8.5 脑区损失

使用 parc98 label map 和 brain mask。对每个有效脑区计算 PET 区域均值，
并按脑区体素数加权。

```text
L_roi_profile  = SmoothL1(区域均值预测, 区域均值目标)
L_roi_contrast = SmoothL1(中心化区域均值预测, 中心化区域均值目标)
```

权重分别为 0.03 和 0.02。

### 8.6 SSIM 损失

```text
L_ssim = 1 - SSIM(PET_recon, PET_target)
```

权重为 0.20。

### 8.7 总损失

```text
L_total = L_image
        + 0.05 * L_table
        + 0.02 * L_bsfe
        + 0.10 * L_global
        + 0.05 * L_local
        + 0.03 * L_roi_profile
        + 0.02 * L_roi_contrast
        + 0.20 * L_ssim
```

## 9. 训练设置

```text
epochs: 300
batch size: 1
learning rate: 5e-5
weight decay: 1e-4
num workers: 4
AMP: bfloat16
seed: 42
```

M_v3 不使用 M_v2 的六阶段 curriculum，而是在 epoch 1 同时启用表格、
BSFE、Mamba、PET decoder、ROI 和 SSIM 路径。

## 10. 当前训练和资源

输出目录：

```text
experiments/Exp034M_v3_Joint_Table_BSFE_Mamba_Decoder_ROI_full/
```

日志：

```text
logs/train_gpu0_20260821_0439.log
```

完整 stage-6 smoke 观测峰值显存约 23.5GB。GPU0 同时存在约 4.9GB
其他占用，因此实际总占用接近 30GB，batch size 保持为 1。

## 11. 与 Exp034N01 的区别

Exp034N01 是独立调优分支，只训练 M_v2 的第一阶段。N01 后来将瓶颈
扩展为：

```text
64 -> 128 -> 256 -> 512 -> 1024 -> 512 -> 256 -> 128 -> 64
```

1024 通道改动只属于 N01，不属于当前 M_v3。
