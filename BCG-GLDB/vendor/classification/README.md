# 分类模型代码

本目录整理了下游三分类实验实际使用的两个模型。原始代码仍保留在
`classier/medicalnet` 和 `classier/train`，因此不会影响正在运行或已有实验。

## MedicalNet50

- `medicalnet50/resnet.py`：3D ResNet-50 backbone。
- `medicalnet50/classifier.py`：MedicalNet 分类器及分类 head。
- `medicalnet50/train_medicalnet_dx.py`：MRI、PET、MRI+PET 训练入口。

原始训练命令示例：

```bash
cd ./data/classier
python train/train_medicalnet_dx.py --config configs/<medicalnet配置>.yaml
```

## TGLM50

- `tglm50/train_exp08_table_global_local.py`：模型定义、数据集与训练入口。
- 主模型类：`Exp08`。
- 组成：MedicalNet-50 backbone、ClinicalTableEncoder、级联双向 BiCA、GlobalViT3D、融合分类 head。
- BiCA3 接收初始表格状态 `T0`，同时更新 `f3` 和表格状态得到 `T1`；BiCA4 接收 `T1`，同时更新 `f4` 和表格状态得到 `T2`，最终分类使用更新后的临床表示。
- 该结构变化会改变模型参数命名和计算路径，修改前保存的 TGLM50 checkpoint 与指标不能直接复用；需要按新结构重新训练并重新进行 final_val 五折评估。

原始训练命令示例：

```bash
cd ./data/classier
python train/train_exp08_table_global_local.py --config configs/<tglm配置>.json
```

## 公共文件

- `paired_t1_dx_dataset.py`：MRI 数据集。
- `paired_pet_dx_dataset.py`：PET 数据集。
- `paired_mri_pet_dx_dataset.py`：MRI+PET 数据集。
- `evaluate_finalval_5fold.py`：固定最佳模型的最终验证集五折子集评估。

配置仍统一保存在 `./data/classier/configs`，实验结果保存在
`./data/classier/experiments`。
