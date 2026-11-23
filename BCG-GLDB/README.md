# MRI2PET: MRI与临床信息引导的三维FDG-PET合成

本仓库整理硕士论文《基于MRI与临床信息引导的三维FDG-PET合成方法研究》的代码，包括生成方法BCG-GLDB及用于疾病分类验证的TGLM。代码按论文描述实现；与历史实验记录、权重或论文表格数值的一致性仍需逐项复核，不能视作已完成的数值复现。

## 方法与实现

生成方法采用Patch-AE和全脑潜空间重组、MRI到PET的扩散桥、MRI多尺度结构条件、缺失感知临床表格交互、全局特征分支及Parc98区域损失。分类模型TGLM用于评估合成影像在下游疾病分类中的信息可利用性。

| 内容 | 位置 |
| --- | --- |
| 生成网络与Patch-AE | `paper_code/generation_model.py` |
| 扩散桥与生成损失 | `paper_code/generation_objective.py` |
| 生成训练及评估 | `paper_code/train_patch_ae.py`、`train_generation.py`、`evaluate_generation.py` |
| 分类网络、数据及训练 | `paper_code/classification_model.py`、`classification_data.py`、`train_classification.py` |
| 实验配置及测试 | `configs/`、`tests/` |
| 旧项目依赖组件 | `vendor/` |

`vendor/`是论文对齐版依赖的原项目源代码快照，包含一些历史模块，并非每个文件都在论文主实验中使用。

## 环境与数据

使用Python 3.10或3.11和PyTorch；依赖见`requirements.txt`。请根据GPU安装匹配的PyTorch CUDA构建，再安装其余依赖。完整三维训练显存需求取决于硬件和配置。

```bash
python -m pip install -r requirements.txt
```

本仓库不分发ADNI/NACC影像、临床表、受试者清单、Parc98标签图、预训练模型或实验权重。请按数据库要求申请数据，完成论文规定的影像预处理，并在`configs/`中填写路径。生成任务配对CSV至少需要`subject_id,sample_id,t1_path,fdg_pet_path,has_fdg_pet,t1_date`；划分CSV至少需要`subject_id,sample_id,split`。详细要求见[数据与复现边界](docs/reproducibility.md)。

## 训练与评估

在仓库根目录运行；先检查配置中的数据和权重路径。每个阶段需要真实数据，示例配置不能直接复现论文数值。

```bash
python -m paper_code.train_patch_ae --config configs/patch_ae.yaml
python -m paper_code.train_generation --config configs/generation.yaml
python -m paper_code.evaluate_generation --source-config configs/generation.yaml --checkpoint experiments/paper_bcg_gldb_adni/best_generation.pt --target-config configs/generation_nacc.yaml --splits final_val --output experiments/adni_to_nacc
python -m paper_code.make_ablation_configs --base-config configs/generation.yaml --output-dir configs/ablations
python -m paper_code.train_classification --config configs/classification.example.yaml --experiment source_ours_tglm
python -m unittest discover -s tests -v
```

跨数据库反向实验需要以另一数据库为训练源重新训练。分类比较依赖真实PET和合成PET文件；外部基线的训练代码与权重未提供。

## 已知边界

论文对齐代码来自本地整理版；服务器论文入口后来发生过修改，包括桥损失系数和临床编码实现。近期的端点直出及GV28损失实验属于后续试验。它们不能与本仓库的论文对齐入口混为一谈，详情见[版本说明](docs/reproducibility.md)。
