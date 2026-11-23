# MRI2PET论文主方法代码

本项目实现论文《基于MRI与临床信息引导的三维FDG-PET合成方法研究》中的Patch-AE、BCG-GLDB生成网络和TGLM分类网络，以及各自的训练和生成影像评价入口。输入影像需要事先完成论文所述预处理；仓库不包含预处理流程、外部对比模型、对比实验脚本、数据或训练权重。

## 项目结构

- `configs/`：配置示例，使用前必须改成实际路径。
- `mri2pet/models/`：Patch-AE、BCG-GLDB和TGLM。
- `mri2pet/losses/`：桥接、图像与Parc98损失。
- `mri2pet/data/`：已处理影像及临床表格的读取。
- `mri2pet/metrics/`：生成及分类评价指标。
- `mri2pet/cli/`：训练和生成评价入口。
- `tests/`：小尺寸单元测试。

## 环境

建议Python 3.10，按设备安装对应版本的PyTorch，然后安装`requirements.txt`中的其他依赖。运行命令前切换到仓库根目录。实际三维网络的显存需求取决于设备与训练设置。

## 输入数据

准备符合论文预处理定义的MRI、FDG-PET及配对表、受试者级划分表、临床表和Parc98标签图，配置文件中的路径均为示例。`data.root`作为配置内相对路径的基准；CSV中的MRI和PET路径应填写可访问的绝对路径。不要提交受限数据、临床记录或训练检查点。

生成任务的配对表至少包括`subject_id,sample_id,t1_path,fdg_pet_path,has_fdg_pet,t1_date`，划分表包括`subject_id,sample_id,split`，其中划分值为`train`、`val`、`final_val`。同一受试者不得跨集合。临床表需包含`PTID`和`configs/generation.yaml`列出的25个字段；有访视日期时使用`VISDATE`。Parc98标签图需要有1—98全部标签，标签空间必须与已处理影像对应。HAC使用该图谱实际的47对左右同源区域；4个非成对结构不参与HAC。对应标签名称、编号和图谱SHA-256见`configs/parc98_hemisphere_pairs.json`。生成训练会校验图谱文件的SHA-256；使用其他版本的图谱前，须核实标签并更新配对及校验值。

分类任务还需包含诊断`label`的样本表（CN=0、MCI=1、AD=2）、相同受试者级划分信息，以及真实PET或合成PET路径列。参考`configs/classification.example.yaml`。分类入口检查固定测试受试者是否出现在生成器训练或验证集合中；临床填补和标准化统计仅从每折分类训练样本估计。

## 运行

按实际位置修改配置文件。生成阶段必须先训练或提供与当前实现兼容的Patch-AE检查点，并在`configs/generation.yaml`配置对应路径。

```bash
python -m mri2pet.cli.train_patch_ae_metric --config configs/patch_ae.yaml
python -m mri2pet.cli.train_generation --config configs/generation.yaml
python -m mri2pet.cli.evaluate_generation --source-config configs/generation.yaml --checkpoint experiments/paper_bcg_gldb_adni/best_generation.pt --target-config configs/generation.yaml --splits final_val --output experiments/evaluation
python -m mri2pet.cli.train_classification --config configs/classification.example.yaml --experiment source_ours_tglm
python -m unittest discover -s tests -v
```

Patch-AE指标训练入口使用分块梯度累积计算完整影像平均L1，并根据验证集平均SSIM选择模型；`train_autoencoder`入口则按验证L1选择模型。生成训练使用验证集平均SSIM选择模型；分类训练使用折内验证集Macro-F1选择模型。`source_real_tglm`和`source_ours_tglm`分别选择真实或合成PET输入，必须在样本表中提供对应路径。

## 版本说明

本版修正了已确认与论文不一致的损失系数、MRI结构编码起始层、时间条件残差块、临床双向交互和全局分支，并补充分类BiCA输出归一化。网络结构与旧版不同，旧生成器及分类检查点可能无法严格加载；不要把旧权重或旧实验结果直接视作本版结果。Patch-AE结构未改变，但若要求训练过程与本版一致，需以单项L1重新训练。

测试使用人工构造的小尺寸数据，不代表完整尺寸训练、历史实验复现或临床验证。请在实际数据和GPU环境中验证显存、路径及训练流程。

临床记录仅在同一受试者、同一访视的重复行中合并；未观测字段保留缺失标记。修复后的47对HAC与论文中49对连续编号的描述存在差异，旧生成结果不能视为修复后的实验结果。分类配置可选用`data.fold_split_csv`固定五折划分；未提供时由训练入口生成。
