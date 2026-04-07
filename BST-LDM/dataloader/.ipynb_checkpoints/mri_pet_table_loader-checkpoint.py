import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

import os
import re
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data.dataloader import default_collate  # 新增关键导入
from monai.transforms import Compose, LoadImaged, ToTensord, EnsureChannelFirstd, CropForegroundd, Resized, EnsureTyped, NormalizeIntensityd
from table.deal_table import prepare_table
from MRI2PET_old.utils.common import date_difference
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MRI2PET_dataset(Dataset):
    def __init__(self, data_path, table_path='', desired_shape=(160, 160, 96)):
        super(MRI2PET_dataset, self).__init__()
        self.data_path = Path(data_path)  # 转换为Path对象
        self.desired_shape = desired_shape

        # 加载并处理表格数据
        self.import_table = bool(table_path)
        if self.import_table:
            raw_df = pd.read_csv(table_path)
            self.table_dict = prepare_table(raw_df)

        # 获取有效样本列表
        self.valid_samples = self._filter_samples()

        # 数据预处理流程
        self.transform = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureTyped(keys=["image", "label"], dtype=np.float32),
            EnsureChannelFirstd(keys=['image', 'label']),
            CropForegroundd(keys=['label'], source_key='label'),
            Resized(keys=['image', 'label'], spatial_size=desired_shape),
            NormalizeIntensityd(keys=['image'], nonzero=False, channel_wise=False),
            NormalizeIntensityd(keys=['label'], nonzero=False, channel_wise=False),
            ToTensord(keys=['image', 'label'])
        ])

    def _filter_samples(self):
        valid = []
        for folder in os.listdir(self.data_path):
            folder_path = self.data_path / folder

            # 关键修改1：严格检查目录和文件存在性
            if not folder_path.is_dir():
                continue

            # 必须同时存在两个文件
            mri_file = folder_path / "MRI.nii.gz"
            pet_file = folder_path / "PET.nii.gz"
            if not (mri_file.exists() and pet_file.exists()):
                #print(f"跳过缺失文件的样本: {folder}")
                continue

            # 关键修改2：增强正则表达式匹配
            match = re.match(r"(\d+_S_\d+)-([a-zA-Z0-9]+)-(\d{4}[_\-]\d{2}[_\-]\d{2})", folder)
            if not match:
                print(f"跳过格式不匹配的文件夹: {folder}")
                continue

            ptid, _, raw_date = match.groups()
            exam_date = raw_date.replace('_', '-')  # 统一日期格式

            # 表格数据匹配
            table_index = self._find_table_match(ptid, exam_date) if self.import_table else None
            if self.import_table and table_index is None:
                #print(f"跳过无表格匹配的样本: {folder}")
                continue

            valid.append({
                'folder': str(folder_path),  # 保持字符串类型兼容性
                'mri_path': str(mri_file),
                'pet_path': str(pet_file),
                'table_index': table_index
            })
        print(f"有效样本数量: {len(valid)}")
        return valid

    def _find_table_match(self, ptid, exam_date):
        """匹配表格中时间和ID最接近的记录"""
        records = self.table_dict['info'][self.table_dict['info']['PTID'] == ptid]
        if records.empty:
            return None

        min_diff = float('inf')
        best_index = None
        for idx, row in records.iterrows():
            try:
                date_diff = date_difference(exam_date, row['EXAMDATE'])
                if 0 <= date_diff <= 30 and date_diff < min_diff:
                    min_diff = date_diff
                    best_index = idx
            except Exception as e:
                print(f"日期解析错误: {row['EXAMDATE']}, 错误: {str(e)}")
                continue
        return best_index

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, index):
        try:
            sample = self.valid_samples[index]

            # 关键修改3：二次验证文件存在性
            if not (Path(sample['mri_path']).exists() and Path(sample['pet_path']).exists()):
                print(f"文件已移除: {sample['folder']}")
                return None

            batch = self.transform({
                'image': sample['mri_path'],
                'label': sample['pet_path']
            })

            # 添加表格数据
            if self.import_table:
                table_idx = sample['table_index']
                batch['cate_x'] = torch.tensor(
                    self.table_dict['cate_x'].iloc[table_idx].values,
                    dtype=torch.float32
                )
                batch['conti_x'] = torch.tensor(
                    self.table_dict['conti_x'].iloc[table_idx].values,
                    dtype=torch.float32
                )

            # 添加元数据
            batch['folder'] = sample['folder']
            return batch
        except Exception as e:
            print(f"加载样本失败: {sample.get('folder', '未知')}, 错误: {str(e)}")
            return None


def safe_collate(batch):
    """关键修改4：过滤无效样本"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def mri2pet_dataloader(data_path, table_path, desired_shape, batch_size, shuffle=True):
    dataset = MRI2PET_dataset(data_path, table_path, desired_shape)
    print(f"总有效样本数: {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=safe_collate,
        num_workers=4, pin_memory=True
    )


if __name__ == "__main__":
    train_dataloader,test_dataloader = mri2pet_dataloader(
        data_path='/zhangyongquan/chengyh/DATASET/MRI_PET',
        table_path='/zhangyongquan/chengyh/GFE-Mamba-y/GEF-Mamba_ADNI_Dataset/train_data/ct_2&5_3year.csv',
        desired_shape=(160, 160, 96), batch_size=4, shuffle=True
    )


    # 验证第一个batch
    batch = next(iter(train_dataloader))
    if batch is not None:
        print(f"Batch keys: {batch.keys()}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Categorical features: {batch['cate_x'][0].shape}")
        print(f"Continuous features: {batch['conti_x'][0].shape}")
    else:
        print("没有有效数据可加载！")