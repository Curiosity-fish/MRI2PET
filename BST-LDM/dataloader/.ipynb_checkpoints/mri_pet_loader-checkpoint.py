import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'

import os
import re
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from monai.transforms import Compose, LoadImaged, ToTensord, EnsureChannelFirstd, CropForegroundd, Resized, ScaleIntensityRanged, EnsureTyped, NormalizeIntensityd, Lambdad, ScaleIntensityRangePercentilesd
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib  # 新增导入，用于检查图像通道数


class MRI2PET_dataset(Dataset):
    def __init__(self, data_path, desired_shape=(160, 160, 96)):
        super(MRI2PET_dataset, self).__init__()
        self.data_path = Path(data_path)  # 转换为Path对象
        self.desired_shape = desired_shape

        # 获取有效样本列表
        self.valid_samples = self._filter_samples()

        # 数据预处理流程
        self.transform = Compose([
            LoadImaged(keys=['image', 'label']),
            EnsureTyped(keys=["image", "label"], dtype=np.float32),
            EnsureChannelFirstd(keys=['image', 'label']),
            CropForegroundd(keys=['label'], source_key='label'),
            Resized(keys=['image', 'label'], spatial_size=desired_shape),
            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=False),
            #ScaleIntensityRanged(keys=['image'], a_min=0, a_max=7000, b_min=-1, b_max=1, clip=True),
            #ScaleIntensityRanged(keys=['label'], a_min=-0.5, a_max=3, b_min=-1, b_max=1, clip=True),
            ToTensord(keys=['image', 'label'])
        ])

    def _check_channel_count(self, file_path):
        """检查图像通道数是否为1"""
        try:
            img = nib.load(file_path)
            # 获取图像的维度
            dim = len(img.shape)
            
            # 如果维度为4，则最后一个维度可能是通道数
            if dim == 4:
                channels = img.shape[-1]
                return channels == 1
            # 如果维度为3，则是单通道图像
            elif dim == 3:
                return True
            else:
                return False
        except Exception as e:
            print(f"检查通道数失败: {file_path}, 错误: {str(e)}")
            return False

    def _filter_samples(self):
        valid = []
        for folder in os.listdir(self.data_path):
            folder_path = self.data_path / folder

            # 检查目录存在性
            if not folder_path.is_dir():
                continue

            # 必须同时存在两个文件
            mri_file = folder_path / "MRI.nii.gz"
            pet_file = folder_path / "PET.nii.gz"
            if not (mri_file.exists() and pet_file.exists()):
                #print(f"跳过缺失文件的样本: {folder}")
                continue

            # 检查图像通道数是否为1
            if not (self._check_channel_count(mri_file) and self._check_channel_count(pet_file)):
                print(f"跳过通道数不为1的样本: {folder}")
                continue

            # 检查文件夹格式
            match = re.match(r"(\d+_S_\d+)-([a-zA-Z0-9]+)-(\d{4}[_\-]\d{2}[_\-]\d{2})", folder)
            if not match:
                print(f"跳过格式不匹配的文件夹: {folder}")
                continue

            valid.append({
                'folder': str(folder_path),  # 保持字符串类型兼容性
                'mri_path': str(mri_file),
                'pet_path': str(pet_file),
            })
        print(f"有效样本数量: {len(valid)}")
        return valid

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, index):
        try:
            sample = self.valid_samples[index]

            # 二次验证文件存在性
            if not (Path(sample['mri_path']).exists() and Path(sample['pet_path']).exists()):
                print(f"文件已移除: {sample['folder']}")
                return None

            batch = self.transform({
                'image': sample['mri_path'],
                'label': sample['pet_path']
            })
            
            # 再次验证通道数（确保转换后也是单通道）
            if batch['image'].shape[0] != 1 or batch['label'].shape[0] != 1:
                print(f"转换后通道数不为1的样本: {sample['folder']}")
                return None

            # 添加元数据
            batch['folder'] = sample['folder']
            return batch
        except Exception as e:
            print(f"加载样本失败: {sample.get('folder', '未知')}, 错误: {str(e)}")
            return None


def safe_collate(batch):
    """过滤无效样本"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def mri2pet_dataloader(data_path, desired_shape, batch_size, shuffle=True):
    dataset = MRI2PET_dataset(data_path, desired_shape)
    print(f"总有效样本数: {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=safe_collate,
        num_workers=4, pin_memory=True
    )