import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import stft

class EEGDenoiseDataset(Dataset):
    def __init__(self, file_list, method='mask', mask_ratio=0.5):
        """
        file_list: EEG 窗口 npz 文件路径列表
        method: 'mask' or 'subsample' 或 'noise2noise'
        mask_ratio: 遮罩比例（仅在 mask 方法时使用）
        """
        self.files = file_list
        self.method = method
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.files)

    def _stft(self, signal, n_fft=256, hop_length=128):
        """
        计算短时傅里叶变换；返回幅度谱 (channels, freq_bins, time_bins)
        """
        channels, timelen = signal.shape
        spec = []
        for ch in range(channels):
            f, t, Zxx = stft(signal[ch], fs=1.0, nperseg=n_fft, noverlap=n_fft-hop_length)
            spec.append(np.abs(Zxx))
        spec = np.stack(spec, axis=0)
        return spec  # 维度 (C, F, T)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        eeg = arr['data']    # 原始信号 (C, T)
        # 1. 计算 STFT 幅度谱
        spec = self._stft(eeg)      # ndarray (C, F, T)

        # 2. 生成自监督输入输出
        if self.method == 'mask':
            # 随机遮罩部分谱图
            mask = (np.random.rand(*spec.shape) > self.mask_ratio).astype(np.float32)
            input_spec = spec * mask
            target_spec = spec
            loss_mask = 1.0 - mask  # 1 表示被遮罩的位置（参与损失计算）
        elif self.method == 'subsample':
            # 每隔一帧保留一帧：如时间轴上奇偶帧
            C, F, T = spec.shape
            mask = np.zeros((C, F, T), dtype=np.float32)
            mask[..., ::2] = 1.0   # 保留偶数帧（或奇数帧）
            input_spec = spec * mask
            target_spec = spec
            loss_mask = 1.0 - mask
        else:
            # Noise2Noise: 假设数据文件包含 'noisy' 和 'noisy2' 视图
            # 这里简化为同 spec 作为输入和目标
            input_spec = spec
            target_spec = spec
            loss_mask = np.ones_like(spec)

        # 转为 PyTorch 张量
        input_spec = torch.from_numpy(input_spec).float()
        target_spec = torch.from_numpy(target_spec).float()
        loss_mask = torch.from_numpy(loss_mask).float()
        return input_spec, target_spec, loss_mask
