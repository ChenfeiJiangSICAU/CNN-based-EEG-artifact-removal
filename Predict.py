import torch
import numpy as np
from scipy.signal import istft
from U_net_model import UNet
import mne
from Dataset import EEGDenoiseDataset
from Train import device


# 加载模型
model = UNet(in_channels=60, out_channels=60).to(device)
model.load_state_dict(torch.load("best_unet.pth"))
model.eval()

# 待去噪 EEG 原始文件
raw = mne.io.read_raw_edf("noisy_eeg.edf", preload=True)
raw.filter(0.5,45, verbose=False)
raw.notch_filter(50, picks='all', verbose=False)
epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
data = epochs.get_data()  # (n_epochs, C, T)

denoised_epochs = []
for epoch in data:
    spec = EEGDenoiseDataset._stft(None, epoch)  # (C, F, T)
    spec_tensor = torch.from_numpy(spec).unsqueeze(0).to(device)  # batch=1
    with torch.no_grad():
        pred_spec = model(spec_tensor).cpu().numpy()[0]  # 网络输出 (C, F, T)
    # 如果模型输出噪声残差，则应 pred_clean = spec - pred_spec
    # 这里假设输出即为去噪后谱
    denoised_spec = pred_spec
    # ISTFT 反变换成时域信号
    clean_epoch = []
    for ch in range(denoised_spec.shape[0]):
        _, x_rec = istft(denoised_spec[ch], fs=1.0, nperseg=256, noverlap=128)
        clean_epoch.append(x_rec[:epoch.shape[1]])
    clean_epoch = np.stack(clean_epoch, axis=0)
    denoised_epochs.append(clean_epoch)

denoised_data = np.stack(denoised_epochs, axis=0)  # (n_epochs, C, T)
# 保存为 npz 或合并回 Raw
np.savez("denoised_eeg.npz", data=denoised_data, sfreq=int(raw.info['sfreq']))
