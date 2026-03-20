import os
import mne
import numpy as np

# 参数配置
RAW_FILE = "Raw_data/sub-CTR001_ses-V1_task-OE_eeg.vhdr"  # 原始 EEG 文件
OUT_DIR = "val_data"  # 导出目录
WINDOW_SEC = 2.0           # 窗口长度（秒）

# 读取原始 EEG
raw = mne.io.read_raw_brainvision(RAW_FILE, preload=True)  # 载入 EDF 【7†L729-L737】
raw.load_data()
# 预处理：带通滤波 0.5–45 Hz，去除直流和超高频噪声
raw.filter(0.5, 45.0, verbose=False)               # 带通滤波
# 去除电源工频噪声（50 Hz）
raw.notch_filter(freqs=50.0, picks='all', verbose=False)  # 陷波【12†L730-L739】【12†L789-L792】

# 分窗保存
os.makedirs(OUT_DIR, exist_ok=True)
sfreq = int(raw.info['sfreq'])
# 生成固定长度的 epochs（不可叠加窗）
epochs = mne.make_fixed_length_epochs(raw, duration=WINDOW_SEC, preload=True)
data = epochs.get_data()  # 形状 (n_epochs, n_channels, n_times)
for i, epoch in enumerate(data):
    # 保存 npz: 包含 EEG 数据和采样率
    fn = os.path.join(OUT_DIR, f"win_{i:05d}.npz")
    np.savez(fn, data=epoch, sfreq=sfreq)
