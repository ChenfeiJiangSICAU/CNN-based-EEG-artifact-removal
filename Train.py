import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from skimage.metrics import structural_similarity as ssim
import glob
from Dataset import EEGDenoiseDataset
from U_net_model import UNet
import torch.nn as nn


# 参数示例
train_files = sorted(glob.glob("./train_data/*.npz"))
val_files = sorted(glob.glob("./val_data/*.npz"))
batch_size = 16; lr = 1e-3; epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集和加载器
train_ds = EEGDenoiseDataset(train_files, method='mask', mask_ratio=0.3)
val_ds = EEGDenoiseDataset(val_files, method='mask', mask_ratio=0.3)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# 模型、优化器
model = UNet(in_channels=60, out_channels=60).to(device)  # 假设3通道谱
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()

best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets, loss_mask in train_loader:
        # 转为 GPU 张量
        inputs = inputs.to(device); targets = targets.to(device); loss_mask = loss_mask.to(device)
        # 前向
        outputs = model(inputs)
        # 计算损失（只在遮罩位置计算）
        # 对齐 outputs 和 targets 尺寸
        if outputs.shape != targets.shape:
            H = min(outputs.shape[2], targets.shape[2])
            W = min(outputs.shape[3], targets.shape[3])

            outputs = outputs[:, :, :H, :W]
            targets = targets[:, :, :H, :W]
            loss_mask = loss_mask[:, :, :H, :W]
        loss_mse = mse_loss(outputs * loss_mask, targets * loss_mask)
        loss_l1  = mae_loss(outputs * loss_mask, targets * loss_mask)
        loss = loss_mse + 0.1 * loss_l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets, loss_mask in val_loader:
            inputs = inputs.to(device); targets = targets.to(device); loss_mask = loss_mask.to(device)
            outputs = model(inputs)
            #对齐（验证阶段也做）
            if outputs.shape != targets.shape:
                H = min(outputs.shape[2], targets.shape[2])
                W = min(outputs.shape[3], targets.shape[3])

                outputs = outputs[:, :, :H, :W]
                targets = targets[:, :, :H, :W]
                loss_mask = loss_mask[:, :, :H, :W]
            loss_val = mse_loss(outputs * loss_mask, targets * loss_mask)
            val_loss += loss_val.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_unet.pth")
