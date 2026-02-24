import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm
import random
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

# =====================================================
# REPRODUCIBILITY
# =====================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =====================================================
# LOGGER
# =====================================================
def setup_logger():
    os.makedirs("training_logs", exist_ok=True)
    log_file = f"training_logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("MBLNet")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =====================================================
# CHANNEL ATTENTION (ECA STYLE)
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1d = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg = self.avg_pool(x).view(b, 1, c)
        mx  = self.max_pool(x).view(b, 1, c)

        att = self.conv1d(avg) + self.conv1d(mx)
        att = self.sigmoid(att).view(b, c, 1, 1)

        return x * att


# =====================================================
# MULTI-SCALE BRANCH
# =====================================================
class MultiScaleBranch(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        padding = kernel_size // 2

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size,
                          padding=padding, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.layer1 = block(1, 16)
        self.layer2 = block(16, 24)
        self.layer3 = block(24, 32)
        self.layer4 = block(32, 48)
        self.layer5 = block(48, 64)

        self.att = ChannelAttention(64)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.att(x)
        return x


# =====================================================
# MAIN MODEL
# =====================================================
class MBLNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.local_branch  = MultiScaleBranch(3)
        self.mid_branch    = MultiScaleBranch(5)
        self.global_branch = MultiScaleBranch(7)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        f1 = self.local_branch(x)
        f2 = self.mid_branch(x)
        f3 = self.global_branch(x)

        f = torch.cat([f1, f2, f3], dim=1)  # 64×3 = 192
        f = self.pool(f).flatten(1)
        f = self.projection(f)

        return F.normalize(f, dim=1)


# =====================================================
# DATASET
# =====================================================
class IrisPairDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        img1 = self.transform(Image.open(r.iloc[0]).convert("L"))
        img2 = self.transform(Image.open(r.iloc[2]).convert("L"))

        label = torch.tensor(r.iloc[4], dtype=torch.float32)
        return img1, img2, label


# =====================================================
# ANGULAR MARGIN LOSS
# =====================================================
class AngularMarginLoss(nn.Module):
    def __init__(self, margin=0.4):
        super().__init__()
        self.margin = margin

    def forward(self, f1, f2, y):
        cos = F.cosine_similarity(f1, f2)
        theta = torch.acos(torch.clamp(cos, -0.9999, 0.9999))

        pos = (1 - torch.cos(theta + self.margin)) * y
        neg = F.relu(cos - 0.3) * (1 - y)

        return (pos + neg).mean()


# =====================================================
# EER COMPUTATION
# =====================================================
@torch.no_grad()
def compute_eer(model, csv_file, transform, device):
    model.eval()
    df = pd.read_csv(csv_file)

    scores, labels = [], []

    for _, r in df.iterrows():
        img1 = transform(Image.open(r.iloc[0]).convert("L")).unsqueeze(0).to(device)
        img2 = transform(Image.open(r.iloc[2]).convert("L")).unsqueeze(0).to(device)

        f1 = model(img1)
        f2 = model(img2)

        scores.append(F.cosine_similarity(f1, f2).item())
        labels.append(int(r.iloc[4]))

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.argmin(np.abs(fpr - fnr))]
    return eer


# =====================================================
# TRAINING
# =====================================================
def train():
    logger = setup_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    train_tf = transforms.Compose([
        transforms.Resize((64,256)),
        transforms.RandomAffine(5, translate=(0.05,0.05), scale=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    test_tf = transforms.Compose([
        transforms.Resize((64,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    train_ds = IrisPairDataset("/home/nishkal/alam/mtp_code/iitd_abletion/splits_iitd/train_pairs.csv", train_tf)
    train_loader = DataLoader(train_ds, batch_size=64,
                              shuffle=True, num_workers=0)

    model = MBLNet().to(device)
    criterion = AngularMarginLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4, weight_decay=5e-4)

    best_eer = 1.0
    train_losses = []
    val_eers = []

    for epoch in range(1, 81):
        model.train()
        total_loss = 0

        for x1, x2, y in tqdm(train_loader,
                              desc=f"Epoch {epoch}"):

            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            f1 = model(x1)
            f2 = model(x2)
            loss = criterion(f1, f2, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        eer = compute_eer(model, "/home/nishkal/alam/mtp_code/iitd_abletion/splits_iitd/val_pairs.csv",
                          test_tf, device)

        val_eers.append(eer)

        logger.info(
            f"Epoch {epoch} | Loss {avg_loss:.4f} | "
            f"Val EER {eer*100:.2f}%"
        )

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(),
                       "best_model.pth")
            logger.info("Best model saved")

    # Plotting
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.savefig("plots/train_loss.png")
    plt.close()

    plt.figure()
    plt.plot([e*100 for e in val_eers])
    plt.title("Validation EER (%)")
    plt.savefig("plots/val_eer.png")
    plt.close()

    logger.info("Training complete.")


if __name__ == "__main__":
    train()