



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image, UnidentifiedImageError
# import pandas as pd
# import numpy as np
# from sklearn.metrics import roc_curve
# from tqdm import tqdm
# import random
# import logging
# import os
# import gc
# from datetime import datetime
# import matplotlib.pyplot as plt


# # =====================================================
# # REPRODUCIBILITY
# # =====================================================
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# # =====================================================
# # LOGGING SETUP
# # =====================================================
# def setup_logger():
#     os.makedirs("training_logs", exist_ok=True)
#     log_file = f"training_logs/train_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

#     logger = logging.getLogger("MBCNet")
#     logger.setLevel(logging.INFO)

#     if logger.hasHandlers():
#         logger.handlers.clear()

#     fh = logging.FileHandler(log_file)
#     ch = logging.StreamHandler()

#     fmt = logging.Formatter("%(asctime)s | %(message)s")
#     fh.setFormatter(fmt)
#     ch.setFormatter(fmt)

#     logger.addHandler(fh)
#     logger.addHandler(ch)

#     return logger, log_file


# # =====================================================
# # CHANNEL ATTENTION
# # =====================================================
# class ChannelAttention(nn.Module):
#     def __init__(self, channels, k_size=3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.conv1d = nn.Conv1d(
#             1, 1,
#             kernel_size=k_size,
#             padding=(k_size - 1) // 2,
#             bias=False
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.shape
#         avg = self.avg_pool(x).view(b, 1, c)
#         mx = self.max_pool(x).view(b, 1, c)
#         att = self.conv1d(avg) + self.conv1d(mx)
#         att = self.sigmoid(att).view(b, c, 1, 1)
#         return x * att


# # =====================================================
# # MULTI-SCALE BRANCH
# # =====================================================
# class MultiScaleBranch(nn.Module):
#     def __init__(self, kernel_size: int):
#         super().__init__()
#         p = kernel_size // 2

#         def conv_block(in_c, out_c, stride=1):
#             return nn.Sequential(
#                 nn.Conv2d(in_c, out_c, kernel_size,
#                           stride=stride, padding=p, bias=False),
#                 nn.BatchNorm2d(out_c),
#                 nn.ReLU(inplace=True)
#             )

#         self.layer1 = conv_block(1, 16, stride=1)
#         self.layer2 = conv_block(16, 24, stride=1)
#         self.layer3 = conv_block(24, 32, stride=2)
#         self.layer4 = conv_block(32, 48, stride=2)
#         self.layer5 = conv_block(48, 64, stride=2)

#         self.att1 = ChannelAttention(16)
#         self.att2 = ChannelAttention(24)
#         self.att3 = ChannelAttention(32)
#         self.att4 = ChannelAttention(48)
#         self.att5 = ChannelAttention(64)

#     def forward(self, x):
#         x = self.att1(self.layer1(x))
#         x = self.att2(self.layer2(x))
#         x = self.att3(self.layer3(x))
#         x = self.att4(self.layer4(x))
#         x = self.att5(self.layer5(x))
#         return x


# # =====================================================
# # MBCNet
# # =====================================================
# class MBCNet(nn.Module):
#     def __init__(self, emb_dim: int = 128):
#         super().__init__()

#         self.local_branch = MultiScaleBranch(kernel_size=3)
#         self.mid_branch = MultiScaleBranch(kernel_size=5)
#         self.global_branch = MultiScaleBranch(kernel_size=7)

#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.projection = nn.Sequential(
#             nn.Linear(192, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, emb_dim)
#         )

#     def forward(self, x):
#         f1 = self.local_branch(x)
#         f2 = self.mid_branch(x)
#         f3 = self.global_branch(x)

#         f = torch.cat([f1, f2, f3], dim=1)
#         f = self.pool(f).flatten(1)

#         return F.normalize(self.projection(f), dim=1)


# # =====================================================
# # DATASET
# # =====================================================
# class IrisPairDataset(Dataset):
#     def __init__(self, csv_file, transform):
#         self.df = pd.read_csv(csv_file).reset_index(drop=True)
#         self.transform = transform
#         self._validate_paths()

#     def _validate_paths(self):
#         missing = []
#         for _, r in self.df.iterrows():
#             for p in [r.iloc[0], r.iloc[2]]:
#                 if not os.path.exists(str(p)):
#                     missing.append(p)
#         if missing:
#             print(f"[WARNING] {len(missing)} missing image path(s) in CSV.")

#     def __len__(self):
#         return len(self.df)

#     def _load_image(self, path):
#         try:
#             return self.transform(Image.open(path).convert("L"))
#         except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
#             print(f"[WARNING] Cannot load '{path}': {e}. Using zero tensor.")
#             return torch.zeros(1, 64, 256)

#     def __getitem__(self, idx):
#         r = self.df.iloc[idx]
#         img1 = self._load_image(r.iloc[0])
#         img2 = self._load_image(r.iloc[2])
#         label = torch.tensor(float(r.iloc[4]), dtype=torch.float32)
#         return img1, img2, label


# # =====================================================
# # LOSS
# # =====================================================
# class CosineContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.4):
#         super().__init__()
#         self.margin = margin

#     def forward(self, f1, f2, y):
#         cos = F.cosine_similarity(f1, f2)
#         cos = torch.clamp(cos, -1 + 1e-7, 1 - 1e-7)

#         pos = (1 - cos) * y
#         neg = F.relu(cos - self.margin) * (1 - y)
#         loss = (pos + neg).mean()

#         if torch.isnan(loss) or torch.isinf(loss):
#             raise RuntimeError(
#                 f"NaN/Inf in loss — cos: min={cos.min():.4f} "
#                 f"max={cos.max():.4f} mean={cos.mean():.4f}"
#             )

#         return loss


# # =====================================================
# # EER
# # =====================================================
# @torch.no_grad()
# def compute_eer(model, csv_file, transform, device, batch_size=64):
#     model.eval()

#     dataset = IrisPairDataset(csv_file, transform)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=False
#     )

#     scores, labels = [], []

#     for x1, x2, y in loader:
#         x1 = x1.to(device, non_blocking=True)
#         x2 = x2.to(device, non_blocking=True)

#         f1 = model(x1)
#         f2 = model(x2)

#         scores.extend(F.cosine_similarity(f1, f2).cpu().tolist())
#         labels.extend(y.tolist())

#         del x1, x2, f1, f2

#     if len(set(labels)) < 2:
#         print("[WARNING] EER skipped — only one class in validation set.")
#         return 1.0

#     fpr, tpr, _ = roc_curve(labels, scores)
#     fnr = 1 - tpr

#     return float(fpr[np.argmin(np.abs(fpr - fnr))])


# # =====================================================
# # SAFE CHECKPOINT
# # =====================================================
# def safe_save(model, path, logger):
#     tmp = path + ".tmp"
#     try:
#         torch.save(model.state_dict(), tmp)
#         os.replace(tmp, path)
#         logger.info(f"✅ Best model saved → {path}")
#     except Exception as e:
#         logger.error(f"Checkpoint save failed: {e}")
#         if os.path.exists(tmp):
#             os.remove(tmp)


# # =====================================================
# # TRAINING
# # =====================================================
# def train():
#     logger, log_file = setup_logger()
#     logger.info("🚀 Training started — MBCNet (Table 1 architecture)")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logger.info(f"Device: {device}")

#     if device == "cuda":
#         logger.info(f"GPU : {torch.cuda.get_device_name(0)}")
#         logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

#     TRAIN_CSV = "/home/nishkal/alam/mtp_code/casia_interval/dataset/train_pairs.csv"
#     VAL_CSV = "/home/nishkal/alam/mtp_code/casia_interval/dataset/train_pairs.csv"
#     CKPT_PATH = "best_model_1.pth"

#     for p in [TRAIN_CSV, VAL_CSV]:
#         if not os.path.exists(p):
#             raise FileNotFoundError(f"CSV not found: {p}")

#     # train_tf = transforms.Compose([
#     #     transforms.Resize((72, 280)),
#     #     transforms.RandomCrop((64, 256)),
#     #     transforms.RandomAffine(5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
#     #     transforms.ColorJitter(brightness=0.2, contrast=0.3),
#     #     transforms.ToTensor(),
#     #     transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
#     #     transforms.Normalize([0.5], [0.5])
#     # ])

#     test_tf = transforms.Compose([
#         transforms.Resize((64, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     train_tf = transforms.Compose([
#         transforms.Resize((72, 280)),
#         transforms.RandomCrop((64, 256)),
#         transforms.RandomAffine(5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
#         transforms.ToTensor(),
#         transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
#         transforms.Normalize([0.5], [0.5])
#     ])

#     train_ds = IrisPairDataset(TRAIN_CSV, train_tf)
#     logger.info(f"Train samples: {len(train_ds)}")

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=32,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=False,
#         drop_last=True
#     )

#     model = MBCNet(emb_dim=128).to(device)
#     criterion = CosineContrastiveLoss(margin=0.4)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

#     best_eer = 1.0
#     train_losses = []
#     val_eers = []

#     for epoch in range(1, 70):
#         model.train()
#         total_loss = 0.0
#         batch_count = 0

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", ncols=120)

#         for x1, x2, y in pbar:
#             x1 = x1.to(device)
#             x2 = x2.to(device)
#             y = y.to(device)

#             optimizer.zero_grad()
#             f1 = model(x1)
#             f2 = model(x2)
#             loss = criterion(f1, f2, y)

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             optimizer.step()

#             total_loss += loss.item()
#             batch_count += 1

#             pbar.set_postfix(loss=f"{loss.item():.4f}")

#         avg_loss = total_loss / batch_count
#         train_losses.append(avg_loss)

#         eer = compute_eer(model, VAL_CSV, test_tf, device)

#         # --------------------------change 
#         val_acc=1-eer
#         val_eers.append(val_acc)
#         # val_eers.append(eer)

#         #  change made --------------------------
#         logger.info(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val EER: {eer*100:.2f}%")

#         if eer < best_eer:
#             best_eer = eer
#             safe_save(model, CKPT_PATH, logger)

#         torch.cuda.empty_cache()
#         gc.collect()

#     logger.info(f"🏆 Training complete. Best Val EER: {best_eer*100:.2f}%")
#     logger.info(f"📄 Log file: {log_file}")


#     os.makedirs("plots", exist_ok=True)

# epochs = range(1, len(train_losses) + 1)
# os.makedirs("plots", exist_ok=True)

# epochs = range(1, len(train_losses) + 1)

# # -------- Accuracy Plot --------
# plt.figure(figsize=(8,5))
# plt.plot(epochs, train_accuracies, marker='o', label="Train Accuracy")
# plt.plot(epochs, val_eers, marker='o', label="Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("IITD — Train and Validation Accuracy")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/iitd_accuracy.png", dpi=150)
# plt.close()

# # -------- Loss Plot --------
# plt.figure(figsize=(8,5))
# plt.plot(epochs, train_losses, marker='o', label="Train Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("IITD — Training Loss")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/iitd_loss.png", dpi=150)
# plt.close()

# print("✅ IITD plots saved.")






# # =====================================================
# if __name__ == "__main__":
#     try:
#         train()
#     except KeyboardInterrupt:
#         print("\n[INFO] Training interrupted by user.")
#     except Exception as e:
#         logging.getLogger("MBCNet").exception(f"Fatal error: {e}")
#         raise


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm
import random
import logging
import os
import gc
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    log_file = f"training_logs/train_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    logger = logging.getLogger("MBCNet")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_file


# =====================================================
# CHANNEL ATTENTION
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size,
                                padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.avg_pool(x).view(b, 1, c)
        mx = self.max_pool(x).view(b, 1, c)
        att = self.conv1d(avg) + self.conv1d(mx)
        att = self.sigmoid(att).view(b, c, 1, 1)
        return x * att


# =====================================================
# MULTI-SCALE BRANCH
# =====================================================
class MultiScaleBranch(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        p = kernel_size // 2

        def conv_block(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size,
                          stride=stride, padding=p, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.layer1 = conv_block(1, 16)
        self.layer2 = conv_block(16, 24)
        self.layer3 = conv_block(24, 32, stride=2)
        self.layer4 = conv_block(32, 48, stride=2)
        self.layer5 = conv_block(48, 64, stride=2)

        self.att1 = ChannelAttention(16)
        self.att2 = ChannelAttention(24)
        self.att3 = ChannelAttention(32)
        self.att4 = ChannelAttention(48)
        self.att5 = ChannelAttention(64)

    def forward(self, x):
        x = self.att1(self.layer1(x))
        x = self.att2(self.layer2(x))
        x = self.att3(self.layer3(x))
        x = self.att4(self.layer4(x))
        x = self.att5(self.layer5(x))
        return x


# =====================================================
# MBCNet
# =====================================================
class MBCNet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.local_branch = MultiScaleBranch(3)
        self.mid_branch = MultiScaleBranch(5)
        self.global_branch = MultiScaleBranch(7)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Linear(192, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim)
        )

    def forward(self, x):
        f1 = self.local_branch(x)
        f2 = self.mid_branch(x)
        f3 = self.global_branch(x)

        f = torch.cat([f1, f2, f3], dim=1)
        f = self.pool(f).flatten(1)

        return F.normalize(self.projection(f), dim=1)


# =====================================================
# DATASET
# =====================================================
# class IrisPairDataset(Dataset):
#     def __init__(self, csv_file, transform):
#         self.df = pd.read_csv(csv_file).reset_index(drop=True)
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         r = self.df.iloc[idx]
#         img1 = self.transform(Image.open(r.iloc[0]).convert("L"))
#         img2 = self.transform(Image.open(r.iloc[2]).convert("L"))
#         label = torch.tensor(float(r.iloc[4]), dtype=torch.float32)
#         return img1, img2, label
    
class IrisPairDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        try:
            img = Image.open(str(path)).convert("L")
            return self.transform(img)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            return torch.zeros(1, 64, 256)  # fallback

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        path1 = r.iloc[0]
        path2 = r.iloc[2]

        img1 = self._load_image(path1)
        img2 = self._load_image(path2)

        label = torch.tensor(float(r.iloc[4]), dtype=torch.float32)

        return img1, img2, label


# =====================================================
# LOSS
# =====================================================
class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.4):
        super().__init__()
        self.margin = margin

    def forward(self, f1, f2, y):
        cos = F.cosine_similarity(f1, f2)
        pos = (1 - cos) * y
        neg = F.relu(cos - self.margin) * (1 - y)
        return (pos + neg).mean()


# =====================================================
# EER
# =====================================================
@torch.no_grad()
def compute_eer(model, csv_file, transform, device):
    model.eval()
    dataset = IrisPairDataset(csv_file, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    scores, labels = [], []

    for x1, x2, y in loader:
        x1, x2 = x1.to(device), x2.to(device)
        f1, f2 = model(x1), model(x2)
        scores.extend(F.cosine_similarity(f1, f2).cpu().tolist())
        labels.extend(y.tolist())

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    return float(fpr[np.argmin(np.abs(fpr - fnr))])


# =====================================================
# TRAIN
# =====================================================
def train():

    logger, log_file = setup_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    TRAIN_CSV = "/home/nishkal/alam/mtp_code/iitd_abletion/splits_iitd/train_pairs.csv"
    VAL_CSV   = "/home/nishkal/alam/mtp_code/iitd_abletion/splits_iitd/val_pairs.csv"

    train_tf = transforms.Compose([
        transforms.Resize((72, 280)),
        transforms.RandomCrop((64, 256)),
        transforms.RandomAffine(5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_tf = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_loader = DataLoader(
        IrisPairDataset(TRAIN_CSV, train_tf),
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    model = MBCNet().to(device)
    criterion = CosineContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_losses = []
    train_accs = []
    val_accs = []
    val_eers = []






    @torch.no_grad()
    def plot_roc(model, csv_file, transform, device):
        model.eval()
        dataset = IrisPairDataset(csv_file, transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        scores, labels = [], []

        for x1, x2, y in loader:
            x1, x2 = x1.to(device), x2.to(device)
            f1, f2 = model(x1), model(x2)

            scores.extend(F.cosine_similarity(f1, f2).cpu().numpy())
            labels.extend(y.numpy())

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

        plt.xlabel("False Acceptance Rate (FAR)")
        plt.ylabel("True Acceptance Rate (TAR)")
        plt.title("IITD ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/iitd_roc.png", dpi=300)
        plt.close()
    @torch.no_grad()
    def plot_far_frr(model, csv_file, transform, device):
        model.eval()
        dataset = IrisPairDataset(csv_file, transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        scores, labels = [], []

        for x1, x2, y in loader:
            x1, x2 = x1.to(device), x2.to(device)
            f1, f2 = model(x1), model(x2)

            scores.extend(F.cosine_similarity(f1, f2).cpu().numpy())
            labels.extend(y.numpy())

        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr

        # Find EER index
        eer_index = np.argmin(np.abs(fpr - fnr))
        eer = fpr[eer_index]
        eer_threshold = thresholds[eer_index]

        plt.figure()
        plt.plot(thresholds, fpr, label="FAR", linewidth=2)
        plt.plot(thresholds, fnr, label="FRR", linewidth=2)

        # Mark EER point
        plt.scatter(eer_threshold, eer, s=60, label=f"EER = {eer:.4f}")

        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.title("IITD FAR–FRR Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/iitd_far_frr.png", dpi=300)
        plt.close()

        

    for epoch in range(1, 30):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x1, x2, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            f1, f2 = model(x1), model(x2)
            loss = criterion(f1, f2, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            cos = F.cosine_similarity(f1, f2)
            preds = (cos > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        eer = compute_eer(model, VAL_CSV, test_tf, device)
        val_acc = 1 - eer

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_eers.append(eer)

        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # ===================== PLOTS =====================

    os.makedirs("plots", exist_ok=True)
    epochs = [e  for e in range(1, len(train_losses) + 1)]

# -------- Accuracy --------
    plt.figure()
    plt.plot(epochs, train_accs, marker='o',label="Train Accuracy")
    plt.plot(epochs, val_accs, marker='s',label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("IITD Accuracy")

    plt.xlim(0, 30)
    plt.xticks(range(0, 31, 3))
   

    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.savefig("plots/iitd_accuracy.png", dpi=150)
    plt.close()


    # -------- Loss --------
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, marker='o',label="Train Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("IITD Loss")

    plt.xlim(0, 30)
    plt.xticks(range(0, 31, 3))

    plt.legend()
    plt.grid(True)
   
    plt.savefig("plots/iitd_loss.png")
    plt.close()

    # -------- EER Plot --------
    plt.figure()
    plt.plot(epochs, val_eers, marker='o', label="Validation EER")

    plt.xlabel("Epoch")
    plt.ylabel("EER")
    plt.title("IITD Validation EER")

    plt.xlim(0, 30)
    plt.xticks(range(0, 31, 3))

    plt.legend()
    plt.grid(True)
    plt.savefig("plots/iitd_eer.png", dpi=150)
    plt.close()
    plot_roc(model, VAL_CSV, test_tf, device)
    plot_far_frr(model, VAL_CSV, test_tf, device)



    print("✅ IITD plots saved in 'plots/' folder.")


   

# =====================================================
if __name__ == "__main__":
    train()