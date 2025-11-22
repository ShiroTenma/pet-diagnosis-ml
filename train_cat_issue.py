import json
from pathlib import Path
from time import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ----- CONFIG -----
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR_ISSUE = BASE_DIR / "data_cat_issue"
TRAIN_DIR = DATA_DIR_ISSUE / "train"
VAL_DIR = DATA_DIR_ISSUE / "val"

IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 30          # boleh lebih panjang karena ada early stopping
LEARNING_RATE = 5e-5     # sedikit lebih kecil
NUM_WORKERS = 2
EARLY_STOP_PATIENCE = 6

WEIGHT_DECAY = 1e-4      # regulasi tambahan

MODEL_PATH = BASE_DIR / "cat_issue_resnet18.pth"
LABELS_PATH = BASE_DIR / "labels_cat_issue.json"
# ------------------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {name}")
        return torch.device("cuda")
    else:
        print("⚠️  GPU tidak terdeteksi, pakai CPU (lebih lambat).")
        return torch.device("cpu")


def create_dataloaders(device: torch.device):
    # augmentasi train (sedikit lebih agresif tapi masih aman)
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # val tanpa augment random
    val_tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tfms)

    # cek konsistensi nama kelas
    print("train class_to_idx:", train_ds.class_to_idx)
    print("val   class_to_idx:", val_ds.class_to_idx)

    # mapping index -> nama kelas
    idx_to_class = {idx: cls for cls, idx in train_ds.class_to_idx.items()}
    with LABELS_PATH.open("w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)
    print("Cat issue label mapping (saved):", idx_to_class)

    # class weights (di sini hampir seimbang, tapi tetap dihitung)
    targets = torch.tensor(train_ds.targets)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    print("Cat issue class counts :", class_counts.tolist())
    print("Cat issue class weights:", class_weights.tolist())

    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    return train_loader, val_loader, len(train_ds.classes), class_weights


def create_model(num_classes: int, device: torch.device) -> nn.Module:
    # ResNet18 pretrained ImageNet
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    # freeze semua layer dulu
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze hanya layer4 (block terakhir) supaya bisa adapt ke domain cat-skin
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    # tambahkan dropout sebelum fully connected
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )

    model = model.to(device)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def main():
    device = get_device()

    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        print("❌ Folder train/ atau val/ untuk cat issue tidak ditemukan.")
        print("   Pastikan struktur: data_cat_issue/train/<kelas_penyakit>")
        print("                        dan data_cat_issue/val/<kelas_penyakit>")
        return

    train_loader, val_loader, num_classes, class_weights = create_dataloaders(device)
    print(f"Jumlah kelas cat issue: {num_classes}")

    model = create_model(num_classes, device)

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # optimizer melatih layer4 + fc
    params_to_update = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = optim.Adam(
        params_to_update,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # scheduler: turunkan LR tiap 8 epoch
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=8, gamma=0.5
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        dur = time() - start

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f} "
            f"({dur:.1f}s)"
        )

        # step scheduler SESUDAH evaluation
        scheduler.step()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve_epochs = 0

            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ New best cat-issue model saved to {MODEL_PATH} "
                  f"(val_acc={val_acc:.3f})")
        else:
            no_improve_epochs += 1
            print(f"  ⚠️  Val loss tidak membaik ({no_improve_epochs}/{EARLY_STOP_PATIENCE})")

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print("⏹  Early stopping, tidak ada perbaikan val loss.")
            break

    print("Training cat issue selesai.")
    print(f"Best val accuracy (cat issue): {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
