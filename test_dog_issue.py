import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# config dasar (samakan dengan train_dog_issue.py)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR_ISSUE = BASE_DIR / "data_dog_issue"
TEST_DIR = DATA_DIR_ISSUE / "test"

IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 2

MODEL_PATH = BASE_DIR / "dog_issue_resnet18.pth"
LABELS_PATH = BASE_DIR / "labels_dog_issue.json"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {name}")
        return torch.device("cuda")
    else:
        print("⚠️  GPU tidak terdeteksi, pakai CPU (lebih lambat).")
        return torch.device("cpu")


def create_test_loader(device: torch.device):
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Folder test untuk dog issue tidak ditemukan: {TEST_DIR}\n"
            "Pastikan struktur: data_dog_issue/test/<kelas_penyakit>"
        )

    test_tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tfms)

    # optional: cek konsistensi dengan labels_dog_issue.json
    if LABELS_PATH.exists():
        with LABELS_PATH.open("r", encoding="utf-8") as f:
            idx_to_class_train = json.load(f)
        idx_to_class_test = {idx: cls for cls, idx in test_ds.class_to_idx.items()}

        if set(idx_to_class_train.values()) != set(idx_to_class_test.values()):
            print("⚠️  Peringatan: kelas di test/ tidak sama dengan labels_dog_issue.json")
            print("    train:", idx_to_class_train)
            print("    test :", idx_to_class_test)

    pin = device.type == "cuda"

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    return test_loader, len(test_ds.classes), test_ds.class_to_idx


@torch.no_grad()
def evaluate_on_test(model, loader, device: torch.device, class_to_idx: dict):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    test_loss = running_loss / total if total > 0 else 0.0
    test_acc = correct / total if total > 0 else 0.0

    print("\n=== Dog Issue Test Result ===")
    print(f"Test loss    : {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.3f} ({correct}/{total})")

    # confusion matrix + classification report (jika scikit-learn terpasang)
    try:
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion matrix (rows=true, cols=pred):")
        print(cm)

        # urutkan nama kelas berdasarkan index
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        print("\nClassification report:")
        print(classification_report(all_labels, all_preds, target_names=target_names))
    except Exception:
        print("\n(skipping confusion matrix & classification report; "
              "install scikit-learn kalau mau pakai)")


def create_model(num_classes: int, device: torch.device) -> nn.Module:
    # harus sama persis dengan arsitektur di train_dog_issue.py
    from torchvision import models

    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model


def main():
    device = get_device()

    test_loader, num_classes, class_to_idx = create_test_loader(device)
    print(f"Jumlah kelas dog issue di test: {num_classes}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model dog issue tidak ditemukan: {MODEL_PATH}\n"
            "Pastikan sudah menjalankan train_dog_issue.py terlebih dahulu."
        )

    model = create_model(num_classes, device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"✅ Dog-issue model weights loaded from {MODEL_PATH}")

    evaluate_on_test(model, test_loader, device, class_to_idx)


if __name__ == "__main__":
    main()
