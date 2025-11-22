import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# import fungsi & config dari train.py
from train import (
    get_device,
    create_model,
    IMG_SIZE,
    DATA_DIR,
    MODEL_PATH,
    LABELS_PATH,
)


def create_test_loader(device: torch.device):
    TEST_DIR = DATA_DIR / "test"

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Folder test tidak ditemukan: {TEST_DIR}\n"
            "Pastikan struktur: data/test/cat_healthy, cat_skin_issue, "
            "dog_healthy, dog_skin_issue"
        )

    # transform sama seperti val (tanpa augmentasi random)
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

    # optional: cek konsistensi label dengan labels.json
    if LABELS_PATH.exists():
        with LABELS_PATH.open("r", encoding="utf-8") as f:
            idx_to_class_train = json.load(f)
        # kelas dari ImageFolder diurutkan alfabetis
        idx_to_class_test = {idx: cls for cls, idx in test_ds.class_to_idx.items()}

        if set(idx_to_class_train.values()) != set(idx_to_class_test.values()):
            print("⚠️  Peringatan: kelas di test/ tidak sama dengan labels.json")
            print("    train:", idx_to_class_train)
            print("    test :", idx_to_class_test)

    pin = device.type == "cuda"

    test_loader = DataLoader(
        test_ds,
        batch_size=8,         # sama seperti train/val
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
    )

    return test_loader, len(test_ds.classes)


@torch.no_grad()
def evaluate_on_test(model, loader, device: torch.device):
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

    print(f"\n=== Test Result ===")
    print(f"Test loss    : {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.3f} ({correct}/{total})")

    # optional: confusion matrix + classification report jika sklearn terpasang
    try:
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion matrix:")
        print(cm)

        print("\nClassification report:")
        print(classification_report(all_labels, all_preds))
    except Exception:
        print("\n(skipping confusion matrix & classification report; "
              "install scikit-learn kalau mau pakai)")


def main():
    device = get_device()

    test_loader, num_classes = create_test_loader(device)
    print(f"Jumlah kelas di test: {num_classes}")

    # buat model dan load weight terbaik
    model = create_model(num_classes, device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}\n"
            "Pastikan sudah menjalankan train.py dan model terbaik tersimpan."
        )

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    print(f"✅ Model weights loaded from {MODEL_PATH}")

    evaluate_on_test(model, test_loader, device)


if __name__ == "__main__":
    main()
