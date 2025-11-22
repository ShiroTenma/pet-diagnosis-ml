import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


# ===== CONFIG PATH =====
BASE_DIR = Path(__file__).resolve().parent

MAIN_MODEL_PATH = BASE_DIR / "pet_skin_resnet18.pth"
DOG_MODEL_PATH = BASE_DIR / "dog_issue_resnet18.pth"
CAT_MODEL_PATH = BASE_DIR / "cat_issue_resnet18.pth"

MAIN_LABELS_PATH = BASE_DIR / "labels.json"
DOG_LABELS_PATH = BASE_DIR / "labels_dog_issue.json"
CAT_LABELS_PATH = BASE_DIR / "labels_cat_issue.json"

IMG_SIZE = 224
# =======================


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️  GPU tidak terdeteksi, pakai CPU.")
        return torch.device("cpu")


# transform inference (tanpa augmentasi random)
INFER_TFMS = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_labels(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Label file tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # pastikan key int
    return {int(k): v for k, v in data.items()}


def create_main_model(num_classes: int, device: torch.device) -> nn.Module:
    # harus konsisten dengan train.py (ResNet18 + Linear fc)
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    for p in model.parameters():
        p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.to(device)
    model.eval()
    return model


def create_dog_model(num_classes: int, device: torch.device) -> nn.Module:
    # konsisten dengan train_dog_issue.py (ResNet18 + Linear fc)
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    for p in model.parameters():
        p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.to(device)
    model.eval()
    return model


def create_cat_model(num_classes: int, device: torch.device) -> nn.Module:
    # konsisten dengan train_cat_issue.py (unfreeze layer4 + Dropout + Linear)
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )

    model.to(device)
    model.eval()
    return model


class PetSkinPipeline:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()

        # load label mapping
        self.main_labels = load_labels(MAIN_LABELS_PATH)
        self.dog_labels = load_labels(DOG_LABELS_PATH)
        self.cat_labels = load_labels(CAT_LABELS_PATH)

        # buat mapping index -> nama
        self.main_idx_to_name = self.main_labels
        self.dog_idx_to_name = self.dog_labels
        self.cat_idx_to_name = self.cat_labels

        # load model utama
        if not MAIN_MODEL_PATH.exists():
            raise FileNotFoundError(f"Main model tidak ditemukan: {MAIN_MODEL_PATH}")
        self.main_model = create_main_model(len(self.main_idx_to_name), self.device)
        state = torch.load(MAIN_MODEL_PATH, map_location=self.device)
        self.main_model.load_state_dict(state)

        # load model dog issue
        if not DOG_MODEL_PATH.exists():
            raise FileNotFoundError(f"Dog issue model tidak ditemukan: {DOG_MODEL_PATH}")
        self.dog_model = create_dog_model(len(self.dog_idx_to_name), self.device)
        state = torch.load(DOG_MODEL_PATH, map_location=self.device)
        self.dog_model.load_state_dict(state)

        # load model cat issue
        if not CAT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Cat issue model tidak ditemukan: {CAT_MODEL_PATH}")
        self.cat_model = create_cat_model(len(self.cat_idx_to_name), self.device)
        state = torch.load(CAT_MODEL_PATH, map_location=self.device)
        self.cat_model.load_state_dict(state)

        print("✅ Semua model & label berhasil dimuat.")

    @torch.no_grad()
    def _predict_single(
        self,
        model: nn.Module,
        tensor: torch.Tensor,
        idx_to_name: Dict[int, str],
    ):
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        label = idx_to_name[int(idx)]
        return label, float(conf), probs.cpu().tolist()

    @torch.no_grad()
    def predict_image(
        self,
        image_path: str,
        issue_threshold: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline penuh di satu gambar.
        issue_threshold: minimal confidence di model utama untuk lanjut ke model issue.
        """
        img = Image.open(image_path).convert("RGB")
        x = INFER_TFMS(img).unsqueeze(0).to(self.device)

        # ---- tahap 1: model utama (4 kelas) ----
        main_label, main_conf, main_probs = self._predict_single(
            self.main_model, x, self.main_idx_to_name
        )

        # parse species & status
        if main_label.startswith("cat_"):
            species = "cat"
        elif main_label.startswith("dog_"):
            species = "dog"
        else:
            species = "unknown"

        is_issue = main_label.endswith("skin_issue")
        detail = None

        # ---- tahap 2: model issue spesifik ----
        if is_issue and main_conf >= issue_threshold:
            if main_label == "dog_skin_issue":
                issue_label, issue_conf, issue_probs = self._predict_single(
                    self.dog_model, x, self.dog_idx_to_name
                )
            elif main_label == "cat_skin_issue":
                issue_label, issue_conf, issue_probs = self._predict_single(
                    self.cat_model, x, self.cat_idx_to_name
                )
            else:
                issue_label, issue_conf, issue_probs = None, None, None

            if issue_label is not None:
                detail = {
                    "issue_label": issue_label,
                    "issue_confidence": issue_conf,
                    "issue_probs": issue_probs,
                }

        result: Dict[str, Any] = {
            "image_path": image_path,
            "species": species,
            "main_label": main_label,
            "main_confidence": main_conf,
            "main_probs": main_probs,
            "is_issue": bool(is_issue),
            "detail": detail,
        }
        return result


def main_cli():
    if len(sys.argv) < 2:
        print("Usage: python predict_pipeline.py <path_to_image> [threshold]")
        sys.exit(1)

    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.6

    pipeline = PetSkinPipeline()
    result = pipeline.predict_image(image_path, issue_threshold=threshold)

    # print hasil sebagai JSON pretty
    print("\n=== Prediction Result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main_cli()
