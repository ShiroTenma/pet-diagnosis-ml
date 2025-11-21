from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SPLITS = ["train", "val"]
CLASSES = ["cat_healthy", "cat_skin_issue", "dog_healthy", "dog_skin_issue"]


def count_images(path: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in exts)


def main():
    print(f"Base dir : {BASE_DIR}")
    print(f"Data dir : {DATA_DIR}")
    print()

    if not DATA_DIR.exists():
        print("❌ data/ folder tidak ditemukan.")
        return

    for split in SPLITS:
        split_dir = DATA_DIR / split
        print(f"=== Split: {split} ===")
        if not split_dir.exists():
            print(f"  ❌ {split_dir} tidak ada")
            continue

        total_split = 0
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"  ⚠️  Kelas '{cls}' TIDAK ditemukan di {split_dir}")
                continue
            n = count_images(cls_dir)
            total_split += n
            print(f"  - {cls:15s}: {n} images")

        print(f"  >> Total in {split}: {total_split} images\n")

    print("✅ Selesai cek dataset.")
    print("Pastikan setiap kelas punya cukup gambar di train & val.")


if __name__ == "__main__":
    main()
