"""PointNet Segmentasyon Model Egitim Scripti.

Teeth3DS+ veri seti uzerinde dis segmentasyonu egitimi yapar.

Kullanim:
    python train.py                          # Varsayilan ayarlar
    python train.py --epochs 50 --batch 8    # Ozel ayarlar
    python train.py --parts 1 2              # Sadece belirli part'lar
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from src.data_loader import Teeth3DSDataset
from src.segmentation import (
    PointNetSegmentation,
    Teeth3DSTorchDataset,
    SegmentationTrainer,
)
from config import PROJECT_ROOT, MODELS_DIR, FDI_TO_INDEX, NUM_CLASSES


def main():
    parser = argparse.ArgumentParser(description="PointNet Dis Segmentasyonu Egitimi")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch sayisi")
    parser.add_argument("--batch", type=int, default=4, help="Batch boyutu")
    parser.add_argument("--lr", type=float, default=0.001, help="Ogrenme orani")
    parser.add_argument("--num-points", type=int, default=24000, help="Orneklenen nokta sayisi")
    parser.add_argument("--parts", nargs="+", type=int, default=list(range(1, 7)),
                        help="Kullanilacak data_part numaralari (varsayilan: 1-6)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validasyon orani")
    parser.add_argument("--save-dir", type=str, default=str(MODELS_DIR))
    args = parser.parse_args()

    print("=" * 60)
    print("  PointNet Dis Segmentasyonu Egitimi")
    print("=" * 60)

    # ─── 1. Veri Yukleme ───
    print(f"\n[1/4] Veri seti yukleniyor (parts: {args.parts})...")
    dataset = Teeth3DSDataset(PROJECT_ROOT, parts=args.parts)
    print(f"  Toplam: {len(dataset)} tarama, Etiketli: {len(dataset.get_labeled_scans())}")

    train_scans, val_scans = dataset.split_train_val(val_ratio=args.val_ratio)
    print(f"  Egitim: {len(train_scans)}, Validasyon: {len(val_scans)}")

    # ─── 2. PyTorch Dataset ───
    print(f"\n[2/4] PyTorch Dataset olusturuluyor ({args.num_points} nokta/ornek)...")
    train_dataset = Teeth3DSTorchDataset(
        train_scans, num_points=args.num_points,
        fdi_to_index=FDI_TO_INDEX, augment=True,
    )
    val_dataset = Teeth3DSTorchDataset(
        val_scans, num_points=args.num_points,
        fdi_to_index=FDI_TO_INDEX, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch,
        shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch,
        shuffle=False, num_workers=0, drop_last=False,
    )

    print(f"  Egitim batch sayisi: {len(train_loader)}")
    print(f"  Validasyon batch sayisi: {len(val_loader)}")
    print(f"  Sinif sayisi: {NUM_CLASSES}")

    # ─── 3. Model ───
    print(f"\n[3/4] Model olusturuluyor...")
    model = PointNetSegmentation(num_classes=NUM_CLASSES, feature_transform=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametre sayisi: {total_params:,}")

    trainer = SegmentationTrainer(model, learning_rate=args.lr)
    print(f"  Cihaz: {trainer.device}")

    # ─── 4. Egitim ───
    print(f"\n[4/4] Egitim basliyor ({args.epochs} epoch)...")
    print("-" * 60)

    best_dice = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)

        # Her 5 epoch'ta validasyon
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = trainer.evaluate(val_loader)
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"mDSC: {val_metrics['mean_dice']:.4f}"
            )

            # En iyi modeli kaydet
            if val_metrics["mean_dice"] > best_dice:
                best_dice = val_metrics["mean_dice"]
                model_path = save_dir / "best_model.pth"
                trainer.save_model(str(model_path))
                print(f"  >>> Yeni en iyi model! mDSC: {best_dice:.4f}")
        else:
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
            )

    # Son modeli kaydet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = save_dir / f"model_{timestamp}.pth"
    trainer.save_model(str(final_path))

    print("\n" + "=" * 60)
    print(f"  Egitim tamamlandi!")
    print(f"  En iyi mDSC: {best_dice:.4f}")
    print(f"  Model: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
