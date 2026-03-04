"""Demo script - Prototip pipeline'i gercek OBJ mesh verileri ile calistirir.

Kullanim: python run_demo.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import Teeth3DSDataset
from src.change_analysis import analyze_changes, generate_risk_scores, compute_regional_statistics
from src.registration import landmark_based_registration
from config import PROJECT_ROOT, FDI_TOOTH_NAMES, FDI_TO_INDEX, NUM_CLASSES


def main():
    print("\n" + "=" * 60)
    print("  DentalAI - 3B Intraoral Tarama Analiz Prototipi")
    print("  Gercek Mesh Verisi ile Demo")
    print("=" * 60)

    # ─── 1. Veri Yukleme ───
    print("\n[1/6] Veri seti yukleniyor (data_part_1..7)...")
    dataset = Teeth3DSDataset(PROJECT_ROOT)
    stats = dataset.get_statistics()
    print(f"  Toplam tarama: {stats['total_scans']}")
    print(f"  Etiketli: {stats['labeled_scans']}, Etiketsiz: {stats['unlabeled_scans']}")
    print(f"  Ust cene: {stats['upper_scans']}, Alt cene: {stats['lower_scans']}")
    print(f"  Benzersiz hasta: {stats['unique_patients']}")

    # ─── 2. Gercek Mesh Analizi ───
    print("\n[2/6] Gercek mesh yukleme ve analiz...")
    scan = dataset.get_labeled_scans()[0]
    scan.load_mesh()
    scan.load_labels()
    print(f"  Hasta: {scan.patient_id}, Cene: {scan.jaw_type}")
    print(f"  Vertex sayisi: {scan.num_vertices:,}")
    print(f"  Face sayisi: {scan.num_faces:,}")
    print(f"  Dis sayisi: {scan.num_teeth}")
    print(f"  FDI numaralari: {scan.unique_teeth}")

    bbox = scan.get_bounding_box()
    size = bbox[1] - bbox[0]
    print(f"  Boyut: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm")

    # Dis istatistikleri
    tooth_stats = scan.get_tooth_stats()
    print(f"\n  Dis bazinda vertex dagilimi:")
    for fdi in sorted(tooth_stats.keys()):
        name = FDI_TOOTH_NAMES.get(fdi, f"FDI {fdi}")
        count = tooth_stats[fdi]["vertex_count"]
        bar = "#" * max(1, count // 1000)
        print(f"    {name:30s}: {count:6d} vertex {bar}")

    # ─── 3. Nokta Ornekleme ───
    print("\n[3/6] Mesh yuzeyinden nokta ornekleme...")
    sampled = scan.sample_points(num_points=24000)
    print(f"  Orneklenen noktalar: {sampled['points'].shape}")
    print(f"  Normaller: {sampled['normals'].shape}")
    print(f"  Etiketler: {sampled['labels'].shape}")
    print(f"  Benzersiz etiketler: {sorted(set(sampled['labels']))}")

    # ─── 4. Degisim Analizi (Gercek mesh uzerinde) ───
    print("\n[4/6] Degisim analizi simulasyonu (gercek mesh uzerinde)...")
    points = sampled["points"]
    labels = sampled["labels"]
    n = len(points)

    np.random.seed(42)
    simulated = points.copy()
    simulated += np.random.normal(0, 0.12, points.shape).astype(np.float32)
    n_patho = max(1, int(n * 0.10))
    patho_idx = np.random.choice(n, n_patho, replace=False)
    simulated[patho_idx] += np.random.normal(0, 1.2, (n_patho, 3)).astype(np.float32)

    result = analyze_changes(points, simulated, threshold_mm=0.5)
    print(f"  Ortalama mesafe: {result.mean_distance:.4f} mm")
    print(f"  Maksimum mesafe: {result.max_distance:.4f} mm")
    print(f"  Hausdorff mesafesi: {result.hausdorff_distance:.4f} mm")
    print(f"  Anlamli degisim orani: {result.significant_change_ratio:.1%}")

    # Risk skorlari (gercek dis etiketleri ile)
    unique_fdi = sorted(set(labels))
    max_label = max(unique_fdi) + 1
    regional = compute_regional_statistics(points, result.distances, labels, num_classes=max_label)
    risks = generate_risk_scores(regional)
    print(f"\n  Dis bazinda risk skorlari:")
    for fdi in sorted(risks.keys()):
        risk = risks[fdi]
        name = FDI_TOOTH_NAMES.get(int(fdi), f"FDI {fdi}")
        bar = "#" * int(risk["score"] / 5)
        print(f"    {name:30s}: {risk['score']:5.1f}/100 [{risk['level']:>6s}] {bar}")

    scan.unload()

    # ─── 5. Landmark Registration ───
    print("\n[5/6] Landmark tabanli registration testi...")
    np.random.seed(42)
    source = np.random.randn(20, 3).astype(np.float64) * 10
    angle = np.radians(25)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    t = np.array([5.0, 3.0, 1.0])
    target = (R @ source.T).T + t

    reg_result = landmark_based_registration(source, target)
    print(f"  Registration RMSE: {reg_result.inlier_rmse:.8f} mm")
    print(f"  Sonuc: {'BASARILI' if reg_result.inlier_rmse < 0.001 else 'DIKKAT'}")

    # ─── 6. PointNet Model ───
    print("\n[6/6] PointNet model kontrolu...")
    try:
        import torch
        from src.segmentation import PointNetSegmentation, Teeth3DSTorchDataset

        model = PointNetSegmentation(num_classes=NUM_CLASSES)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parametreleri: {total_params:,}")
        print(f"  Sinif sayisi: {NUM_CLASSES}")

        dummy = torch.randn(1, 3, 24000)
        with torch.no_grad():
            output, _, _ = model(dummy)
        print(f"  Forward pass: {dummy.shape} -> {output.shape}")

        device = "MPS" if torch.backends.mps.is_available() else "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"  Cihaz: {device}")

        # PyTorch Dataset testi
        test_scans = dataset.get_labeled_scans()[:2]
        torch_ds = Teeth3DSTorchDataset(test_scans, num_points=5000, fdi_to_index=FDI_TO_INDEX)
        points_t, labels_t = torch_ds[0]
        print(f"\n  PyTorch Dataset testi:")
        print(f"    Giris: {points_t.shape}, Etiket: {labels_t.shape}")
        print(f"    Etiket aralik: [{labels_t.min()}, {labels_t.max()}]")
        print(f"  Egitim icin hazir! Calistirin: python train.py")

    except ImportError:
        print("  [!] PyTorch yuklu degil.")

    # ─── Ozet ───
    print("\n" + "=" * 60)
    print("  DEMO TAMAMLANDI")
    print("=" * 60)
    print(f"""
  Prototip Durumu:
    [x] Veri yukleme: {stats['total_scans']} OBJ mesh + {stats['labeled_scans']} JSON label
    [x] Gercek mesh analizi: vertex/face/bbox/dis istatistikleri
    [x] Nokta ornekleme: mesh yuzeyinden + label transferi
    [x] Degisim analizi: Hausdorff, risk skorlama (gercek dis etiketleri ile)
    [x] Registration: SVD tabanli landmark registration
    [x] PointNet modeli: {NUM_CLASSES} sinif segmentasyon
    [x] PyTorch Dataset: Egitim icin hazir
    [x] Streamlit dashboard: Gercek 3B mesh gorsellestirme
    [x] Egitim scripti: python train.py

  Komutlar:
    streamlit run app.py        -> Web dashboard
    python train.py             -> Model egitimi
    python tests/test_pipeline.py -> Testler
""")


if __name__ == "__main__":
    main()
