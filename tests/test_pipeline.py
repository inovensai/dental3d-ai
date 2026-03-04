"""Pipeline test modulu.

Gercek OBJ mesh + JSON label verileri ile tum modulleri test eder.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import Teeth3DSDataset, DentalMesh
from src.change_analysis import (
    compute_point_to_point_distances,
    compute_hausdorff_distance,
    analyze_changes,
    classify_change_regions,
    generate_risk_scores,
    compute_regional_statistics,
)
from src.registration import landmark_based_registration
from config import PROJECT_ROOT, FDI_TO_INDEX, NUM_CLASSES


def test_data_loading():
    """Veri seti yukleme testi (data_part_1..7)."""
    print("=" * 50)
    print("TEST: Veri Seti Yukleme (OBJ + JSON)")
    print("=" * 50)

    dataset = Teeth3DSDataset(PROJECT_ROOT)
    stats = dataset.get_statistics()

    print(f"Toplam tarama: {stats['total_scans']}")
    print(f"Etiketli: {stats['labeled_scans']}")
    print(f"Etiketsiz: {stats['unlabeled_scans']}")
    print(f"Hasta sayisi: {stats['unique_patients']}")

    assert stats["total_scans"] > 0, "Veri seti bos!"
    assert stats["labeled_scans"] > 0, "Etiketli veri yok!"
    print("\n[BASARILI] Veri seti yukleme testi gecti.\n")
    return dataset


def test_mesh_loading(dataset: Teeth3DSDataset):
    """Gercek OBJ mesh yukleme testi."""
    print("=" * 50)
    print("TEST: OBJ Mesh Yukleme")
    print("=" * 50)

    scan = dataset.get_labeled_scans()[0]
    print(f"Hasta: {scan.patient_id}, Cene: {scan.jaw_type}")
    print(f"OBJ: {scan.obj_path.name}")

    # Mesh yukle
    scan.load_mesh()
    print(f"Vertex sayisi: {scan.num_vertices:,}")
    print(f"Face sayisi: {scan.num_faces:,}")

    assert scan.num_vertices > 0, "Vertex yok!"
    assert scan.num_faces > 0, "Face yok!"

    # Label yukle
    scan.load_labels()
    print(f"Label sayisi: {len(scan.labels):,}")
    print(f"Dis sayisi: {scan.num_teeth}")
    print(f"FDI numaralari: {scan.unique_teeth}")

    assert len(scan.labels) == scan.num_vertices, "Label ve vertex sayisi esit degil!"
    assert scan.num_teeth > 0, "Hic dis bulunamadi!"

    # Tek dis vertex'leri
    first_tooth = scan.unique_teeth[0]
    tooth_verts = scan.get_tooth_vertices(first_tooth)
    print(f"FDI {first_tooth} vertex sayisi: {len(tooth_verts)}")
    assert len(tooth_verts) > 0

    scan.unload()
    print("\n[BASARILI] OBJ mesh yukleme testi gecti.\n")


def test_point_sampling(dataset: Teeth3DSDataset):
    """Mesh yuzeyinden nokta ornekleme testi."""
    print("=" * 50)
    print("TEST: Nokta Ornekleme")
    print("=" * 50)

    scan = dataset.get_labeled_scans()[0]
    sampled = scan.sample_points(num_points=10000)

    print(f"Noktalar: {sampled['points'].shape}")
    print(f"Normaller: {sampled['normals'].shape}")
    print(f"Etiketler: {sampled['labels'].shape}")

    assert sampled["points"].shape == (10000, 3)
    assert sampled["normals"].shape == (10000, 3)
    assert sampled["labels"].shape == (10000,)
    assert len(set(sampled["labels"])) > 1, "Tek sinif var, yanlis!"

    scan.unload()
    print("\n[BASARILI] Nokta ornekleme testi gecti.\n")


def test_change_analysis():
    """Degisim analizi testi."""
    print("=" * 50)
    print("TEST: Degisim Analizi")
    print("=" * 50)

    np.random.seed(42)
    n = 1000
    source = np.random.randn(n, 3) * 10
    target = source + np.random.normal(0, 0.3, source.shape)
    target[:50] += np.random.normal(0, 2.0, (50, 3))

    distances = compute_point_to_point_distances(source, target)
    hausdorff = compute_hausdorff_distance(source, target)
    result = analyze_changes(source, target, threshold_mm=0.5)

    print(f"Ort. mesafe: {result.mean_distance:.4f}")
    print(f"Hausdorff: {result.hausdorff_distance:.4f}")
    print(f"Anlamli degisim: {result.significant_change_ratio:.1%}")

    assert result.mean_distance > 0
    assert result.hausdorff_distance >= result.max_distance

    # Risk skorlari
    labels = np.random.randint(0, 8, n)
    signed = np.random.randn(n) * 0.5
    regions = classify_change_regions(signed)
    regional = compute_regional_statistics(source, signed, labels, 8)
    risks = generate_risk_scores(regional)
    assert len(risks) > 0

    print("\n[BASARILI] Degisim analizi testi gecti.\n")


def test_landmark_registration():
    """Landmark registration testi."""
    print("=" * 50)
    print("TEST: Landmark Registration")
    print("=" * 50)

    source = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0.5,0.5,1]], dtype=float)
    angle = np.radians(30)
    R = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    t = np.array([5.0, 3.0, 1.0])
    target = (R @ source.T).T + t

    result = landmark_based_registration(source, target)
    print(f"RMSE: {result.inlier_rmse:.8f} mm")
    assert result.inlier_rmse < 1e-6

    print("\n[BASARILI] Registration testi gecti.\n")


def test_pointnet_model():
    """PointNet model testi."""
    print("=" * 50)
    print("TEST: PointNet Model")
    print("=" * 50)

    try:
        import torch
        from src.segmentation import PointNetSegmentation

        model = PointNetSegmentation(num_classes=NUM_CLASSES)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parametreler: {total_params:,}")
        print(f"Sinif sayisi: {NUM_CLASSES}")

        dummy = torch.randn(2, 3, 5000)
        with torch.no_grad():
            output, _, _ = model(dummy)
        print(f"Giris: {dummy.shape} -> Cikis: {output.shape}")
        assert output.shape == (2, NUM_CLASSES, 5000)

        print("\n[BASARILI] PointNet testi gecti.\n")
    except ImportError:
        print("[ATLANDI] PyTorch yuklu degil.\n")


def test_torch_dataset(dataset: Teeth3DSDataset):
    """PyTorch Dataset testi."""
    print("=" * 50)
    print("TEST: PyTorch Dataset")
    print("=" * 50)

    try:
        from src.segmentation import Teeth3DSTorchDataset

        scans = dataset.get_labeled_scans()[:3]
        torch_ds = Teeth3DSTorchDataset(scans, num_points=5000, fdi_to_index=FDI_TO_INDEX)

        points, labels = torch_ds[0]
        print(f"Points: {points.shape}, Labels: {labels.shape}")
        print(f"Label range: [{labels.min()}, {labels.max()}]")
        print(f"Sinif sayisi: {torch_ds.num_classes}")

        assert points.shape == (5000, 3)
        assert labels.shape == (5000,)
        assert labels.max() < NUM_CLASSES

        print("\n[BASARILI] PyTorch Dataset testi gecti.\n")
    except ImportError:
        print("[ATLANDI] PyTorch yuklu degil.\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DentalAI Pipeline Testleri (Gercek Veri)")
    print("=" * 60 + "\n")

    dataset = test_data_loading()
    test_mesh_loading(dataset)
    test_point_sampling(dataset)
    test_change_analysis()
    test_landmark_registration()
    test_pointnet_model()
    test_torch_dataset(dataset)

    print("\n" + "=" * 60)
    print("  TUM TESTLER BASARIYLA TAMAMLANDI!")
    print("=" * 60 + "\n")
