"""Landmark analizi modülü.

Teeth3DS+ landmark verilerini kullanarak dental anatomik
analiz ve karşılaştırma yapar. Mevcut veri seti ile
doğrudan çalışabilir (OBJ mesh dosyaları olmadan).
"""
import json
import numpy as np
from pathlib import Path
from typing import Optional
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import ConvexHull

from .data_loader import DentalScan, Teeth3DSDataset


def compute_arch_dimensions(scan: DentalScan) -> dict:
    """Çene ark boyutlarını landmark noktalarından hesaplar.

    Returns:
        Ark genişliği, derinliği ve çevresi bilgileri
    """
    coords = scan.landmark_coords
    if len(coords) == 0:
        return {}

    bbox_min, bbox_max = scan.get_bounding_box()
    dimensions = bbox_max - bbox_min

    # Konveks gövde çevresi (ark uzunluğu tahmini)
    try:
        hull = ConvexHull(coords[:, :2])  # XY düzleminde
        arch_perimeter = hull.area  # 2D'de area = çevre
    except Exception:
        arch_perimeter = 0.0

    return {
        "width_mm": float(dimensions[0]),       # Mesio-distal genişlik
        "depth_mm": float(dimensions[1]),       # Bukko-lingual derinlik
        "height_mm": float(dimensions[2]),      # Oklüzal yükseklik
        "centroid": scan.get_centroid().tolist(),
        "arch_perimeter_mm": float(arch_perimeter),
        "num_landmarks": scan.num_landmarks,
    }


def compute_inter_landmark_distances(scan: DentalScan) -> np.ndarray:
    """Tüm landmark çiftleri arasındaki mesafeleri hesaplar.

    Returns:
        NxN mesafe matrisi
    """
    coords = scan.landmark_coords
    if len(coords) == 0:
        return np.array([])
    return squareform(pdist(coords))


def compare_landmark_patterns(
    scan_a: DentalScan,
    scan_b: DentalScan,
) -> dict:
    """İki tarama arasındaki landmark pattern farkını analiz eder.

    Aynı hastanın farklı zamanlardaki taramalarını karşılaştırmak
    veya farklı hastaları karşılaştırmak için kullanılır.
    """
    coords_a = scan_a.landmark_coords
    coords_b = scan_b.landmark_coords

    # Merkez nokta farkı
    centroid_a = scan_a.get_centroid()
    centroid_b = scan_b.get_centroid()
    centroid_diff = np.linalg.norm(centroid_a - centroid_b)

    # Boyut farkları
    dim_a = compute_arch_dimensions(scan_a)
    dim_b = compute_arch_dimensions(scan_b)

    # Landmark sayısı farkı
    n_common = min(len(coords_a), len(coords_b))

    # En yakın noktalar arası ortalama mesafe (eşleştirilmemiş karşılaştırma)
    if n_common > 0:
        cross_distances = cdist(coords_a, coords_b)
        min_distances_a = cross_distances.min(axis=1)
        min_distances_b = cross_distances.min(axis=0)
        avg_nearest = float(np.mean(np.concatenate([min_distances_a, min_distances_b])))
    else:
        avg_nearest = float("inf")

    return {
        "centroid_difference_mm": float(centroid_diff),
        "width_diff_mm": dim_a.get("width_mm", 0) - dim_b.get("width_mm", 0),
        "depth_diff_mm": dim_a.get("depth_mm", 0) - dim_b.get("depth_mm", 0),
        "height_diff_mm": dim_a.get("height_mm", 0) - dim_b.get("height_mm", 0),
        "avg_nearest_distance_mm": avg_nearest,
        "landmark_count_a": scan_a.num_landmarks,
        "landmark_count_b": scan_b.num_landmarks,
    }


def analyze_class_distribution(scan: DentalScan) -> dict:
    """Landmark sınıf dağılımını analiz eder."""
    class_counts = {}
    class_centroids = {}

    for cls in set(scan.landmark_classes):
        lms = scan.get_landmarks_by_class(cls)
        coords = np.array([lm.coord for lm in lms])
        class_counts[cls] = len(lms)
        class_centroids[cls] = coords.mean(axis=0).tolist()

    return {
        "class_counts": class_counts,
        "class_centroids": class_centroids,
        "total_landmarks": scan.num_landmarks,
    }


def compute_symmetry_score(scan: DentalScan) -> dict:
    """Çene simetrisini landmark noktalarından hesaplar.

    Sol-sağ simetriyi ölçer. Düşük değerler daha simetrik yapıyı gösterir.
    """
    coords = scan.landmark_coords
    if len(coords) < 4:
        return {"symmetry_score": float("inf"), "midline_x": 0.0}

    # Orta hattı X ekseni ortalaması olarak tahmin et
    midline_x = float(coords[:, 0].mean())

    # Sol ve sağ tarafa ayır
    left_mask = coords[:, 0] < midline_x
    right_mask = coords[:, 0] >= midline_x

    left_points = coords[left_mask].copy()
    right_points = coords[right_mask].copy()

    # Sol noktaları aynala (X ekseninde yansıt)
    left_mirrored = left_points.copy()
    left_mirrored[:, 0] = 2 * midline_x - left_mirrored[:, 0]

    # Aynalanan sol noktalar ile sağ noktalar arasındaki ortalama mesafe
    if len(left_mirrored) > 0 and len(right_points) > 0:
        cross_dist = cdist(left_mirrored, right_points)
        min_dists = cross_dist.min(axis=1)
        symmetry_score = float(min_dists.mean())
    else:
        symmetry_score = float("inf")

    return {
        "symmetry_score_mm": symmetry_score,
        "midline_x": midline_x,
        "left_count": int(left_mask.sum()),
        "right_count": int(right_mask.sum()),
    }


def generate_dataset_report(dataset: Teeth3DSDataset) -> dict:
    """Veri seti genelinde kapsamlı analiz raporu üretir."""
    all_dimensions = []
    all_symmetry = []
    class_totals = {}

    for scan in dataset.scans:
        dims = compute_arch_dimensions(scan)
        all_dimensions.append(dims)

        sym = compute_symmetry_score(scan)
        all_symmetry.append(sym)

        for cls, count in analyze_class_distribution(scan)["class_counts"].items():
            class_totals[cls] = class_totals.get(cls, 0) + count

    # İstatistikler
    widths = [d["width_mm"] for d in all_dimensions if d]
    depths = [d["depth_mm"] for d in all_dimensions if d]
    heights = [d["height_mm"] for d in all_dimensions if d]
    sym_scores = [s["symmetry_score_mm"] for s in all_symmetry if s.get("symmetry_score_mm") != float("inf")]

    report = {
        "dataset_info": dataset.get_statistics(),
        "arch_width": {
            "mean": float(np.mean(widths)) if widths else 0,
            "std": float(np.std(widths)) if widths else 0,
            "min": float(np.min(widths)) if widths else 0,
            "max": float(np.max(widths)) if widths else 0,
        },
        "arch_depth": {
            "mean": float(np.mean(depths)) if depths else 0,
            "std": float(np.std(depths)) if depths else 0,
            "min": float(np.min(depths)) if depths else 0,
            "max": float(np.max(depths)) if depths else 0,
        },
        "arch_height": {
            "mean": float(np.mean(heights)) if heights else 0,
            "std": float(np.std(heights)) if heights else 0,
        },
        "symmetry": {
            "mean_score": float(np.mean(sym_scores)) if sym_scores else 0,
            "std_score": float(np.std(sym_scores)) if sym_scores else 0,
        },
        "landmark_class_totals": class_totals,
    }

    return report
