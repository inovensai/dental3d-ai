"""Değişim analizi modülü.

Çakıştırılmış (registered) 3B modeller arasındaki yüzey
değişimlerini hesaplar ve analiz eder. Çürük, aşınma, dişeti
çekilmesi gibi patolojilerin tespiti için kullanılır.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from scipy.spatial import KDTree


@dataclass
class ChangeAnalysisResult:
    """Değişim analizi sonuçlarını tutan veri sınıfı."""
    distances: np.ndarray        # Her nokta için mesafe değeri (mm)
    mean_distance: float         # Ortalama mesafe
    max_distance: float          # Maksimum mesafe
    std_distance: float          # Mesafe standart sapması
    hausdorff_distance: float    # Hausdorff mesafesi
    volume_change: Optional[float] = None  # Hacim değişimi (mm³)
    significant_change_ratio: float = 0.0  # Anlamlı değişim oranı

    @property
    def summary(self) -> str:
        return (
            f"Ortalama mesafe: {self.mean_distance:.4f} mm\n"
            f"Maksimum mesafe: {self.max_distance:.4f} mm\n"
            f"Std sapma: {self.std_distance:.4f} mm\n"
            f"Hausdorff mesafesi: {self.hausdorff_distance:.4f} mm\n"
            f"Anlamlı değişim oranı: {self.significant_change_ratio:.2%}"
        )


def compute_point_to_point_distances(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """İki nokta bulutu arasındaki en yakın nokta mesafelerini hesaplar.

    Args:
        source_points: Kaynak noktaları (Nx3)
        target_points: Hedef noktaları (Mx3)

    Returns:
        Her kaynak noktası için en yakın hedef noktasına olan mesafe (N,)
    """
    tree = KDTree(target_points)
    distances, _ = tree.query(source_points)
    return distances


def compute_signed_distances(
    source_points: np.ndarray,
    source_normals: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """İşaretli (signed) mesafe hesaplama.

    Normal vektörleri kullanarak mesafenin yönünü (madde kaybı/birikimi)
    belirler. Negatif = madde kaybı, Pozitif = madde birikimi.
    """
    tree = KDTree(target_points)
    distances, indices = tree.query(source_points)

    # Mesafe yönünü normal vektörü ile belirle
    diff_vectors = target_points[indices] - source_points
    signs = np.sign(np.sum(diff_vectors * source_normals, axis=1))
    signed_distances = distances * signs

    return signed_distances


def compute_hausdorff_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> float:
    """İki nokta bulutu arasındaki Hausdorff mesafesini hesaplar.

    Hausdorff mesafesi = max(max(d(a,B)), max(d(b,A)))
    İki yüzey arasındaki en kötü durum mesafesini ölçer.
    """
    tree_a = KDTree(points_a)
    tree_b = KDTree(points_b)

    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)

    return float(max(dist_a_to_b.max(), dist_b_to_a.max()))


def analyze_changes(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_normals: Optional[np.ndarray] = None,
    threshold_mm: float = 0.5,
) -> ChangeAnalysisResult:
    """İki çakıştırılmış model arasındaki değişimleri analiz eder.

    Args:
        source_points: T0 (önceki) tarama noktaları
        target_points: T1 (sonraki) tarama noktaları
        source_normals: Kaynak normal vektörleri (işaretli mesafe için)
        threshold_mm: Anlamlı değişim eşiği (mm)

    Returns:
        ChangeAnalysisResult
    """
    if source_normals is not None:
        distances = compute_signed_distances(source_points, source_normals, target_points)
    else:
        distances = compute_point_to_point_distances(source_points, target_points)

    hausdorff = compute_hausdorff_distance(source_points, target_points)

    abs_distances = np.abs(distances)
    significant_mask = abs_distances > threshold_mm
    significant_ratio = significant_mask.sum() / len(distances)

    return ChangeAnalysisResult(
        distances=distances,
        mean_distance=float(abs_distances.mean()),
        max_distance=float(abs_distances.max()),
        std_distance=float(abs_distances.std()),
        hausdorff_distance=hausdorff,
        significant_change_ratio=float(significant_ratio),
    )


def classify_change_regions(
    distances: np.ndarray,
    threshold_loss: float = -0.3,
    threshold_gain: float = 0.3,
) -> np.ndarray:
    """Değişim bölgelerini sınıflandırır.

    Args:
        distances: İşaretli mesafe değerleri
        threshold_loss: Madde kaybı eşiği (negatif, mm)
        threshold_gain: Madde birikimi eşiği (pozitif, mm)

    Returns:
        Sınıf etiketleri: 0=değişim yok, 1=madde kaybı, 2=madde birikimi
    """
    labels = np.zeros(len(distances), dtype=int)
    labels[distances < threshold_loss] = 1  # Madde kaybı (aşınma, çürük)
    labels[distances > threshold_gain] = 2  # Madde birikimi (kalkülüs vb.)
    return labels


def compute_regional_statistics(
    points: np.ndarray,
    distances: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 33,
) -> dict:
    """Her diş bölgesi için istatistikleri hesaplar.

    Args:
        points: Nokta koordinatları
        distances: Mesafe değerleri
        labels: Segmentasyon etiketleri (diş numaraları)
        num_classes: Toplam sınıf sayısı

    Returns:
        Diş bazında istatistikler
    """
    stats = {}
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() == 0:
            continue

        cls_distances = distances[mask]
        stats[cls] = {
            "num_points": int(mask.sum()),
            "mean_distance": float(np.abs(cls_distances).mean()),
            "max_distance": float(np.abs(cls_distances).max()),
            "std_distance": float(cls_distances.std()),
            "material_loss": float((cls_distances < -0.3).sum() / mask.sum()),
            "material_gain": float((cls_distances > 0.3).sum() / mask.sum()),
        }

    return stats


def generate_risk_scores(
    regional_stats: dict,
    weights: Optional[dict] = None,
) -> dict:
    """Her diş bölgesi için risk skoru üretir.

    Risk skoru 0-100 arası bir değerdir.
    Yüksek değerler daha fazla patoloji riski gösterir.

    Args:
        regional_stats: compute_regional_statistics çıktısı
        weights: Özellik ağırlıkları

    Returns:
        Diş bazında risk skorları
    """
    if weights is None:
        weights = {
            "mean_distance": 0.3,
            "max_distance": 0.2,
            "material_loss": 0.3,
            "std_distance": 0.2,
        }

    risk_scores = {}
    for cls, stats in regional_stats.items():
        score = 0.0
        # Ortalama mesafe (0-2mm -> 0-100)
        score += weights["mean_distance"] * min(stats["mean_distance"] / 2.0, 1.0) * 100
        # Maksimum mesafe (0-5mm -> 0-100)
        score += weights["max_distance"] * min(stats["max_distance"] / 5.0, 1.0) * 100
        # Madde kaybı oranı (0-1 -> 0-100)
        score += weights["material_loss"] * stats["material_loss"] * 100
        # Varyans (0-1mm -> 0-100)
        score += weights["std_distance"] * min(stats["std_distance"] / 1.0, 1.0) * 100

        risk_scores[cls] = {
            "score": round(min(score, 100), 1),
            "level": _risk_level(score),
        }

    return risk_scores


def _risk_level(score: float) -> str:
    if score < 20:
        return "Dusuk"
    elif score < 50:
        return "Orta"
    elif score < 75:
        return "Yuksek"
    else:
        return "Kritik"
