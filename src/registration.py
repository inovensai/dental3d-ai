"""3B Mesh Registration (Çakıştırma/Süperimpozisyon) modülü.

İki aşamalı registration pipeline:
1. Kaba hizalama (RANSAC + FPFH features)
2. İnce hizalama (ICP Point-to-Plane)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


@dataclass
class RegistrationResult:
    """Registration sonuçlarını tutan veri sınıfı."""
    transformation: np.ndarray  # 4x4 dönüşüm matrisi
    fitness: float              # [0, 1] arası uyum skoru
    inlier_rmse: float          # RMSE (mm)
    correspondence_set: Optional[np.ndarray] = None

    @property
    def rotation(self) -> np.ndarray:
        """3x3 rotasyon matrisini döndürür."""
        return self.transformation[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """Öteleme vektörünü döndürür."""
        return self.transformation[:3, 3]

    def __repr__(self):
        return (
            f"RegistrationResult(\n"
            f"  fitness={self.fitness:.4f},\n"
            f"  inlier_rmse={self.inlier_rmse:.4f} mm,\n"
            f"  translation={self.translation}\n"
            f")"
        )


def ransac_registration(
    source_pcd,
    target_pcd,
    source_fpfh,
    target_fpfh,
    max_correspondence_distance: float = 2.0,
    max_iteration: int = 100000,
    confidence: float = 0.999,
) -> RegistrationResult:
    """RANSAC tabanlı kaba hizalama.

    FPFH feature eşleştirmesi ile global registration yapar.

    Args:
        source_pcd: Kaynak nokta bulutu
        target_pcd: Hedef nokta bulutu
        source_fpfh: Kaynak FPFH özellikleri
        target_fpfh: Hedef FPFH özellikleri
        max_correspondence_distance: Maks. eşleştirme mesafesi (mm)
        max_iteration: Maks. iterasyon sayısı
        confidence: Yakınsama güven eşiği

    Returns:
        RegistrationResult
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd,
        target_pcd,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration, confidence
        ),
    )

    return RegistrationResult(
        transformation=np.array(result.transformation),
        fitness=result.fitness,
        inlier_rmse=result.inlier_rmse,
    )


def icp_registration(
    source_pcd,
    target_pcd,
    max_correspondence_distance: float = 0.5,
    init_transformation: Optional[np.ndarray] = None,
    max_iteration: int = 200,
    method: str = "point_to_plane",
) -> RegistrationResult:
    """ICP (Iterative Closest Point) ile ince hizalama.

    Args:
        source_pcd: Kaynak nokta bulutu (normalleri hesaplanmış)
        target_pcd: Hedef nokta bulutu (normalleri hesaplanmış)
        max_correspondence_distance: Maks. eşleştirme mesafesi (mm)
        init_transformation: Başlangıç dönüşüm matrisi (4x4)
        max_iteration: Maks. iterasyon sayısı
        method: "point_to_point" veya "point_to_plane"

    Returns:
        RegistrationResult
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    if init_transformation is None:
        init_transformation = np.eye(4)

    if method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif method == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        raise ValueError(f"Bilinmeyen ICP metodu: {method}")

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iteration,
        relative_fitness=1e-6,
        relative_rmse=1e-6,
    )

    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance,
        init_transformation,
        estimation,
        criteria,
    )

    return RegistrationResult(
        transformation=np.array(result.transformation),
        fitness=result.fitness,
        inlier_rmse=result.inlier_rmse,
        correspondence_set=np.asarray(result.correspondence_set) if result.correspondence_set else None,
    )


def full_registration_pipeline(
    source_mesh,
    target_mesh,
    num_points: int = 50000,
    voxel_size: float = 0.3,
    ransac_distance: float = 2.0,
    icp_distance: float = 0.5,
) -> RegistrationResult:
    """Tam registration pipeline: Kaba (RANSAC) + İnce (ICP) hizalama.

    Args:
        source_mesh: Kaynak mesh (Open3D TriangleMesh)
        target_mesh: Hedef mesh (Open3D TriangleMesh)
        num_points: Örneklenecek nokta sayısı
        voxel_size: Voxel downsampling boyutu (mm)
        ransac_distance: RANSAC eşleştirme mesafesi
        icp_distance: ICP eşleştirme mesafesi

    Returns:
        Nihai RegistrationResult
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    # 1. Nokta bulutu oluştur
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=num_points)
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=num_points)

    # 2. Downsampling
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # 3. Normal hesaplama
    radius_normal = voxel_size * 2
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    source_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    target_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # 4. FPFH feature hesaplama
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    # 5. Kaba hizalama (RANSAC)
    ransac_result = ransac_registration(
        source_down, target_down,
        source_fpfh, target_fpfh,
        max_correspondence_distance=ransac_distance,
    )
    print(f"RANSAC sonucu - Fitness: {ransac_result.fitness:.4f}, RMSE: {ransac_result.inlier_rmse:.4f}")

    # 6. İnce hizalama (ICP Point-to-Plane)
    icp_result = icp_registration(
        source_pcd,
        target_pcd,
        max_correspondence_distance=icp_distance,
        init_transformation=ransac_result.transformation,
        method="point_to_plane",
    )
    print(f"ICP sonucu - Fitness: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.4f}")

    return icp_result


def landmark_based_registration(
    source_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
) -> RegistrationResult:
    """Landmark tabanlı rigid registration.

    Eşleştirilmiş landmark noktaları kullanarak SVD ile
    optimal rotasyon ve öteleme hesaplar.

    Args:
        source_landmarks: Kaynak landmark koordinatları (Nx3)
        target_landmarks: Hedef landmark koordinatları (Nx3)

    Returns:
        RegistrationResult
    """
    assert source_landmarks.shape == target_landmarks.shape, \
        "Kaynak ve hedef landmark sayıları eşit olmalı"

    # Merkezleme
    source_center = source_landmarks.mean(axis=0)
    target_center = target_landmarks.mean(axis=0)

    source_centered = source_landmarks - source_center
    target_centered = target_landmarks - target_center

    # SVD ile optimal rotasyon
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Yansıma düzeltmesi
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = target_center - R @ source_center

    # 4x4 dönüşüm matrisi
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    # RMSE hesapla
    transformed = (R @ source_landmarks.T).T + t
    errors = np.linalg.norm(transformed - target_landmarks, axis=1)
    rmse = np.sqrt(np.mean(errors**2))

    return RegistrationResult(
        transformation=transformation,
        fitness=1.0,
        inlier_rmse=rmse,
    )
