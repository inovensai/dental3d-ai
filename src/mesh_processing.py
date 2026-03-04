"""3B mesh ön işleme modülü.

Mesh temizleme, düzeltme, normalizasyon ve örnekleme işlemlerini içerir.
"""
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def clean_mesh(mesh, remove_duplicates: bool = True, remove_degenerate: bool = True):
    """Mesh'i temizler: tekrarlayan vertex ve degenerate üçgenleri kaldırır.

    Args:
        mesh: Open3D TriangleMesh
        remove_duplicates: Tekrarlayan vertex'leri kaldır
        remove_degenerate: Degenerate üçgenleri kaldır

    Returns:
        Temizlenmiş mesh
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    if remove_duplicates:
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()

    if remove_degenerate:
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()
    return mesh


def normalize_mesh(mesh) -> tuple:
    """Mesh'i birim küreye normalize eder.

    Returns:
        (normalized_mesh, center, scale) - geri dönüşüm için merkez ve ölçek
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    vertices_centered = vertices - center
    scale = np.max(np.linalg.norm(vertices_centered, axis=1))

    mesh_normalized = o3d.geometry.TriangleMesh(mesh)
    mesh_normalized.vertices = o3d.utility.Vector3dVector(vertices_centered / scale)
    mesh_normalized.compute_vertex_normals()

    return mesh_normalized, center, scale


def compute_mesh_stats(mesh) -> dict:
    """Mesh istatistiklerini hesaplar."""
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # Üçgen alanlarını hesapla
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    return {
        "num_vertices": len(vertices),
        "num_triangles": len(triangles),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_size_mm": bbox_size.tolist(),
        "total_surface_area": float(areas.sum()),
        "mean_triangle_area": float(areas.mean()),
        "is_watertight": mesh.is_watertight(),
    }


def sample_points_from_mesh(mesh, num_points: int = 10000, method: str = "uniform"):
    """Mesh yüzeyinden nokta örnekleme yapar.

    Args:
        mesh: Open3D TriangleMesh
        num_points: Örneklenecek nokta sayısı
        method: "uniform" veya "poisson"

    Returns:
        Open3D PointCloud
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    if method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    elif method == "poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    else:
        raise ValueError(f"Bilinmeyen örnekleme metodu: {method}")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )
    return pcd


def compute_fpfh_features(pcd, radius: float = 1.0, max_nn: int = 100):
    """FPFH (Fast Point Feature Histograms) özellik vektörlerini hesaplar.

    Bu özellikler RANSAC tabanlı kaba hizalama için kullanılır.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 0.5, max_nn=max_nn)
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn),
    )
    return fpfh


def downsample_pointcloud(pcd, voxel_size: float = 0.3):
    """Nokta bulutunu voxel downsampling ile azaltır."""
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    return pcd_down


def crop_region(pcd, center: np.ndarray, radius: float):
    """Belirli bir merkez ve yarıçap etrafındaki noktaları kırpar.

    Referans bölge (palatal rugae vb.) seçimi için kullanılır.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d gerekli.")

    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points - center, axis=1)
    mask = distances <= radius
    indices = np.where(mask)[0]

    cropped = pcd.select_by_index(indices.tolist())
    return cropped
