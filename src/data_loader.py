"""Teeth3DS+ veri seti yükleme ve yönetim modülü.

OBJ mesh dosyaları ve JSON segmentasyon etiketlerini yükler.
Her JSON dosyasında:
  - labels:    vertex bazlı FDI diş numaraları (0=dişeti, 11-48=diş)
  - instances: vertex bazlı instance ID'leri
  - id_patient: hasta kimliği
  - jaw:       "upper" veya "lower"

Veri yapısı:
  data_part_{1..7}/
    lower/ upper/
      {PATIENT_ID}/
        {PATIENT_ID}_{jaw}.obj   <- 3B mesh
        {PATIENT_ID}_{jaw}.json  <- segmentasyon etiketleri
"""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


# ═══════════════════════════════════════════════════════
#  DentalMesh: Tek bir çene taraması (OBJ + labels)
# ═══════════════════════════════════════════════════════
@dataclass
class DentalMesh:
    """Bir hastanın tek çene taramasını temsil eder."""
    patient_id: str
    jaw_type: str               # "upper" veya "lower"
    obj_path: Path              # OBJ dosya yolu
    json_path: Optional[Path]   # JSON label dosyası (test setinde olmayabilir)
    part_num: int = 0           # Hangi data_part klasöründen geldiği
    _vertices: Optional[np.ndarray] = field(default=None, repr=False)
    _faces: Optional[np.ndarray] = field(default=None, repr=False)
    _labels: Optional[np.ndarray] = field(default=None, repr=False)
    _instances: Optional[np.ndarray] = field(default=None, repr=False)

    # ─── Özellikler ───
    @property
    def has_labels(self) -> bool:
        return self.json_path is not None and self.json_path.exists()

    @property
    def vertices(self) -> np.ndarray:
        if self._vertices is None:
            self.load_mesh()
        return self._vertices

    @property
    def faces(self) -> np.ndarray:
        if self._faces is None:
            self.load_mesh()
        return self._faces

    @property
    def labels(self) -> Optional[np.ndarray]:
        if self._labels is None and self.has_labels:
            self.load_labels()
        return self._labels

    @property
    def instances(self) -> Optional[np.ndarray]:
        if self._instances is None and self.has_labels:
            self.load_labels()
        return self._instances

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def unique_teeth(self) -> list[int]:
        """Mevcut FDI diş numaralarını döndürür (0=dişeti hariç)."""
        if self.labels is None:
            return []
        return sorted([int(x) for x in set(self.labels) if x != 0])

    @property
    def num_teeth(self) -> int:
        return len(self.unique_teeth)

    # ─── Yükleme ───
    def load_mesh(self) -> "trimesh.Trimesh":
        """OBJ mesh dosyasını yükler ve vertex/face bilgilerini saklar."""
        if not HAS_TRIMESH:
            raise ImportError("trimesh gerekli: pip install trimesh")
        mesh = trimesh.load(str(self.obj_path), process=False)
        self._vertices = np.array(mesh.vertices, dtype=np.float32)
        self._faces = np.array(mesh.faces, dtype=np.int64)
        return mesh

    def load_labels(self) -> Optional[np.ndarray]:
        """JSON segmentasyon etiketlerini yükler."""
        if not self.has_labels:
            return None
        with open(self.json_path, "r") as f:
            data = json.load(f)
        self._labels = np.array(data["labels"], dtype=np.int32)
        self._instances = np.array(data["instances"], dtype=np.int32)
        return self._labels

    # ─── Diş İşlemleri ───
    def get_tooth_vertices(self, fdi_number: int) -> np.ndarray:
        """Belirli bir dişin vertex koordinatlarını döndürür."""
        if self.labels is None:
            return np.array([]).reshape(0, 3)
        mask = self.labels == fdi_number
        return self.vertices[mask]

    def get_tooth_mask(self, fdi_number: int) -> np.ndarray:
        """Belirli bir diş için boolean maske."""
        if self.labels is None:
            return np.array([], dtype=bool)
        return self.labels == fdi_number

    def get_gingiva_vertices(self) -> np.ndarray:
        """Dişeti vertex'lerini döndürür."""
        return self.get_tooth_vertices(0)

    def get_centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def get_tooth_stats(self) -> dict:
        """Her diş için vertex sayısı, centroid, bbox boyutu döndürür."""
        if self.labels is None:
            return {}
        stats = {}
        counts = Counter(self.labels.tolist())
        for fdi, count in sorted(counts.items()):
            tooth_verts = self.vertices[self.labels == fdi]
            stats[int(fdi)] = {
                "vertex_count": count,
                "centroid": tooth_verts.mean(axis=0).tolist(),
                "bbox_size_mm": (tooth_verts.max(axis=0) - tooth_verts.min(axis=0)).tolist(),
            }
        return stats

    # ─── Örnekleme ───
    def sample_points(self, num_points: int = 24000) -> dict:
        """Mesh yüzeyinden rastgele noktalar örnekler (eğitim için).

        Returns:
            {"points": (N,3), "normals": (N,3), "labels": (N,)}
        """
        if not HAS_TRIMESH:
            raise ImportError("trimesh gerekli")

        mesh = trimesh.Trimesh(
            vertices=self.vertices, faces=self.faces, process=False
        )

        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
        points = np.array(points, dtype=np.float32)

        finite_mask = np.all(np.isfinite(points), axis=1)
        if not np.all(finite_mask):
            points[~finite_mask] = 0.0

        normals = np.array(mesh.face_normals[face_indices], dtype=np.float32)

        result = {"points": points, "normals": normals}

        if self.labels is not None:
            face_verts = self.faces[face_indices]  # (N, 3)
            result["labels"] = self.labels[face_verts[:, 0]]

        return result

    def unload(self):
        """Belleği serbest bırakır."""
        self._vertices = self._faces = self._labels = self._instances = None

    def __repr__(self):
        lbl = "labeled" if self.has_labels else "unlabeled"
        return f"DentalMesh({self.patient_id}, {self.jaw_type}, {lbl})"


# ═══════════════════════════════════════════════════════
#  Teeth3DSDataset: Veri seti yöneticisi
# ═══════════════════════════════════════════════════════
class Teeth3DSDataset:
    """data_part_1..7 klasörlerindeki OBJ + JSON dosyalarını yönetir.

    Toplamda ~1900 OBJ mesh + ~1800 JSON label barındırır.
    """

    def __init__(self, data_root: str | Path, parts: list[int] | None = None):
        """
        Args:
            data_root: Proje kök dizini (data_part_* klasörlerini içeren)
            parts: Yüklenecek part numaraları (default: [1..7])
        """
        self.data_root = Path(data_root)
        self.parts = parts or list(range(1, 8))
        self.scans: list[DentalMesh] = []
        self._build_index()

    def _build_index(self):
        """Tüm data_part klasörlerini tarayarak index oluşturur (lazy - mesh yüklemez)."""
        for part_num in self.parts:
            part_dir = self.data_root / f"data_part_{part_num}"
            if not part_dir.exists():
                continue
            for jaw_type in ["lower", "upper"]:
                jaw_dir = part_dir / jaw_type
                if not jaw_dir.exists():
                    continue
                for patient_dir in sorted(jaw_dir.iterdir()):
                    if not patient_dir.is_dir():
                        continue

                    patient_id = patient_dir.name
                    obj_files = list(patient_dir.glob("*.obj"))
                    json_files = list(patient_dir.glob("*.json"))

                    if not obj_files:
                        continue

                    self.scans.append(DentalMesh(
                        patient_id=patient_id,
                        jaw_type=jaw_type,
                        obj_path=obj_files[0],
                        json_path=json_files[0] if json_files else None,
                        part_num=part_num,
                    ))

    # ─── Filtreleme ───
    def get_labeled_scans(self) -> list[DentalMesh]:
        """Label'lı (JSON'u olan) taramaları döndürür."""
        return [s for s in self.scans if s.has_labels]

    def get_unlabeled_scans(self) -> list[DentalMesh]:
        """Label'sız taramaları döndürür (test amaçlı)."""
        return [s for s in self.scans if not s.has_labels]

    def get_patient_ids(self) -> list[str]:
        return sorted(set(s.patient_id for s in self.scans))

    def get_scan(self, patient_id: str, jaw_type: str) -> Optional[DentalMesh]:
        for s in self.scans:
            if s.patient_id == patient_id and s.jaw_type == jaw_type:
                return s
        return None

    def get_patient_scans(self, patient_id: str) -> list[DentalMesh]:
        return [s for s in self.scans if s.patient_id == patient_id]

    def get_by_jaw(self, jaw_type: str) -> list[DentalMesh]:
        return [s for s in self.scans if s.jaw_type == jaw_type]

    # ─── İstatistikler ───
    def get_statistics(self) -> dict:
        labeled = self.get_labeled_scans()
        unlabeled = self.get_unlabeled_scans()
        return {
            "total_scans": len(self.scans),
            "labeled_scans": len(labeled),
            "unlabeled_scans": len(unlabeled),
            "upper_scans": sum(1 for s in self.scans if s.jaw_type == "upper"),
            "lower_scans": sum(1 for s in self.scans if s.jaw_type == "lower"),
            "unique_patients": len(self.get_patient_ids()),
            "data_parts": len(self.parts),
        }

    # ─── Train/Test Split ───
    def split_train_val(self, val_ratio: float = 0.15, seed: int = 42):
        """Label'lı verileri hasta bazlı train/val olarak ayırır."""
        labeled = self.get_labeled_scans()
        patient_ids = sorted(set(s.patient_id for s in labeled))

        rng = np.random.RandomState(seed)
        rng.shuffle(patient_ids)

        n_val = max(1, int(len(patient_ids) * val_ratio))
        val_ids = set(patient_ids[:n_val])
        train_ids = set(patient_ids[n_val:])

        train = [s for s in labeled if s.patient_id in train_ids]
        val = [s for s in labeled if s.patient_id in val_ids]
        return train, val

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        return self.scans[idx]

    def __repr__(self):
        return (
            f"Teeth3DSDataset(total={len(self.scans)}, "
            f"labeled={len(self.get_labeled_scans())}, "
            f"parts={self.parts})"
        )
