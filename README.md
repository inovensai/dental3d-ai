# DentalAI - 3B Intraoral Tarama Analiz Platformu

Dis hekimligi alaninda 3B intraoral taramalardan elde edilen dijital modellerin yapay zeka ile analiz edilmesi icin gelistirilmis prototip projesi. Teeth3DS+ veri seti uzerinde 1900 gercek OBJ mesh ve 1800 vertex-seviyesi segmentasyon etiketi ile calisir.

## Proje Amaci

Bu proje, bir hastanin farkli zamanlarda alinan agiz ici 3B taramalarini hizalayarak (register ederek) aradaki degisimleri otomatik olarak tespit etmeyi amaclar. Temel hedefler:

- **3B Mesh Cakistirma (Superimposition)**: ICP + RANSAC tabanli registration pipeline
- **Dis Segmentasyonu**: PointNet tabanli otomatik dis bolgeleme (33 sinif: 32 dis + diseti)
- **Hastalik Tespiti**: Curuk, asinma, diseti cekilmesi gibi patolojilerin tespiti
- **Degisim Analizi**: Temporal taramalar arasi yuzey degisimi olcumu
- **Risk Skorlama**: Dis bazinda risk degerlendirmesi ve tahmin

## Proje Yapisi

```
inovens-dis/
├── app.py                          # Streamlit web dashboard (5 sayfa)
├── train.py                        # PointNet model egitim scripti
├── run_demo.py                     # Demo calistirma scripti
├── config.py                       # Proje konfigurasyonu (FDI mapping vb.)
├── requirements.txt                # Python bagimliliklari
├── README.md                       # Bu dosya
│
├── src/                            # Kaynak kod modulleri
│   ├── __init__.py
│   ├── data_loader.py              # OBJ mesh + JSON label yukleme (DentalMesh, Teeth3DSDataset)
│   ├── mesh_processing.py          # Open3D tabanli mesh on isleme
│   ├── registration.py             # ICP + RANSAC registration pipeline
│   ├── segmentation.py             # PointNet segmentasyon + PyTorch Dataset
│   ├── change_analysis.py          # Degisim analizi ve risk skorlama
│   ├── visualization.py            # Plotly 3B gorsellestirme
│   └── landmark_analysis.py        # Landmark tabanli dental analiz
│
├── tests/                          # Test dosyalari
│   └── test_pipeline.py            # Pipeline testleri (7 test)
│
├── models/                         # Egitilmis model dosyalari
├── outputs/                        # Cikti dosyalari
│
├── data_part_1/                    # Mesh verisi - part 1 (150 hasta x 2 cene)
│   ├── lower/                      #   Alt cene OBJ + JSON
│   │   └── {PATIENT_ID}/
│   │       ├── {PATIENT_ID}_lower.obj
│   │       └── {PATIENT_ID}_lower.json
│   └── upper/                      #   Ust cene OBJ + JSON
│       └── {PATIENT_ID}/
│           ├── {PATIENT_ID}_upper.obj
│           └── {PATIENT_ID}_upper.json
├── data_part_2/                    # Part 2 (150 hasta x 2 cene)
├── data_part_3/                    # Part 3 (150 hasta x 2 cene)
├── data_part_4/                    # Part 4 (150 hasta x 2 cene)
├── data_part_5/                    # Part 5 (150 hasta x 2 cene)
├── data_part_6/                    # Part 6 (150 hasta x 2 cene)
├── data_part_7/                    # Part 7 (50 hasta x 2 cene, etiketsiz test seti)
│
├── 3DTeethLand_landmarks_train/    # Landmark egitim verisi (120 hasta x 2 cene)
│   ├── lower/
│   └── upper/
│
└── 3DTeethLand_landmarks_test/     # Landmark test verisi (50 hasta x 2 cene)
    ├── lower/
    └── upper/
```

## Veri Seti: Teeth3DS+

MICCAI konferansi kapsaminda olusturulmus ilk kapsamli acik kaynak 3B agiz ici tarama veri setidir.

| Ozellik | Deger |
|---------|-------|
| Toplam OBJ mesh | 1.900 intraoral tarama |
| Etiketli tarama | 1.800 (vertex-seviye JSON segmentasyon) |
| Etiketsiz (test) | 100 (data_part_7) |
| Hasta sayisi | ~950 hasta (ust ve alt cene ayri) |
| Mesh boyutu | ~93.000 vertex, ~186.000 face / tarama |
| Fiziksel boyut | ~80 x 58 x 40 mm (ortalama cene) |
| Sinif sayisi | 33 (32 dis + 1 diseti) |
| Etiket formati | FDI numaralama (0=diseti, 11-48=disler) |
| Dogrulama | 5+ yil deneyimli ortodontist ve cerrahlar |

### Veri Yapisi

**OBJ dosyasi:** Standart 3B mesh formati (vertex + face)
```
v -19.1730 -29.5240 -8.5000    # vertex koordinatlari (x, y, z) mm
v -19.1280 -29.5850 -8.5210
...
f 1 2 3                         # face (ucgen) tanimlari
f 2 3 4
```

**JSON dosyasi:** Her vertex icin FDI dis numarasi etiketi
```json
{
  "id_patient": "O52P1SZT",
  "jaw": "lower",
  "labels": [0, 0, 0, 31, 31, 32, ...],    // 93.288 vertex icin FDI etiketi
  "instances": [0, 0, 0, 1, 1, 2, ...]      // Dis instance ID'leri
}
```

### FDI Dis Numaralama Sistemi

```
         Ust Cene
   18-11  |  21-28
   ───────┼───────
   48-41  |  31-38
         Alt Cene

0  = Diseti (Gingiva)
11 = Ust sag santral kesici    ...  18 = Ust sag 3. molar
21 = Ust sol santral kesici    ...  28 = Ust sol 3. molar
31 = Alt sol santral kesici    ...  38 = Alt sol 3. molar
41 = Alt sag santral kesici    ...  48 = Alt sag 3. molar
```

## Kurulum

### Gereksinimler

- Python 3.10+
- macOS / Linux / Windows

### Adim Adim Kurulum

```bash
# 1. Proje dizinine gidin
cd inovens-dis

# 2. Sanal ortam olusturun (onerilen)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Bagimliliklari yukleyin
pip install -r requirements.txt

# 4. PyTorch yukleyin (segmentasyon ve egitim icin gerekli)
# macOS (MPS destegi):
pip install torch torchvision

# Linux/Windows (CUDA destegi):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Kullanim

### 1. Demo Calistirma

Projenin tum modullerini gercek OBJ mesh verileri ile bastan sona test eder:

```bash
python run_demo.py
```

Cikti ornegi:
```
============================================================
  DentalAI - Gercek Veri Demo (OBJ + JSON)
============================================================

[1/7] Veri seti yukleniyor...
  Toplam: 1900 tarama (data_part_1..7)
  Etiketli: 1800, Etiketsiz (test): 100
  Hasta sayisi: 950

[2/7] Gercek mesh yukleniyor...
  Hasta: O52P1SZT, Cene: lower
  Vertex: 93,288 | Face: 186,499
  Boyut: 80.2 x 58.1 x 39.7 mm

[3/7] Segmentasyon etiketleri...
  Dis sayisi: 14
  FDI numaralari: [0, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45]
  FDI 31 vertex sayisi: 4,521

[4/7] Nokta ornekleme (mesh yuzeyinden)...
  10,000 nokta orneklendi
  Noktalar: (10000, 3), Normaller: (10000, 3), Etiketler: (10000,)

[5/7] Degisim analizi (gercek mesh uzerinde simulasyon)...
  Hausdorff mesafesi: 4.2143 mm
  Ortalama mesafe: 0.3847 mm
  Anlamli degisim orani: 18.7%
  Dis bazinda risk skorlari: 12 bolge

[6/7] Landmark registration testi...
  RMSE: 0.00000000 mm

[7/7] PointNet model kontrolu...
  Parametreler: 3,531,370
  Sinif sayisi: 33
  Giris: (2, 3, 5000) -> Cikis: (2, 33, 5000)
```

### 2. Model Egitimi

PointNet segmentasyon modelini gercek Teeth3DS+ verileri ile egitin:

```bash
# Varsayilan ayarlarla egitim (data_part_1-6, 50 epoch)
python train.py

# Ozel ayarlarla
python train.py --epochs 100 --batch 8 --num-points 24000

# Sadece ilk 2 part ile hizli test
python train.py --parts 1 2 --epochs 10

# Tum parametreler
python train.py --epochs 50 --batch 4 --lr 0.001 --num-points 24000 \
                --parts 1 2 3 4 5 6 --val-ratio 0.15
```

Egitim ciktisi:
```
[1/4] Veri seti yukleniyor (parts: [1, 2, 3, 4, 5, 6])...
  Toplam: 1800 tarama, Etiketli: 1800
  Egitim: 1530, Validasyon: 270

[2/4] PyTorch Dataset olusturuluyor (24000 nokta/ornek)...
  Egitim batch sayisi: 382
  Sinif sayisi: 33

[3/4] Model olusturuluyor...
  Parametre sayisi: 3,531,370
  Cihaz: cuda (veya mps/cpu)

[4/4] Egitim basliyor (50 epoch)...
  Epoch   1/50 | Train Loss: 2.1534, Acc: 0.4521 | Val Loss: 1.8234, Acc: 0.5123, mDSC: 0.3456
  Epoch   5/50 | Train Loss: 1.2345, Acc: 0.6789 | Val Loss: 1.1234, Acc: 0.7012, mDSC: 0.5678
  ...
```

### 3. Web Dashboard

Interaktif analiz platformunu baslatir:

```bash
streamlit run app.py
```

Dashboard sayfalari:

| Sayfa | Aciklama |
|-------|----------|
| **Genel Bakis** | Veri seti istatistikleri, ornek mesh metrikleri, dis dagilim grafigi |
| **3B Mesh Gorsellestirme** | Gercek OBJ mesh'in FDI etiketlerine gore renkli 3B goruntulenmesi |
| **Dis Analizi** | Tekil dis secimi ve 3B inceleme (boyut, vertex sayisi) |
| **Degisim Simulasyonu** | Gercek mesh uzerinde sentetik degisim ile risk analizi demo |
| **Egitim Bilgisi** | Model egitimi komutlari ve pipeline bilgisi |

### 4. Testleri Calistirma

```bash
python tests/test_pipeline.py
```

7 test modulu:
1. **Veri Seti Yukleme** - 1900 OBJ + 1800 JSON tarama
2. **OBJ Mesh Yukleme** - Vertex, face, label dogrulama
3. **Nokta Ornekleme** - Mesh yuzeyinden uniform sampling
4. **Degisim Analizi** - Mesafe, Hausdorff, risk hesaplama
5. **Landmark Registration** - SVD tabanli rigid registration
6. **PointNet Model** - Forward pass ve cikti boyut kontrolu
7. **PyTorch Dataset** - FDI->index donusumu, augmentasyon

## Modul Detaylari

### data_loader.py - Veri Yukleme

OBJ mesh + JSON segmentasyon etiketlerini yukler ve yonetir.

```python
from src.data_loader import Teeth3DSDataset, DentalMesh

# Veri setini yukle (tum data_part'lar)
dataset = Teeth3DSDataset(".", parts=[1, 2, 3, 4, 5, 6, 7])
stats = dataset.get_statistics()
print(f"Toplam: {stats['total_scans']}")     # 1900
print(f"Etiketli: {stats['labeled_scans']}")  # 1800

# Etiketli taramalari al
labeled = dataset.get_labeled_scans()

# Belirli hastanin taramasini al
scan = dataset.get_scan("O52P1SZT", "lower")

# Mesh yukle (lazy loading)
scan.load_mesh()
print(f"Vertex: {scan.num_vertices:,}")   # 93,288
print(f"Face: {scan.num_faces:,}")        # 186,499

# Label yukle
scan.load_labels()
print(f"Dis sayisi: {scan.num_teeth}")    # 14
print(f"FDI numaralari: {scan.unique_teeth}")

# Tekil dis vertex'lerini al
tooth_verts = scan.get_tooth_vertices(31)  # FDI 31 = alt sol santral kesici
print(f"FDI 31 vertex: {len(tooth_verts)}")

# Mesh yuzeyinden nokta ornekle (egitim icin)
sampled = scan.sample_points(num_points=24000)
# sampled["points"]: (24000, 3), sampled["normals"]: (24000, 3), sampled["labels"]: (24000,)

# Bellek temizle
scan.unload()

# Egitim/validasyon bolme (hasta bazinda, veri sizintisi yok)
train_scans, val_scans = dataset.split_train_val(val_ratio=0.15)
```

**Siniflar:**
- `DentalMesh`: Tek tarama veri sinifi (OBJ + JSON lazy loading)
- `Teeth3DSDataset`: Veri seti yonetici (data_part tarama, filtreleme, bolme)

### segmentation.py - Dis Segmentasyonu + PyTorch Dataset

PointNet tabanli 3B dis segmentasyon modeli ve egitim altyapisi.

```python
from src.segmentation import PointNetSegmentation, Teeth3DSTorchDataset, SegmentationTrainer
from config import FDI_TO_INDEX, NUM_CLASSES
import torch

# ─── PyTorch Dataset ───
scans = dataset.get_labeled_scans()
torch_ds = Teeth3DSTorchDataset(
    scans, num_points=24000,
    fdi_to_index=FDI_TO_INDEX,  # FDI -> ardisik indeks (0-32)
    augment=True,                # Random rotation + jitter
)
points, labels = torch_ds[0]    # (24000, 3), (24000,)

# ─── Model ───
model = PointNetSegmentation(num_classes=NUM_CLASSES)  # 33 sinif
total_params = sum(p.numel() for p in model.parameters())  # ~3.5M

# Tahmin
model.eval()
dummy = torch.randn(2, 3, 24000)  # (batch, xyz, num_points)
with torch.no_grad():
    output, _, _ = model(dummy)
    pred_labels = output.argmax(dim=1)  # (2, 24000)

# ─── Egitim ───
trainer = SegmentationTrainer(model, learning_rate=0.001)
print(f"Cihaz: {trainer.device}")  # cuda / mps / cpu

train_metrics = trainer.train_epoch(train_loader)
val_metrics = trainer.evaluate(val_loader)
print(f"Val mDSC: {val_metrics['mean_dice']:.4f}")

# Model kaydet/yukle
trainer.save_model("models/best_model.pth")
trainer.load_model("models/best_model.pth")
```

**Mimari:**
- **TNet**: Spatial Transformer Network (3x3 ve 64x64 donusum)
- **PointNetEncoder**: Lokal (64-dim) + global (1024-dim) feature cikarimi
- **PointNetSegmentation**: Concat [64 + 1024] -> MLP -> 33 sinif/nokta

**Metrikler:** Dice Similarity Coefficient (DSC), Accuracy

### registration.py - 3B Cakistirma (Registration)

Iki asamali registration pipeline.

```python
from src.registration import full_registration_pipeline, landmark_based_registration

# Yontem 1: Tam pipeline (OBJ mesh dosyalari ile)
result = full_registration_pipeline(mesh_t0, mesh_t1)
print(f"Fitness: {result.fitness:.4f}")
print(f"RMSE: {result.inlier_rmse:.4f} mm")

# Yontem 2: Landmark tabanli (SVD rigid registration)
import numpy as np
result = landmark_based_registration(source_points, target_points)
print(f"Donusum matrisi:\n{result.transformation}")
```

**Pipeline Akisi:**
1. **Kaba Hizalama (RANSAC)**: FPFH feature eslestirmesi ile global registration
2. **Ince Hizalama (ICP)**: Point-to-Plane ICP ile hassas cakistirma

### change_analysis.py - Degisim Analizi

Cakistirilmis modeller arasi yuzey degisim analizi.

```python
from src.change_analysis import (
    analyze_changes, generate_risk_scores,
    compute_regional_statistics, classify_change_regions,
)

# Degisim analizi
result = analyze_changes(source_points, target_points, threshold_mm=0.5)
print(f"Ortalama mesafe: {result.mean_distance:.4f} mm")
print(f"Hausdorff mesafesi: {result.hausdorff_distance:.4f} mm")
print(f"Anlamli degisim: {result.significant_change_ratio:.1%}")

# Dis bazinda bolgesel istatistikler (FDI etiketleri ile)
regional = compute_regional_statistics(points, distances, labels, num_classes=33)

# Risk skorlari
risk_scores = generate_risk_scores(regional)
# Cikti: {31: {"score": 45.3, "level": "Orta"}, 32: {"score": 12.1, "level": "Dusuk"}, ...}
```

**Risk Seviyeleri:**
| Skor | Seviye |
|------|--------|
| 0-20 | Dusuk |
| 20-50 | Orta |
| 50-75 | Yuksek |
| 75-100 | Kritik |

### visualization.py - Gorsellestirme

Plotly tabanli interaktif 3B gorsellestirmeler.

```python
from src.visualization import plot_pointcloud_with_distances, plot_risk_heatmap

# Yuzey mesafe haritasi
fig = plot_pointcloud_with_distances(points, distances, title="Degisim Haritasi")
fig.show()

# Risk isitma haritasi
fig = plot_risk_heatmap(risk_scores)
fig.show()
```

### mesh_processing.py - Mesh On Isleme

Open3D tabanli mesh temizleme ve isleme fonksiyonlari.

```python
from src.mesh_processing import clean_mesh, normalize_mesh, compute_mesh_stats

import open3d as o3d
mesh = o3d.io.read_triangle_mesh("hasta.obj")
mesh = clean_mesh(mesh)
mesh_norm, center, scale = normalize_mesh(mesh)
stats = compute_mesh_stats(mesh)
```

### landmark_analysis.py - Landmark Analizi

3DTeethLand landmark verileri ile calisan dental analiz fonksiyonlari.

```python
from src.landmark_analysis import compute_arch_dimensions, compute_symmetry_score

dims = compute_arch_dimensions(scan)
sym = compute_symmetry_score(scan)
report = generate_dataset_report(dataset)
```

## Teknik Altyapi

### Teknoloji Yigini

| Katman | Araclar |
|--------|---------|
| Veri formatlari | OBJ mesh + JSON segmentasyon etiketi |
| 3B isleme | Open3D, Trimesh, NumPy |
| Derin ogrenme | PyTorch, PointNet |
| Gorsellestirme | Plotly |
| Web arayuzu | Streamlit |
| Programlama | Python 3.10+ |

### PointNet Segmentasyon Mimarisi

```
Giris (B, 3, N)        N = 24.000 nokta
      |
      v
  Input T-Net (3x3 donusum)
      |
      v
  MLP (3 -> 64)
      |
      v
  Feature T-Net (64x64 donusum)
      |
      v                     Lokal Feature (B, 64, N)
  MLP (64 -> 128 -> 1024)          |
      |                            |
      v                            |
  Max Pooling -> Global Feature (B, 1024)
      |                            |
      v                            v
  Concat [Lokal (64) + Global (1024)] = (B, 1088, N)
      |
      v
  MLP (1088 -> 512 -> 256 -> 128 -> 33)
      |
      v
  Cikis (B, 33, N) - her nokta icin 33 sinif tahmini
                      (0=diseti, 11-48=disler)
```

### Registration Pipeline

```
Kaynak Mesh (T0)          Hedef Mesh (T1)
      |                        |
      v                        v
  Nokta Bulutu             Nokta Bulutu
  Olusturma                Olusturma
      |                        |
      v                        v
  FPFH Feature             FPFH Feature
  Cikarimi                 Cikarimi
      |                        |
      └──────────┬─────────────┘
                 |
                 v
         RANSAC Kaba Hizalama
         (Global Registration)
                 |
                 v
         ICP Ince Hizalama
         (Point-to-Plane)
                 |
                 v
         Donusum Matrisi (4x4)
         Fitness + RMSE
```

## Sonraki Adimlar

### Kisa Vade (Prototip Iyilestirme)
1. **PointNet modelini egitin** - `python train.py` ile gercek veri uzerinde segmentasyon egitimi
2. **Model performansini degerlendirin** - mDSC metrigini takip edin (hedef: 0.90+)
3. **Registration pipeline'i gercek OBJ ile test edin** - Ayni hastanin farkli taramalari ile

### Orta Vade (Klinik Entegrasyon)
4. **Kendi verilerinizi ekleyin** - TRIOS 6 taramalari ile longitudinal veri toplayin
5. **Temporal degisim analizi** - T0, T1, T2 taramalari ile gercek degisim olcumu
6. **Hastalik tespiti** - Curuk, asinma, diseti cekilmesi siniflandirmasi

### Uzun Vade (Urun)
7. **Tahmin modeli** - LSTM/regresyon ile hastalik ilerleme tahmini
8. **Klinik validasyon** - Uzman dis hekimleri ile kor degerlendirme
9. **Makale yazimi** - Akademik yayin hazirligi

## Akademik Referanslar

- **Teeth3DS+**: MICCAI challenge benchmark veri seti (1800+ intraoral 3B tarama)
- **TS-MDL**: Iki asamali mesh deep learning ile dis segmentasyonu (DSC: 0.964)
- **Ahmed et al. (2025)**: 3B intraoral taramalardan AI ile curuk tespiti
- **Michou et al. (2021)**: TRIOS IOS sistemi ile otomatik curuk skorlama validasyonu
- **Kuralt & Fidler (2021)**: Periodontitis hastalarinda seri 3B model superimpozisyonu
- **Dritsas et al. (2022)**: 3B dijital dental modellerle diseti cekilmesi olcumu

## Lisans

Bu proje akademik arastirma amaclidir. Teeth3DS+ veri seti MICCAI challenge kapsaminda sunulmustur.
