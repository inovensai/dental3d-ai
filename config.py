"""Proje konfigürasyon dosyası."""
from pathlib import Path

# Proje kök dizini
PROJECT_ROOT = Path(__file__).parent

# ─── Veri seti dizinleri ───
DATA_PARTS = [PROJECT_ROOT / f"data_part_{i}" for i in range(1, 8)]
LANDMARK_TRAIN_DIR = PROJECT_ROOT / "3DTeethLand_landmarks_train"
LANDMARK_TEST_DIR = PROJECT_ROOT / "3DTeethLand_landmarks_test"

# Çıktı dizinleri
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ─── FDI diş numaralama sistemi ───
FDI_TOOTH_NAMES = {
    0:  "Diseti (Gingiva)",
    # Üst sağ (1. kadran)
    11: "Ust sag santral kesici",   12: "Ust sag lateral kesici",
    13: "Ust sag kanin",            14: "Ust sag 1. premolar",
    15: "Ust sag 2. premolar",      16: "Ust sag 1. molar",
    17: "Ust sag 2. molar",         18: "Ust sag 3. molar",
    # Üst sol (2. kadran)
    21: "Ust sol santral kesici",   22: "Ust sol lateral kesici",
    23: "Ust sol kanin",            24: "Ust sol 1. premolar",
    25: "Ust sol 2. premolar",      26: "Ust sol 1. molar",
    27: "Ust sol 2. molar",         28: "Ust sol 3. molar",
    # Alt sol (3. kadran)
    31: "Alt sol santral kesici",   32: "Alt sol lateral kesici",
    33: "Alt sol kanin",            34: "Alt sol 1. premolar",
    35: "Alt sol 2. premolar",      36: "Alt sol 1. molar",
    37: "Alt sol 2. molar",         38: "Alt sol 3. molar",
    # Alt sağ (4. kadran)
    41: "Alt sag santral kesici",   42: "Alt sag lateral kesici",
    43: "Alt sag kanin",            44: "Alt sag 1. premolar",
    45: "Alt sag 2. premolar",      46: "Alt sag 1. molar",
    47: "Alt sag 2. molar",         48: "Alt sag 3. molar",
}

# Eşsiz FDI etiket -> ardışık indeks (model eğitimi için)
ALL_FDI_LABELS = sorted(FDI_TOOTH_NAMES.keys())  # [0, 11, 12, ..., 48]
FDI_TO_INDEX = {fdi: idx for idx, fdi in enumerate(ALL_FDI_LABELS)}
INDEX_TO_FDI = {idx: fdi for fdi, idx in FDI_TO_INDEX.items()}
NUM_CLASSES = len(ALL_FDI_LABELS)  # 33

# Landmark sınıfları (3DTeethLand)
LANDMARK_CLASSES = ["Mesial", "Distal", "Cusp", "FacialPoint", "InnerPoint", "OuterPoint"]

# ─── ICP parametreleri ───
ICP_MAX_ITERATION = 200
ICP_THRESHOLD = 0.5
RANSAC_MAX_ITERATION = 100000
RANSAC_CONFIDENCE = 0.999

# ─── Segmentasyon parametreleri ───
NUM_POINTS = 24000       # Mesh'den örneklenecek nokta sayısı
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# ─── Görselleştirme ───
COLORMAP = "jet"
DISTANCE_THRESHOLD_MM = 0.5  # mm cinsinden anlamlı değişim eşiği
