"""PointNet tabanlı 3B diş segmentasyonu modülü.

Bu modül, 3B intraoral taramalardan dişleri otomatik olarak segmente
etmek için PointNet mimarisini kullanır. Her vertex/nokta için bir
sınıf etiketi (diş numarası veya dişeti) tahmin eder.

Teeth3DS+ veri seti ile doğrudan eğitilebilir.

Mimari: PointNet (Qi et al., 2017) - Segmentation Head
"""
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TNet(nn.Module):
    """Spatial Transformer Network (T-Net).

    Giriş noktalarına veya feature'lara rigid/affine dönüşüm uygular.
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # Global max pooling

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Birim matris ekleme (identity initialization)
        identity = torch.eye(self.k, device=x.device).flatten().unsqueeze(0)
        identity = identity.repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """PointNet Feature Encoder.

    Her nokta için lokal + global feature üretir.
    """

    def __init__(self, feature_transform: bool = True):
        super().__init__()
        self.feature_transform = feature_transform

        self.tnet_input = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if feature_transform:
            self.tnet_feature = TNet(k=64)

    def forward(self, x):
        num_points = x.shape[2]

        # Input transform
        trans_input = self.tnet_input(x)
        x = torch.bmm(trans_input, x)

        x = F.relu(self.bn1(self.conv1(x)))
        local_features = x  # 64-dim lokal özellikler

        # Feature transform
        trans_feat = None
        if self.feature_transform:
            trans_feat = self.tnet_feature(x)
            x = torch.bmm(trans_feat, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Global feature (max pooling)
        global_feature = torch.max(x, 2)[0]  # (B, 1024)

        # Segmentasyon için: global feature'ı her noktaya ekle
        global_expanded = global_feature.unsqueeze(2).repeat(1, 1, num_points)
        combined = torch.cat([local_features, global_expanded], dim=1)  # (B, 1088, N)

        return combined, global_feature, trans_input, trans_feat


class PointNetSegmentation(nn.Module):
    """PointNet Segmentation Network.

    Her nokta için sınıf tahmini yapar.
    num_classes: 33 (32 diş + 1 dişeti)
    """

    def __init__(self, num_classes: int = 33, feature_transform: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PointNetEncoder(feature_transform=feature_transform)

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Args:
            x: (B, 3, N) - nokta koordinatları

        Returns:
            pred: (B, num_classes, N) - sınıf olasılıkları
            trans_input: Input transform matrisi (regularization için)
            trans_feat: Feature transform matrisi (regularization için)
        """
        combined, global_feat, trans_input, trans_feat = self.encoder(x)

        x = F.relu(self.bn1(self.conv1(combined)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.conv4(x)

        return x, trans_input, trans_feat


def feature_transform_regularizer(trans):
    """Feature transform matrisinin ortogonal olmasını teşvik eden kayıp.

    L_reg = ||I - A*A^T||^2
    """
    if trans is None:
        return torch.tensor(0.0)

    d = trans.size()[1]
    I = torch.eye(d, device=trans.device).unsqueeze(0)
    loss = torch.mean(torch.norm(I - torch.bmm(trans, trans.transpose(2, 1)), dim=(1, 2)))
    return loss


class SegmentationTrainer:
    """Segmentasyon modeli eğitim sınıfı."""

    def __init__(
        self,
        model: PointNetSegmentation,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch gerekli.")

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, reg_weight: float = 0.001) -> dict:
        """Bir epoch eğitim yapar."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (points, labels) in enumerate(dataloader):
            points = points.to(self.device).float()  # (B, N, 3) -> (B, 3, N)
            points = points.transpose(2, 1)
            labels = labels.to(self.device).long()  # (B, N)

            self.optimizer.zero_grad()

            pred, trans_input, trans_feat = self.model(points)
            # pred: (B, C, N), labels: (B, N)
            loss = self.criterion(pred, labels)
            loss += reg_weight * feature_transform_regularizer(trans_feat)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            correct += (predicted == labels).sum().item()
            total += labels.numel()

        self.scheduler.step()

        return {
            "loss": total_loss / (batch_idx + 1),
            "accuracy": correct / total if total > 0 else 0,
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """Model değerlendirmesi yapar."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch_idx, (points, labels) in enumerate(dataloader):
            points = points.to(self.device).float().transpose(2, 1)
            labels = labels.to(self.device).long()

            pred, _, trans_feat = self.model(points)
            loss = self.criterion(pred, labels)

            total_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            correct += (predicted == labels).sum().item()
            total += labels.numel()

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        # Dice Score (DSC) hesaplama
        unique_classes = np.unique(labels)
        dice_scores = {}
        for cls in unique_classes:
            pred_mask = (preds == cls)
            label_mask = (labels == cls)
            intersection = (pred_mask & label_mask).sum()
            dice = 2.0 * intersection / (pred_mask.sum() + label_mask.sum() + 1e-8)
            dice_scores[int(cls)] = float(dice)

        return {
            "loss": total_loss / (batch_idx + 1),
            "accuracy": correct / total if total > 0 else 0,
            "mean_dice": float(np.mean(list(dice_scores.values()))),
            "per_class_dice": dice_scores,
        }

    def save_model(self, path: str):
        """Modeli kaydeder."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Model kaydedildi: {path}")

    def load_model(self, path: str):
        """Modeli yükler."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model yüklendi: {path}")


# ═══════════════════════════════════════════════════════
#  Teeth3DS PyTorch Dataset
# ═══════════════════════════════════════════════════════
class Teeth3DSTorchDataset(Dataset):
    """Teeth3DS+ verilerini PyTorch eğitimi için hazırlayan Dataset.

    Her öğe: (points, labels) - (N, 3), (N,)
    FDI etiketleri ardışık indekslere dönüştürülür.

    Kullanım:
        from src.data_loader import Teeth3DSDataset
        ds = Teeth3DSDataset(".", parts=[1,2])
        train_scans, val_scans = ds.split_train_val()
        train_dataset = Teeth3DSTorchDataset(train_scans, num_points=24000)
        loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    """

    def __init__(
        self,
        scans: list,
        num_points: int = 24000,
        fdi_to_index: dict | None = None,
        augment: bool = False,
    ):
        """
        Args:
            scans: DentalMesh listesi (has_labels=True olanlar)
            num_points: Her örnekten alınacak nokta sayısı
            fdi_to_index: FDI -> ardışık indeks eşlemesi
            augment: Veri çoğaltma uygula (rotasyon, jitter)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch gerekli")

        self.scans = [s for s in scans if s.has_labels]
        self.num_points = num_points
        self.augment = augment

        # FDI -> index eşlemesi (config'den veya otomatik)
        if fdi_to_index is not None:
            self.fdi_to_index = fdi_to_index
        else:
            all_labels = set()
            for scan in self.scans[:50]:
                scan.load_labels()
                all_labels.update(scan.labels.tolist())
                scan.unload()
            sorted_labels = sorted(all_labels)
            self.fdi_to_index = {fdi: idx for idx, fdi in enumerate(sorted_labels)}

        self.num_classes = len(self.fdi_to_index)

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan = self.scans[idx]
        data = scan.sample_points(self.num_points)

        points = data["points"]  # (N, 3)
        fdi_labels = data["labels"]  # (N,) FDI numaraları

        # FDI -> ardışık indeks
        index_labels = np.array(
            [self.fdi_to_index.get(int(l), 0) for l in fdi_labels],
            dtype=np.int64,
        )

        points = points.astype(np.float32)
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 1e-8:
            points = points / max_dist
        else:
            points = np.zeros_like(points)

        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

        if self.augment:
            points = self._augment(points)

        # Bellegi serbest birak
        scan.unload()

        return torch.from_numpy(points).float(), torch.from_numpy(index_labels).long()

    def _augment(self, points: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(points)):
            points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

        angle = np.random.uniform(0, 2 * np.pi)
        cos_a = np.float32(np.cos(angle))
        sin_a = np.float32(np.sin(angle))
        R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32)

        with np.errstate(all="ignore"):
            points = points @ R.T

        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

        jitter = np.random.normal(0, 0.002, points.shape).astype(np.float32)
        points = points + jitter

        return np.clip(points, -10.0, 10.0)
