"""3B görselleştirme modülü.

Plotly ve matplotlib ile interaktif 3B görselleştirmeler oluşturur.
Open3D pencereli görselleştirme de desteklenir.
"""
import numpy as np
from typing import Optional

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_landmarks_3d(
    landmarks: np.ndarray,
    classes: list[str],
    title: str = "Dental Landmarks",
    patient_id: str = "",
) -> "go.Figure":
    """Landmark noktalarını interaktif 3B olarak görselleştirir.

    Args:
        landmarks: Nx3 koordinat matrisi
        classes: Landmark sınıfları listesi
        title: Grafik başlığı
        patient_id: Hasta ID (görselleştirmede gösterilir)

    Returns:
        Plotly Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly gerekli.")

    class_colors = {
        "Mesial": "#FF6B6B",
        "Distal": "#4ECDC4",
        "Cusp": "#45B7D1",
        "FacialPoint": "#96CEB4",
        "InnerPoint": "#FFEAA7",
        "OuterPoint": "#DDA0DD",
    }

    fig = go.Figure()

    unique_classes = list(set(classes))
    for cls in unique_classes:
        mask = [c == cls for c in classes]
        cls_points = landmarks[mask]
        color = class_colors.get(cls, "#888888")

        fig.add_trace(go.Scatter3d(
            x=cls_points[:, 0],
            y=cls_points[:, 1],
            z=cls_points[:, 2],
            mode="markers",
            marker=dict(size=4, color=color, opacity=0.8),
            name=cls,
            hovertemplate=(
                f"<b>{cls}</b><br>"
                "X: %{x:.2f}<br>"
                "Y: %{y:.2f}<br>"
                "Z: %{z:.2f}<extra></extra>"
            ),
        ))

    display_title = f"{title} - {patient_id}" if patient_id else title
    fig.update_layout(
        title=display_title,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )

    return fig


def plot_pointcloud_with_distances(
    points: np.ndarray,
    distances: np.ndarray,
    title: str = "Yuzey Mesafe Haritasi",
    colorscale: str = "RdYlBu_r",
    distance_range: Optional[tuple] = None,
) -> "go.Figure":
    """Nokta bulutunu mesafe değerlerine göre renklendirir.

    Args:
        points: Nx3 koordinat matrisi
        distances: N uzunluğunda mesafe değerleri
        title: Grafik başlığı
        colorscale: Renk skalası
        distance_range: (min, max) mesafe aralığı

    Returns:
        Plotly Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly gerekli.")

    if distance_range is None:
        vmin, vmax = distances.min(), distances.max()
    else:
        vmin, vmax = distance_range

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=distances,
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title="Mesafe (mm)",
                thickness=20,
            ),
            opacity=0.8,
        ),
        hovertemplate=(
            "X: %{x:.2f}<br>"
            "Y: %{y:.2f}<br>"
            "Z: %{z:.2f}<br>"
            "Mesafe: %{marker.color:.3f} mm<extra></extra>"
        ),
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )

    return fig


def plot_segmentation_result(
    points: np.ndarray,
    labels: np.ndarray,
    title: str = "Dis Segmentasyonu",
) -> "go.Figure":
    """Segmentasyon sonucunu renkli olarak görselleştirir."""
    if not HAS_PLOTLY:
        raise ImportError("plotly gerekli.")

    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel

    fig = go.Figure()

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = colors[i % len(colors)]
        name = f"Dis {label}" if label > 0 else "Diseti"

        fig.add_trace(go.Scatter3d(
            x=points[mask, 0],
            y=points[mask, 1],
            z=points[mask, 2],
            mode="markers",
            marker=dict(size=2, color=color, opacity=0.7),
            name=name,
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )

    return fig


def plot_landmark_distribution(
    class_counts: dict,
    title: str = "Landmark Sinif Dagilimi",
) -> "go.Figure":
    """Landmark sınıf dağılımını bar chart olarak gösterir."""
    if not HAS_PLOTLY:
        raise ImportError("plotly gerekli.")

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    fig = go.Figure(data=[go.Bar(
        x=classes,
        y=counts,
        marker_color=colors[:len(classes)],
        text=counts,
        textposition="auto",
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Landmark Sinifi",
        yaxis_title="Sayi",
        height=400,
    )

    return fig


def plot_risk_heatmap(
    risk_scores: dict,
    title: str = "Dis Bazinda Risk Haritasi",
) -> "go.Figure":
    """Risk skorlarını heatmap olarak görselleştirir."""
    if not HAS_PLOTLY:
        raise ImportError("plotly gerekli.")

    teeth = sorted(risk_scores.keys())
    scores = [risk_scores[t]["score"] for t in teeth]
    levels = [risk_scores[t]["level"] for t in teeth]
    tooth_labels = [f"Dis {t}" for t in teeth]

    colors = []
    for level in levels:
        if level == "Dusuk":
            colors.append("#2ECC71")
        elif level == "Orta":
            colors.append("#F39C12")
        elif level == "Yuksek":
            colors.append("#E74C3C")
        else:
            colors.append("#8E44AD")

    fig = go.Figure(data=[go.Bar(
        x=tooth_labels,
        y=scores,
        marker_color=colors,
        text=[f"{s:.0f} ({l})" for s, l in zip(scores, levels)],
        textposition="auto",
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Dis",
        yaxis_title="Risk Skoru (0-100)",
        yaxis_range=[0, 100],
        height=400,
    )

    return fig


def plot_registration_comparison(
    source_points: np.ndarray,
    target_points: np.ndarray,
    transformed_points: np.ndarray,
    title: str = "Registration Karsilastirmasi",
) -> "go.Figure":
    """Registration öncesi ve sonrasını karşılaştırır."""
    if not HAS_PLOTLY:
        raise ImportError("plotly gerekli.")

    # Alt örnekleme (performans için)
    n = min(5000, len(source_points))
    idx_s = np.random.choice(len(source_points), n, replace=False)
    idx_t = np.random.choice(len(target_points), n, replace=False)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=["Hizalama Oncesi", "Hizalama Sonrasi"],
    )

    # Hizalama öncesi
    fig.add_trace(go.Scatter3d(
        x=source_points[idx_s, 0], y=source_points[idx_s, 1], z=source_points[idx_s, 2],
        mode="markers", marker=dict(size=1, color="red", opacity=0.5),
        name="Kaynak (T0)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=target_points[idx_t, 0], y=target_points[idx_t, 1], z=target_points[idx_t, 2],
        mode="markers", marker=dict(size=1, color="blue", opacity=0.5),
        name="Hedef (T1)",
    ), row=1, col=1)

    # Hizalama sonrası
    fig.add_trace(go.Scatter3d(
        x=transformed_points[idx_s, 0], y=transformed_points[idx_s, 1], z=transformed_points[idx_s, 2],
        mode="markers", marker=dict(size=1, color="red", opacity=0.5),
        name="Hizalanmis (T0)",
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter3d(
        x=target_points[idx_t, 0], y=target_points[idx_t, 1], z=target_points[idx_t, 2],
        mode="markers", marker=dict(size=1, color="blue", opacity=0.5),
        name="Hedef (T1)",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=title,
        height=600,
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig
