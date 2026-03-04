"""Streamlit Web Dashboard - Dis Hekimligi AI Prototipi.

Teeth3DS+ OBJ mesh + JSON label verileri uzerinde interaktif
3B gorsellestirme, segmentasyon analizi ve degisim tespiti.

Calistirma: streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import Teeth3DSDataset, DentalMesh
from src.visualization import (
    plot_pointcloud_with_distances,
    plot_risk_heatmap,
    plot_landmark_distribution,
)
from src.change_analysis import (
    analyze_changes,
    generate_risk_scores,
    compute_regional_statistics,
)
from config import PROJECT_ROOT, FDI_TOOTH_NAMES

# ───────────────── Sayfa Ayarlari ─────────────────
st.set_page_config(
    page_title="DentalAI - 3B Intraoral Tarama Analizi",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("DentalAI - 3B Intraoral Tarama Analiz Platformu")
st.markdown("*Teeth3DS+ veri seti | 1900 OBJ mesh + 1800 segmentasyon etiketi*")


# ───────────────── Veri Yukleme ─────────────────
@st.cache_resource
def load_dataset():
    return Teeth3DSDataset(PROJECT_ROOT, parts=[1, 2, 3, 4, 5, 6, 7])


try:
    dataset = load_dataset()
    data_loaded = True
except Exception as e:
    st.error(f"Veri seti yuklenemedi: {e}")
    data_loaded = False


# ───────────────── Yardimci Fonksiyonlar ─────────────────
def plot_mesh_3d(scan: DentalMesh, max_points: int = 15000) -> go.Figure:
    """Mesh'i segmentasyon etiketlerine gore renklendirilmis gosterir."""
    verts = scan.vertices
    labels = scan.labels

    # Performans icin alt ornekleme
    if len(verts) > max_points:
        idx = np.random.RandomState(42).choice(len(verts), max_points, replace=False)
        verts = verts[idx]
        labels = labels[idx] if labels is not None else None

    if labels is not None:
        unique_labels = sorted(set(labels))
        colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Bold
        fig = go.Figure()
        for i, fdi in enumerate(unique_labels):
            mask = labels == fdi
            name = FDI_TOOTH_NAMES.get(int(fdi), f"FDI {fdi}")
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter3d(
                x=verts[mask, 0], y=verts[mask, 1], z=verts[mask, 2],
                mode="markers",
                marker=dict(size=1.5, color=color, opacity=0.7),
                name=name,
                hovertemplate=f"<b>{name}</b><br>X:%{{x:.1f}}<br>Y:%{{y:.1f}}<br>Z:%{{z:.1f}}<extra></extra>",
            ))
    else:
        fig = go.Figure(data=[go.Scatter3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="#4ECDC4", opacity=0.7),
            name="Mesh",
        )])

    fig.update_layout(
        scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        legend=dict(font=dict(size=10)),
    )
    return fig


def plot_tooth_distribution(scan: DentalMesh) -> go.Figure:
    """Dis bazinda vertex dagilim grafigi."""
    stats = scan.get_tooth_stats()
    fdi_nums = sorted(stats.keys())
    names = [FDI_TOOTH_NAMES.get(f, f"FDI {f}") for f in fdi_nums]
    counts = [stats[f]["vertex_count"] for f in fdi_nums]

    colors = ["#E74C3C" if f == 0 else "#3498DB" for f in fdi_nums]

    fig = go.Figure(data=[go.Bar(x=names, y=counts, marker_color=colors, text=counts, textposition="auto")])
    fig.update_layout(
        title="Dis Bazinda Vertex Dagilimi",
        xaxis_title="", yaxis_title="Vertex Sayisi",
        height=400, xaxis_tickangle=-45,
    )
    return fig


# ───────────────── Sidebar ─────────────────
st.sidebar.header("Navigasyon")
page = st.sidebar.radio(
    "Sayfa Secin:",
    ["Genel Bakis", "3B Mesh Gorsellestirme", "Dis Analizi", "Degisim Simulasyonu", "Egitim Bilgisi"],
)

# ═══════════════════════════════════════════
# SAYFA 1: Genel Bakis
# ═══════════════════════════════════════════
if page == "Genel Bakis" and data_loaded:
    st.header("Veri Seti Genel Bakisi")
    stats = dataset.get_statistics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Tarama", stats["total_scans"])
    c2.metric("Etiketli Tarama", stats["labeled_scans"])
    c3.metric("Etiketsiz (Test)", stats["unlabeled_scans"])
    c4.metric("Benzersiz Hasta", stats["unique_patients"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Ust Cene", stats["upper_scans"])
    c2.metric("Alt Cene", stats["lower_scans"])
    c3.metric("Data Parts", stats["data_parts"])

    # Ornek bir mesh yukle ve istatistiklerini goster
    st.subheader("Ornek Tarama Istatistikleri")
    sample = dataset.get_labeled_scans()[0]
    sample.load_mesh()
    sample.load_labels()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vertex Sayisi", f"{sample.num_vertices:,}")
    c2.metric("Face Sayisi", f"{sample.num_faces:,}")
    c3.metric("Dis Sayisi", sample.num_teeth)
    bbox = sample.get_bounding_box()
    size = bbox[1] - bbox[0]
    c4.metric("Boyut (mm)", f"{size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f}")

    fig = plot_tooth_distribution(sample)
    st.plotly_chart(fig, use_container_width=True)
    sample.unload()

# ═══════════════════════════════════════════
# SAYFA 2: 3B Mesh Gorsellestirme
# ═══════════════════════════════════════════
elif page == "3B Mesh Gorsellestirme" and data_loaded:
    st.header("3B Mesh Gorsellestirme (Segmentasyon Etiketleri)")

    labeled = dataset.get_labeled_scans()
    patient_ids = sorted(set(s.patient_id for s in labeled))

    selected_patient = st.sidebar.selectbox("Hasta ID", patient_ids)
    jaw_type = st.sidebar.selectbox("Cene Tipi", ["lower", "upper"])
    max_points = st.sidebar.slider("Gosterilen Nokta Sayisi", 5000, 30000, 15000, 1000)

    scan = dataset.get_scan(selected_patient, jaw_type)

    if scan and scan.has_labels:
        with st.spinner("Mesh yukleniyor..."):
            scan.load_mesh()
            scan.load_labels()

        c1, c2, c3 = st.columns(3)
        c1.metric("Vertex", f"{scan.num_vertices:,}")
        c2.metric("Face", f"{scan.num_faces:,}")
        c3.metric("Dis Sayisi", scan.num_teeth)

        fig = plot_mesh_3d(scan, max_points=max_points)
        fig.update_layout(title=f"Hasta {selected_patient} - {jaw_type.upper()} Cene")
        st.plotly_chart(fig, use_container_width=True)

        # Dis istatistikleri
        with st.expander("Dis Detaylari"):
            stats = scan.get_tooth_stats()
            for fdi in sorted(stats.keys()):
                if fdi == 0:
                    continue
                s = stats[fdi]
                name = FDI_TOOTH_NAMES.get(fdi, f"FDI {fdi}")
                bbox = s["bbox_size_mm"]
                st.write(f"**{name}** (FDI {fdi}): {s['vertex_count']} vertex, "
                         f"boyut: {bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f} mm")

        fig2 = plot_tooth_distribution(scan)
        st.plotly_chart(fig2, use_container_width=True)

        scan.unload()
    else:
        st.warning("Bu hasta/cene kombinasyonu icin etiketli veri bulunamadi.")

# ═══════════════════════════════════════════
# SAYFA 3: Dis Analizi
# ═══════════════════════════════════════════
elif page == "Dis Analizi" and data_loaded:
    st.header("Tekil Dis Analizi")

    labeled = dataset.get_labeled_scans()
    patient_ids = sorted(set(s.patient_id for s in labeled))

    selected_patient = st.sidebar.selectbox("Hasta ID", patient_ids)
    jaw_type = st.sidebar.selectbox("Cene Tipi", ["lower", "upper"])

    scan = dataset.get_scan(selected_patient, jaw_type)

    if scan and scan.has_labels:
        with st.spinner("Mesh yukleniyor..."):
            scan.load_mesh()
            scan.load_labels()

        teeth = scan.unique_teeth
        tooth_names = {f: FDI_TOOTH_NAMES.get(f, f"FDI {f}") for f in teeth}
        selected_fdi = st.sidebar.selectbox(
            "Dis Secin",
            teeth,
            format_func=lambda f: f"{tooth_names[f]} (FDI {f})",
        )

        tooth_verts = scan.get_tooth_vertices(selected_fdi)

        st.subheader(f"{tooth_names[selected_fdi]} (FDI {selected_fdi})")
        c1, c2 = st.columns(2)
        c1.metric("Vertex Sayisi", f"{len(tooth_verts):,}")
        bbox = tooth_verts.max(axis=0) - tooth_verts.min(axis=0)
        c2.metric("Boyut (mm)", f"{bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f}")

        fig = go.Figure(data=[go.Scatter3d(
            x=tooth_verts[:, 0], y=tooth_verts[:, 1], z=tooth_verts[:, 2],
            mode="markers",
            marker=dict(size=2, color="#E74C3C", opacity=0.8),
        )])
        fig.update_layout(
            title=f"{tooth_names[selected_fdi]}",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            height=500, margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        scan.unload()

# ═══════════════════════════════════════════
# SAYFA 4: Degisim Simulasyonu
# ═══════════════════════════════════════════
elif page == "Degisim Simulasyonu" and data_loaded:
    st.header("Degisim Analizi Simulasyonu (Gercek Mesh Uzerinde)")
    st.info("Bu sayfa, gercek mesh verisi uzerinde sentetik patolojik degisimler olusturarak "
            "degisim analizi pipeline'ini gosterir.")

    labeled = dataset.get_labeled_scans()
    patient_ids = sorted(set(s.patient_id for s in labeled))[:50]
    selected_patient = st.sidebar.selectbox("Hasta ID", patient_ids)

    scan = dataset.get_scan(selected_patient, "lower") or dataset.get_scan(selected_patient, "upper")

    if scan and scan.has_labels:
        with st.spinner("Mesh yukleniyor..."):
            data = scan.sample_points(num_points=15000)

        points = data["points"]
        labels = data["labels"]
        n = len(points)

        noise_scale = st.slider("Degisim Buyuklugu (mm)", 0.1, 3.0, 0.8, 0.1)
        np.random.seed(42)

        simulated = points.copy()
        n_patho = max(1, int(n * 0.12))
        patho_idx = np.random.choice(n, n_patho, replace=False)
        simulated += np.random.normal(0, noise_scale * 0.15, points.shape).astype(np.float32)
        simulated[patho_idx] += np.random.normal(0, noise_scale, (n_patho, 3)).astype(np.float32)

        result = analyze_changes(points, simulated, threshold_mm=noise_scale * 0.4)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ort. Mesafe", f"{result.mean_distance:.3f} mm")
        c2.metric("Maks. Mesafe", f"{result.max_distance:.3f} mm")
        c3.metric("Hausdorff", f"{result.hausdorff_distance:.3f} mm")
        c4.metric("Anlamli Degisim", f"{result.significant_change_ratio:.1%}")

        fig = plot_pointcloud_with_distances(
            points, np.abs(result.distances),
            title="Yuzey Mesafe Haritasi",
            distance_range=(0, noise_scale * 2),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk skorlari (gercek dis etiketleri ile)
        st.subheader("Dis Bazinda Risk Degerlendirmesi")
        unique_fdi = sorted(set(labels))
        regional = compute_regional_statistics(points, result.distances, labels, num_classes=max(unique_fdi) + 1)
        risks = generate_risk_scores(regional)

        if risks:
            renamed = {}
            for fdi, risk in risks.items():
                name = FDI_TOOTH_NAMES.get(int(fdi), f"FDI {fdi}")
                renamed[name] = risk
            fig_risk = plot_risk_heatmap(renamed)
            st.plotly_chart(fig_risk, use_container_width=True)

        scan.unload()

# ═══════════════════════════════════════════
# SAYFA 5: Egitim Bilgisi
# ═══════════════════════════════════════════
elif page == "Egitim Bilgisi" and data_loaded:
    st.header("PointNet Segmentasyon Egitimi")

    st.markdown("""
    ### Model Egitimi Nasil Yapilir

    ```bash
    # Temel egitim (data_part 1-6 ile)
    python train.py

    # Ozel ayarlarla
    python train.py --epochs 100 --batch 8 --num-points 24000

    # Sadece ilk 2 part ile hizli test
    python train.py --parts 1 2 --epochs 10
    ```

    ### Model Mimarisi
    - **PointNet** segmentasyon agi
    - Giris: 24.000 nokta (x, y, z)
    - Cikis: Her nokta icin 33 sinif tahmini (32 dis + diseti)
    - Parametreler: ~3.5M
    """)

    st.subheader("Veri Seti Ozeti")
    stats = dataset.get_statistics()
    st.json(stats)

    st.markdown("""
    ### Pipeline Calistirma Sirasi
    1. `python train.py` - Model egitimi
    2. `streamlit run app.py` - Dashboard
    3. `python run_demo.py` - Tam demo
    4. `python tests/test_pipeline.py` - Testler
    """)

# ───────────────── Footer ─────────────────
st.sidebar.divider()
st.sidebar.markdown("**DentalAI** v0.2")
st.sidebar.markdown("Teeth3DS+ | 1900 Mesh")
