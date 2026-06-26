import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os

# ── Page config ──
st.set_page_config(
    page_title="PaveDetect — Deteksi Kerusakan Jalan",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #F7F8FA;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

/* Header bar */
.header-bar {
    background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.header-bar h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    color: white;
}
.header-bar p {
    font-size: 0.95rem;
    opacity: 0.85;
    margin: 0;
    color: white;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    flex: 1;
    text-align: center;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1E40AF;
    line-height: 1;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #6B7280;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Result image container */
.result-box {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

/* Detection table */
.det-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
.det-table th {
    background: #F3F4F6;
    padding: 0.6rem 1rem;
    text-align: left;
    font-weight: 600;
    color: #374151;
    border-bottom: 1px solid #E5E7EB;
}
.det-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #F3F4F6;
    color: #374151;
}
.det-table tr:last-child td { border-bottom: none; }

/* Class badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-D00 { background: #DBEAFE; color: #1D4ED8; }
.badge-D10 { background: #D1FAE5; color: #065F46; }
.badge-D20 { background: #FEF3C7; color: #92400E; }
.badge-D40 { background: #FEE2E2; color: #991B1B; }

/* Info box */
.info-box {
    background: #EFF6FF;
    border-left: 4px solid #3B82F6;
    padding: 0.875rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.875rem;
    color: #1E40AF;
    margin-bottom: 1rem;
}

/* Warning box */
.warn-box {
    background: #FFFBEB;
    border-left: 4px solid #F59E0B;
    padding: 0.875rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.875rem;
    color: #92400E;
    margin-bottom: 1rem;
}

/* Section label */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9CA3AF;
    margin-bottom: 0.5rem;
}

/* Sidebar model info */
.model-info {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1rem;
    font-size: 0.8rem;
}
.model-info .row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    border-bottom: 1px solid #F3F4F6;
    color: #374151;
}
.model-info .row:last-child { border-bottom: none; }
.model-info .key { color: #6B7280; }
.model-info .val { font-weight: 600; color: #111827; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──
CLASS_INFO = {
    "D00": {"name": "Longitudinal Crack", "name_id": "Retak Memanjang",   "color": "#1D4ED8", "badge": "badge-D00"},
    "D10": {"name": "Transverse Crack",   "name_id": "Retak Melintang",   "color": "#065F46", "badge": "badge-D10"},
    "D20": {"name": "Alligator Crack",    "name_id": "Retak Buaya",       "color": "#92400E", "badge": "badge-D20"},
    "D40": {"name": "Pothole",            "name_id": "Lubang Jalan",      "color": "#991B1B", "badge": "badge-D40"},
}
MODEL_PATH = "best.pt"

# ── Load model ──
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🛣️ PaveDetect")
    st.markdown("---")

    st.markdown('<div class="section-label">Parameter Deteksi</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence minimum", 0.05, 0.95, 0.25, 0.05,
        help="Objek dengan confidence di bawah nilai ini tidak ditampilkan.")
    iou_threshold  = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.5, 0.05,
        help="Mengontrol seberapa banyak bounding box yang tumpang tindih diperbolehkan.")

    st.markdown("---")
    st.markdown('<div class="section-label">Informasi Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-info">
        <div class="row"><span class="key">Model</span><span class="val">YOLOv9c</span></div>
        <div class="row"><span class="key">Training</span><span class="val">100 epoch</span></div>
        <div class="row"><span class="key">mAP@50</span><span class="val">0.708</span></div>
        <div class="row"><span class="key">Precision</span><span class="val">0.791</span></div>
        <div class="row"><span class="key">Recall</span><span class="val">0.746</span></div>
        <div class="row"><span class="key">Dataset</span><span class="val">PaveDistress</span></div>
        <div class="row"><span class="key">Kelas</span><span class="val">4 kelas</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Kelas Kerusakan</div>', unsafe_allow_html=True)
    for code, info in CLASS_INFO.items():
        st.markdown(
            f'<span class="badge {info["badge"]}">{code}</span> '
            f'<span style="font-size:0.82rem;color:#374151"> {info["name_id"]}</span><br>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#9CA3AF;text-align:center">'
        'Penulisan Ilmiah · Universitas Gunadarma · 2026</div>',
        unsafe_allow_html=True
    )

# ── Main area ──
st.markdown("""
<div class="header-bar">
    <h1>🛣️ PaveDetect</h1>
    <p>Sistem deteksi kerusakan jalan aspal berbasis YOLOv9c · Upload foto jalan untuk mengidentifikasi jenis kerusakan secara otomatis</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2 = st.tabs(["📷 Upload Foto", "🖼️ Contoh Deteksi"])

def run_detection(image_input, source_label=""):
    img_array = np.array(image_input.convert("RGB"))
    results   = model.predict(img_array, conf=conf_threshold,
                              iou=iou_threshold, verbose=False)
    result    = results[0]
    plotted   = result.plot()
    plotted   = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    boxes = result.boxes
    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id   = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf_val = float(box.conf.item())
            detections.append({"kode": cls_name, "conf": conf_val})

    return plotted, detections

def show_results(plotted, detections, original_size):
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.image(plotted, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        n_det   = len(detections)
        n_kelas = len(set(d["kode"] for d in detections))
        avg_conf = sum(d["conf"] for d in detections) / n_det if n_det > 0 else 0

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="value">{n_det}</div>
                <div class="label">Total Deteksi</div>
            </div>
            <div class="metric-card">
                <div class="value">{n_kelas}</div>
                <div class="label">Jenis Kerusakan</div>
            </div>
            <div class="metric-card">
                <div class="value">{avg_conf:.0%}</div>
                <div class="label">Rata-rata Conf.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if n_det == 0:
            st.markdown("""
            <div class="info-box">
                Tidak ada kerusakan terdeteksi pada threshold saat ini.
                Coba turunkan nilai confidence minimum di sidebar.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-label">Detail Deteksi</div>', unsafe_allow_html=True)
            rows_html = ""
            for d in sorted(detections, key=lambda x: x["conf"], reverse=True):
                info   = CLASS_INFO.get(d["kode"], {})
                badge  = info.get("badge", "")
                name   = info.get("name_id", d["kode"])
                conf_p = f"{d['conf']:.1%}"
                rows_html += f"""
                <tr>
                    <td><span class="badge {badge}">{d['kode']}</span></td>
                    <td>{name}</td>
                    <td><strong>{conf_p}</strong></td>
                </tr>"""

            st.markdown(f"""
            <table class="det-table">
                <thead><tr><th>Kode</th><th>Jenis Kerusakan</th><th>Confidence</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-box" style="margin-top:1rem">
            <strong>Catatan:</strong> Hasil deteksi bersifat otomatis dan dapat mengandung
            kesalahan. Gunakan sebagai referensi awal, bukan keputusan akhir.
        </div>
        """, unsafe_allow_html=True)

# ── Tab 1: Upload ──
with tab1:
    st.markdown('<div class="section-label">Upload foto jalan aspal</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Pilih file gambar (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        with st.spinner("Mendeteksi kerusakan..."):
            plotted, detections = run_detection(image)
        show_results(plotted, detections, image.size)

    else:
        st.markdown("""
        <div style="background:white;border:2px dashed #D1D5DB;border-radius:12px;
                    padding:3rem;text-align:center;color:#9CA3AF;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem">📸</div>
            <div style="font-weight:600;color:#374151;margin-bottom:0.25rem">Upload foto jalan</div>
            <div style="font-size:0.85rem">Format yang didukung: JPG, JPEG, PNG</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: Sample images ──
with tab2:
    st.markdown('<div class="section-label">Contoh deteksi dari dataset PaveDistress</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Foto di bawah merupakan sampel dari dataset pengujian (test set) yang digunakan
        untuk mengevaluasi performa model.
    </div>
    """, unsafe_allow_html=True)

    sample_dir = "samples"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if sample_files:
            selected = st.selectbox("Pilih foto sampel", sample_files,
                                    label_visibility="collapsed")
            image = Image.open(os.path.join(sample_dir, selected))
            with st.spinner("Mendeteksi..."):
                plotted, detections = run_detection(image)
            show_results(plotted, detections, image.size)
    else:
        st.markdown("""
        <div style="background:white;border:2px dashed #D1D5DB;border-radius:12px;
                    padding:2rem;text-align:center;color:#9CA3AF;font-size:0.875rem">
            Folder <code>samples/</code> belum tersedia.
        </div>
        """, unsafe_allow_html=True)
