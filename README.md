# 🛣️ PaveDetect — Deteksi Kerusakan Jalan Aspal

Aplikasi web interaktif untuk mendeteksi dan mengklasifikasikan kerusakan permukaan jalan aspal secara otomatis menggunakan model **YOLOv9c** berbasis deep learning.

Dibangun sebagai bagian dari **Penulisan Ilmiah**, Program Studi Sistem Informasi, Universitas Gunadarma, 2026.

---

## 🔗 Demo Aplikasi

**[asphalt-distress-detector.streamlit.app](https://asphalt-distress-detector.streamlit.app)**

---

## 📋 Jenis Kerusakan yang Dideteksi

| Kode | Nama | Deskripsi |
|------|------|-----------|
| D00 | Longitudinal Crack | Retak memanjang sejajar jalan |
| D10 | Transverse Crack | Retak melintang tegak lurus jalan |
| D20 | Alligator Crack | Retak berpola seperti sisik buaya |
| D40 | Pothole | Lubang pada permukaan jalan |

---

## 📊 Performa Model

| Metrik | Nilai |
|--------|-------|
| mAP@50 | 0.708 |
| Precision | 0.791 |
| Recall | 0.746 |
| F1-Score | 0.767 |

---

## 🗂️ Dataset

- **Nama**: PaveDistress — Asphalt Pavement Dataset
- **Sumber**: [Mendeley Data](https://data.mendeley.com/datasets/cbm6dkvggn/1)
- **Total citra**: 6.200 (train: 4.959 / val: 620 / test: 620)
- **DOI**: 10.17632/cbm6dkvggn/1

---

## ⚙️ Teknologi

- **Model**: YOLOv9c (Ultralytics 8.4.75)
- **Training**: 100 epoch, AdamW optimizer, augmentasi sedang
- **Platform pelatihan**: Google Colaboratory Pro (GPU NVIDIA A100)
- **Deployment**: Streamlit Community Cloud

---

## 📁 Struktur Repository

```
asphalt-distress-detector/
├── app.py              # Aplikasi Streamlit
├── best.pt             # Model YOLOv9c (Git LFS)
├── requirements.txt    # Dependensi Python
└── samples/            # Foto contoh untuk demo
```

---

## 📚 Referensi

- Wang, C.-Y., et al. (2024). *YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information*. arXiv:2402.13616.
- Hua, J., et al. (2024). *PaveDistress: A Comprehensive Dataset of Pavement Distresses Detection*. Mendeley Data.
- Redmon, J., et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016.
