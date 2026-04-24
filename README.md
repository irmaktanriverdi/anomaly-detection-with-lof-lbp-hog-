# 🏭 Casting Product Defect Detection — Unsupervised Anomaly Detection

Bu proje, endüstriyel döküm (casting) ürünlerindeki yüzey kusurlarını **etiket gerektirmeden** tespit etmeyi amaçlar. HOG, LBP ve HOG+LBP özellik çıkarımı ile **Local Outlier Factor (LOF)** modelini birleştiren, PCA destekli bir anomali tespiti pipeline'ı içerir.

---

## 📦 Dataset

**Real-Life Industrial Dataset of Casting Product**  
🔗 [Kaggle — ravirajsinh45](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

Veri seti, gerçek bir endüstriyel üretim hattından alınmış döküm ürünü görüntüleri içermektedir:

| Sınıf | Klasör Adı | Açıklama |
|---|---|---|
| Normal | `ok_front` | Kusursuz ürünler |
| Anomali | `def_front` | Çukur, çatlak vb. yüzey kusurları |

Görüntüler gri tonlamalı, yaklaşık 300×300 piksel çözünürlüğündedir. Pipeline, görüntüleri `128×128`'e yeniden boyutlandırarak işler.

---

## 🚀 Kurulum

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/kullanici-adi/casting-anomaly-detection.git
cd casting-anomaly-detection
```

### 2. Sanal Ortam Oluşturun (Önerilen)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 4. Veri Setini İndirin

[Kaggle sayfasından](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) veri setini indirip proje dizinine çıkarın. `main.py` içindeki `DATA_PATH` değişkenini kendi yolunuzla güncelleyin:

```python
DATA_PATH = r"C:\Kullanici\casting_data"   # <-- Bunu değiştirin
```

---

## ▶️ Kullanım

### Normal Çalıştırma

```bash
python main.py
```

### t-SNE'yi Atlayarak (Daha Hızlı)

t-SNE hesaplaması büyük veri setlerinde uzun sürebilir. Atlamak için:

```bash
python main.py --skip_tsne
```

---

## 🔬 Yöntem

### Pipeline Özeti

```
Ham Görüntü
    │
    ▼
Ön İşleme (CLAHE + Gaussian Blur)
    │
    ▼
Özellik Çıkarımı
    ├── HOG   (Histogram of Oriented Gradients)
    ├── LBP   (Local Binary Pattern — çok ölçekli)
    └── HOG+LBP (Birleşik)
    │
    ▼
RobustScaler Normalizasyonu
    │
    ▼
PCA Boyut İndirgeme (%95 varyans)
    │
    ▼
LOF Model Eğitimi (Yalnızca Normal Sınıfla)
    │
    ▼
Youden Eşiği ile Değerlendirme
```

### Özellik Çıkarımı

| Özellik | Parametreler |
|---|---|
| HOG | 12 yönelim, 8×8 hücre, L2-Hys normalizasyon |
| LBP | (R=1, P=8), (R=2, P=16), (R=3, P=24) — uniform |
| HOG+LBP | İki özellik vektörünün birleşimi |

### Model — Local Outlier Factor (LOF)

- Yalnızca **normal sınıfın** eğitim verisiyle fit edilir (`novelty=True`)
- En iyi `k` komşu sayısı `[5, 10, 20, 30, 50, 75, 100]` arasından **Train AUC**'u maksimize ederek seçilir
- Eşik değeri **Youden's J İstatistiği** ile belirlenir

---

## 📊 Çıktılar

Tüm grafikler `./grafikler_casting/` dizinine kaydedilir.

### Her Özellik İçin (HOG / LBP / HOG_LBP)

| Dosya | İçerik |
|---|---|
| `1_tSNE_Dagilimi.png` | 2D t-SNE projeksiyonu |
| `2_ROC_Karmasiklik_Matrisi.png` | ROC eğrisi + Konfüzyon Matrisi |
| `3_Anomali_Skoru_Yogunlugu.png` | LOF skor dağılımı ve eşik çizgisi |
| `4_En_Normal_Resimler.png` | En düşük anomali skorlu 10 görüntü |
| `5_En_Anormal_Resimler.png` | En yüksek anomali skorlu 10 görüntü |

### Genel Karşılaştırma

| Dosya | İçerik |
|---|---|
| `Genel_ROC_Karsilastirmasi.png` | HOG, LBP, HOG+LBP ROC eğrileri |
| `Genel_Metrik_Karsilastirmasi.png` | AUC, Precision, Recall, F1 bar grafiği |

---

## ⚙️ Yapılandırma

`main.py` dosyasının üst kısmındaki sabitleri düzenleyerek pipeline davranışını özelleştirebilirsiniz:

```python
DATA_PATH         = r"..."        # Veri seti konumu
IMAGE_SIZE        = (128, 128)    # Giriş görüntü boyutu
TEST_SIZE         = 0.2           # Test seti oranı
HOG_ORIENTATIONS  = 12            # HOG yönelim sayısı
HOG_CELL_SIZE     = (8, 8)        # HOG hücre boyutu
LBP_SETTINGS      = [(1,8),(2,16),(3,24)]  # LBP ölçekleri
PCA_COMPONENTS    = 0.95          # PCA varyans eşiği
LOF_K_VALUES      = [5,10,20,30,50,75,100] # Denenecek k değerleri
LOF_CONTAMINATION = 0.10          # LOF beklenen anomali oranı
NORMAL_CLASS      = "ok_front"    # Normal sınıf klasör adı
```

---

## 📁 Proje Yapısı

```
casting-anomaly-detection/
├── main.py              # Ana pipeline
├── requirements.txt     # Bağımlılıklar
├── README.md
└── grafikler_casting/   # Otomatik oluşturulur
    ├── HOG/
    ├── LBP/
    ├── HOG_LBP/
    ├── Genel_ROC_Karsilastirmasi.png
    └── Genel_Metrik_Karsilastirmasi.png
```

---

## 🛠️ Gereksinimler

- Python 3.8+
- numpy
- opencv-python
- matplotlib
- seaborn
- scikit-image
- scikit-learn

---

## 📄 Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.
