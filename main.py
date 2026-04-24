import os
import glob
import argparse
import warnings
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from matplotlib.colors import ListedColormap
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
DATA_PATH         = r"C:\Users\Irmak\Desktop\casting_data"
IMAGE_SIZE        = (128, 128)
TEST_SIZE         = 0.2
RANDOM_STATE      = 42

HOG_CELL_SIZE     = (8, 8)
HOG_ORIENTATIONS  = 12
HOG_BLOCK_NORM    = "L2-Hys"

LBP_SETTINGS      = [(1, 8), (2, 16), (3, 24)]   # çok ölçekli

PCA_COMPONENTS    = 0.95
LOF_K_VALUES      = [5, 10, 20, 30, 50, 75, 100]
LOF_CONTAMINATION = 0.10
LOF_METRIC        = "minkowski"

OUTPUT_DIR        = "./grafikler_casting"
TSNE_MAX_SAMPLES  = 2000
NORMAL_CLASS      = "ok_front"


@contextmanager
def timed(stage_name):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  [{stage_name}] {elapsed:.2f}s")


# =========================
# 1. Veri yükleme
# =========================
def preprocess(img_gray):
    """CLAHE → Gaussian Blur: lokal kontrast iyileştirme + gürültü bastırma."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img_gray)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def load_data(data_path, image_size, normal_class=NORMAL_CLASS):
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(glob.glob(os.path.join(data_path, "**", ext), recursive=True))
    if not image_paths:
        raise ValueError(f"Resim bulunamadı: {data_path}")

    images, labels, kept_paths = [], [], []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
        img = preprocess(img)
        parent_dir = os.path.basename(os.path.dirname(path)).lower()
        label = 0 if normal_class.lower() in parent_dir else 1
        images.append(img)
        labels.append(label)
        kept_paths.append(path)

    return (
        np.array(images, dtype=np.uint8),
        np.array(labels, dtype=np.int32),
        np.array(kept_paths),
    )


# =========================
# 2. Feature extraction
# =========================
def _lbp_histogram(image_uint8, radius, n_points):
    lbp    = local_binary_pattern(image_uint8, P=n_points, R=radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist   = hist.astype(np.float32)
    hist  /= hist.sum() + 1e-8
    return hist

def _extract_hog_and_lbp(img):
    img_float = img.astype(np.float32) / 255.0
    hog_feat = hog(
        img_float,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_CELL_SIZE,
        cells_per_block=(2, 2),
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    )
    lbp_feat = np.concatenate([
        _lbp_histogram(img, r, p) for r, p in LBP_SETTINGS
    ])
    return hog_feat, lbp_feat

def extract_features_parallel(images_uint8, n_jobs=1):
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(_extract_hog_and_lbp, images_uint8))
    hog_list, lbp_list = zip(*results)
    return np.array(hog_list, dtype=np.float32), np.array(lbp_list, dtype=np.float32)


# =========================
# 3. Model eğitimi
# =========================
def train_lof(X_train_pca, y_train):
    normal_mask = y_train == 0
    best_lof, best_score = None, -1

    for k in LOF_K_VALUES:
        k = min(k, max(2, int(normal_mask.sum()) - 1))
        if k < 2:
            continue
        lof_c = LocalOutlierFactor(
            n_neighbors=k,
            novelty=True,
            contamination=LOF_CONTAMINATION,
            metric=LOF_METRIC,
        )
        lof_c.fit(X_train_pca[normal_mask])
        score = roc_auc_score(y_train, -lof_c.decision_function(X_train_pca))
        if score > best_score:
            best_score = score
            best_lof   = lof_c

    print(f"    En iyi k = {best_lof.n_neighbors} | Train AUC = {best_score:.4f}")
    return best_lof


def _youden_threshold(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    best = np.argmax(tpr - fpr)
    return thresholds[best], fpr, tpr


def evaluate_lof(lof, X_test_pca, y_test):
    lof_scores = -lof.decision_function(X_test_pca)
    auc = roc_auc_score(y_test, lof_scores)
    thr, fpr, tpr = _youden_threshold(y_test, lof_scores)
    pred = (lof_scores >= thr).astype(int)
    cm   = confusion_matrix(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    return {
        "scores": lof_scores, "auc": auc, "threshold": thr,
        "fpr": fpr, "tpr": tpr, "pred": pred, "cm": cm,
        "precision": p, "recall": r, "f1": f1,
    }


# =========================
# 4. Grafikler
# =========================
def plot_isolated_results(
    feature_name, X_all_scaled, y_all, res, y_test, img_test, output_dir, skip_tsne=False
):
    feat_dir = os.path.join(output_dir, feature_name)
    os.makedirs(feat_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. t-SNE
    if not skip_tsne:
        print(f"  [{feature_name}] t-SNE hesaplanıyor...")
        if len(X_all_scaled) > TSNE_MAX_SAMPLES:
            rng = np.random.default_rng(RANDOM_STATE)
            sel_idx = rng.choice(len(X_all_scaled), size=TSNE_MAX_SAMPLES, replace=False)
            X_tsne, y_tsne = X_all_scaled[sel_idx], y_all[sel_idx]
        else:
            X_tsne, y_tsne = X_all_scaled, y_all
        n_comp = min(50, X_tsne.shape[1], max(2, X_tsne.shape[0] - 1))
        X_pre  = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(X_tsne)
        perp   = min(30, max(5, (len(X_pre) - 1) // 3))
        X_2d   = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_STATE,
                      init="pca", learning_rate="auto").fit_transform(X_pre)
        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_tsne,
                        cmap=ListedColormap(["#2ecc71", "#e74c3c"]), s=12, alpha=0.7)
        ax.legend(*sc.legend_elements(), labels=["Normal", "Anomaly"])
        ax.set_title(f"t-SNE Projection: {feature_name}")
        fig.tight_layout()
        fig.savefig(os.path.join(feat_dir, "1_tSNE_Dagilimi.png"), dpi=150)
        plt.close(fig)

    # 2. ROC & CM
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(res["fpr"], res["tpr"], color="#e74c3c", lw=2, label=f"AUC = {res['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title(f"ROC Curve ({feature_name})")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    ConfusionMatrixDisplay(res["cm"], display_labels=["Normal", "Anomaly"]).plot(
        ax=axes[1], colorbar=False, cmap="Blues"
    )
    axes[1].set_title(f"Confusion Matrix ({feature_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(feat_dir, "2_ROC_Karmasiklik_Matrisi.png"), dpi=150)
    plt.close(fig)

    # 3. Anomaly Score Distribution
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(res["scores"][y_test == 0], bins=35, kde=True, stat="density",
                 color="#2ecc71", alpha=0.5, label="Normal", ax=ax)
    sns.histplot(res["scores"][y_test == 1], bins=35, kde=True, stat="density",
                 color="#e74c3c", alpha=0.5, label="Anomaly", ax=ax)
    ax.axvline(res["threshold"], color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {res['threshold']:.3f}")
    ax.set_title(f"LOF Anomaly Score Distribution: {feature_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(feat_dir, "3_Anomali_Skoru_Yogunlugu.png"), dpi=150)
    plt.close(fig)

    # 4 & 5. En Normal ve En Anormal
    for title, fname, idx_fn, cmap in [
        ("Top 10 Most 'Normal' Predictions (Lowest LOF Score)",
         "4_En_Normal_Resimler.png", lambda s: np.argsort(s)[:10], "Greens"),
        ("Top 10 Most 'Anomalous' Predictions (Highest LOF Score)",
         "5_En_Anormal_Resimler.png", lambda s: np.argsort(s)[-10:][::-1], "Reds"),
    ]:
        indices = idx_fn(res["scores"])
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for ax in axes.flatten():
            ax.axis("off")
        for k, idx in enumerate(indices):
            ax = axes[k // 5, k % 5]
            ax.imshow(img_test[idx], cmap=cmap)
            true_lbl = "Normal" if y_test[idx] == 0 else "Anomaly"
            ax.set_title(f"Score: {res['scores'][idx]:.2f}\nTrue: {true_lbl}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{title} - {feature_name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(feat_dir, fname), dpi=150)
        plt.close(fig)

    print(f"  [{feature_name}] Tüm grafikler {feat_dir}/ dizinine kaydedildi.")


# =========================
# Main Flow
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_tsne", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_jobs = max(1, (os.cpu_count() or 2) - 1)

    print("Veriler yükleniyor...")
    with timed("load_data"):
        images, labels, _ = load_data(DATA_PATH, IMAGE_SIZE)

    print(f"Toplam Görüntü: {len(images)} | Normal ({NORMAL_CLASS}): {(labels==0).sum()} | Anomali: {(labels==1).sum()}")

    print("\nÖzellikler (Features) çıkarılıyor...")
    with timed("feature_extraction"):
        X_hog, X_lbp = extract_features_parallel(images, n_jobs=n_jobs)
    X_hog_lbp = np.concatenate((X_hog, X_lbp), axis=1)

    features_dict = {
        "HOG":     X_hog,
        "LBP":     X_lbp,
        "HOG+LBP": X_hog_lbp,
    }

    # Sabit veri ayrımı
    idx = np.arange(len(images))
    idx_train, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    y_train, y_test = labels[idx_train], labels[idx_test]
    img_test = images[idx_test]

    all_results = {}

    for name, X_raw in features_dict.items():
        print(f"\n================ {name} Boru Hattı (Pipeline) ================")
        X_train_raw = X_raw[idx_train]
        X_test_raw  = X_raw[idx_test]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled  = scaler.transform(X_test_raw)
        X_all_scaled   = scaler.transform(X_raw)

        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca  = pca.transform(X_test_scaled)
        print(f"  PCA Sonrası Bileşen Sayısı: {X_train_pca.shape[1]}")

        lof = train_lof(X_train_pca, y_train)
        res = evaluate_lof(lof, X_test_pca, y_test)
        all_results[name] = res

        print(f"  AUC: {res['auc']:.4f} | Precision: {res['precision']:.4f} | "
              f"Recall: {res['recall']:.4f} | F1: {res['f1']:.4f}")

        plot_isolated_results(
            feature_name=name.replace("+", "_"),
            X_all_scaled=X_all_scaled,
            y_all=labels,
            res=res,
            y_test=y_test,
            img_test=img_test,
            output_dir=OUTPUT_DIR,
            skip_tsne=args.skip_tsne,
        )

    # ── Karşılaştırma Grafikleri ─────────────────────────────────────────
    print("\nKarşılaştırma Grafikleri Çiziliyor...")
    colors = {"HOG": "#e74c3c", "LBP": "#3498db", "HOG+LBP": "#2ecc71"}

    # Ortak ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in all_results.items():
        ax.plot(res["fpr"], res["tpr"], color=colors[name], lw=2,
                label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance")
    ax.set_title("Ortak ROC Eğrisi Karşılaştırması: HOG, LBP, HOG+LBP (LOF Modeli)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Genel_ROC_Karsilastirmasi.png"), dpi=150)
    plt.close(fig)

    # Metrik bar grafiği
    metrics = ["AUC", "Precision", "Recall", "F1-Score"]
    x       = np.arange(len(metrics))
    width   = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, res) in enumerate(all_results.items()):
        vals = [res["auc"], res["precision"], res["recall"], res["f1"]]
        ax.bar(x + offsets[i], vals, width, label=name, color=colors[name])
        for j, v in enumerate(vals):
            ax.text(x[j] + offsets[i], v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Scores")
    ax.set_title("Performans Metrikleri: HOG vs LBP vs HOG+LBP")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Genel_Metrik_Karsilastirmasi.png"), dpi=150)
    plt.close(fig)

    print("Tüm işlemler başarıyla tamamlandı. Sonuç grafikleri 'grafikler/' içinde bulunabilir.")


if __name__ == "__main__":
    main()