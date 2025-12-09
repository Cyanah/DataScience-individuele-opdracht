import streamlit as st
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import joblib
from PIL import Image
from io import BytesIO
from time import time
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, mean_squared_error, mean_absolute_error
from Library_individuele_opdracht import load_real_dataset, load_anomaly_dataset, make_dataloader, ConvAE, DenoiseAE, VAE, UNetAE, ResNetAE, AnoVAEGAN, get_transforms, ImageListDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
BATCH_SIZE = 32

MODEL_URL = {
    "DBSCAN_MODEL": "https://www.mediafire.com/file/dx6he90s3kr340k/dbscan_model.joblib/file",
    "DBSCAN_PCA": "https://www.mediafire.com/file/valbogw23gmsd8v/dbscan_pca.joblib/file",
    "DBSCAN_SCALER": "https://www.mediafire.com/file/3rn5l5kmr3xnpe0/dbscan_scaler.joblib/file",
    "GMM_MODEL": "https://www.mediafire.com/file/ic7txf03ssjpmry/gmm_model.joblib/file",
    "GMM_PCA": "https://www.mediafire.com/file/qw1bmo1svdehlj1/gmm_pca.joblib/file",
    "GMM_SCALER": "https://www.mediafire.com/file/pyy0agatddbo2yl/gmm_scaler.joblib/file",
    "KMEANS_MODEL": "https://www.mediafire.com/file/sr9hsykn118ao3h/kmeans_model.joblib/file",
    "KMEANS_PCA": "https://www.mediafire.com/file/prf4g3heoyf7cev/kmeans_pca.joblib/file",
    "KMEANS_SCALER": "https://www.mediafire.com/file/vrayvnq3p465l1v/kmeans_scaler.joblib/file",
    "RESNET": "https://www.mediafire.com/file/h3bqhhw1pysi01l/resnet_ae_best.pth/file",
    "UNET": "https://www.mediafire.com/file/luo1y64w8ejj9kk/unet_ae_best.pth/file",
    "VAE": "https://www.mediafire.com/file/322l3j2qzcr5647/vae_best.pth/file",
    "DENOISE": "https://www.mediafire.com/file/ru43fvb9ym2ikod/ae_denoise_best.pth/file",
    "VAEGAN": "https://www.mediafire.com/file/7ypj74bv4pl927l/ano_vaegan_best.pth/file",
    "CONV": "https://www.mediafire.com/file/ksenkwc4r40257p/conv_ae_best.pth/file"
}

def download_mediafire(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        return

    st.write(f"Downloading {output_path} ...")

    html = requests.get(url).text
    import re
    match = re.search(r'href="(https://download[^"]+)"', html)
    if not match:
        st.error("ERROR: Could not extract Mediafire direct link.")
        return

    real_url = match.group(1)
    file_bytes = requests.get(real_url).content

    with open(output_path, "wb") as f:
        f.write(file_bytes)

    st.success(f"Downloaded {output_path}")

def load_ae(model_class, path):
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def pixelate(img, scale=0.15):
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def evaluate_ae(model, dataloader, labels):
    scores = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            mse = torch.mean((batch - out)**2, dim=[1,2,3]).cpu().numpy()
            scores.extend(mse)

    scores = np.array(scores)
    preds = (scores > np.median(scores)).astype(int)

    cm = confusion_matrix(labels, preds)

    return {
        "ROC-AUC": roc_auc_score(labels, scores),
        "F1": f1_score(labels, preds),
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds),
        "Recall": recall_score(labels, preds),
        "MSE": np.mean(scores),
        "MAE": np.mean(np.abs(scores)),
        "ConfusionMatrix": cm
    }

st.title("🔎 Autoencoder Anomaly Detector — Hands vs Dice")

st.write("""
Upload an image OR load the full dataset, download the models, 
run anomaly detection, visualize blur, and compare algorithms!
""")

st.header("📥 Step 1 — Download Models")

if st.button("Download All Models"):
    for name, url in MODEL_URL.items():
        download_mediafire(url, f"models/{name.lower()}.pth" if "pth" in url else f"models/{name.lower()}.joblib")
    st.success("All models are downloaded and ready!")

st.header("📦 Step 2 — Load Dataset")

if st.button("Load hands.zip + dice.zip"):
    real = load_real_dataset("hands.zip")
    anomaly = load_anomaly_dataset("dice.zip")

    # Downsample anomaly for speed
    dice_sample = anomaly[:3600]

    transform = get_transforms(image_size=IMG_SIZE)
    dataset_resized = ImageListDataset(dice_sample, transform)

    dice_resized = [img.resize((IMG_SIZE, IMG_SIZE)) for img in dice_sample]

    # Combined for evaluation
    hands_test = real["test"]
    combined_imgs = hands_test + dice_resized
    combined_labels = np.array([0]*len(hands_test) + [1]*len(dice_resized))

    combined_loader = make_dataloader(combined_imgs, batch=BATCH_SIZE, shuffle=False)

    st.session_state["combined_loader"] = combined_loader
    st.session_state["combined_labels"] = combined_labels
    st.session_state["dice_resized"] = dice_resized

    st.success("Dataset loaded!")

st.header("🤖 Step 3 — Choose a Model")

ae_choice = st.selectbox(
    "Select Autoencoder",
    ["conv_ae", "ae_denoise", "vae", "unet_ae", "resnet_ae", "ano_vaegan"]
)

if st.button("Run Evaluation"):
    if "combined_loader" not in st.session_state:
        st.error("Please load the dataset first.")
    else:
        path_map = {
            "conv_ae": "models/conv.pth",
            "ae_denoise": "models/denoise.pth",
            "vae": "models/vae.pth",
            "unet_ae": "models/unet.pth",
            "resnet_ae": "models/resnet.pth",
            "ano_vaegan": "models/vaegan.pth"
        }

        ae_classes = {
            "conv_ae": ConvAE,
            "ae_denoise": DenoiseAE,
            "vae": VAE,
            "unet_ae": UNetAE,
            "resnet_ae": ResNetAE,
            "ano_vaegan": AnoVAEGAN
        }

        model = load_ae(ae_classes[ae_choice], path_map[ae_choice])

        results = evaluate_ae(model,
                              st.session_state["combined_loader"],
                              st.session_state["combined_labels"])

        st.subheader("📊 Performance Metrics")
        st.write(results)

        cm = results["ConfusionMatrix"]
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
        st.pyplot(fig)

st.header("🎨 Step 4 — Pixelated Blur for Dice")

if st.button("Show Pixelated Example"):
    if "dice_resized" not in st.session_state:
        st.error("Load dataset first.")
    else:
        img = np.array(random.choice(st.session_state["dice_resized"]))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pix = pixelate(img_bgr, scale=0.15)
        pix_rgb = cv2.cvtColor(pix, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Dice")
        with col2:
            st.image(pix_rgb, caption="Pixelated (Advanced Blur)")


st.header("📤 Step 5 — Upload Custom Image")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image")

    # Resize + pixelate
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    blurred = pixelate(arr_bgr, scale=0.15)
    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    st.image(blurred, caption="Pixelated Blur")
