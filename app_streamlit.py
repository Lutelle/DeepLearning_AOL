# app_streamlit.py
import streamlit as st
from pathlib import Path
import torch
import torchvision.utils as vutils
from torchvision.transforms import functional as TF
from generator import Generator
from PIL import Image
import io
import zipfile
import os
import numpy as np

st.set_page_config(layout="centered", page_title="DCGAN Anime Generator")

# Sidebar
st.title("DCGAN — Anime Face Generator")
st.sidebar.header("Settings")
ckpt_path = st.sidebar.text_input("Checkpoint path", "checkpoints/dcgan_epoch_050.pth")
device_choice = st.sidebar.selectbox("Device", ["cuda" if torch.cuda.is_available() else "cpu", "cpu"])
n_images = st.sidebar.slider("Number of images", 1, 64, 8)
seed = st.sidebar.number_input("Random seed (0 = random)", min_value=0, value=0)
nz = st.sidebar.number_input("Latent dim (nz)", min_value=16, max_value=512, value=100)
batch_size = st.sidebar.number_input("Batch size for generation", min_value=1, max_value=64, value=8)

device = torch.device(device_choice)

# Load model button
@st.cache_resource
def load_generator(ckpt, nz, device):
    netG = Generator(nz=nz, ngf=64, nc=3).to(device)
    if Path(ckpt).exists():
        ckpt_dict = torch.load(ckpt, map_location=device)
        try:
            netG.load_state_dict(ckpt_dict['netG_state_dict'])
        except Exception:
            # allow loading if checkpoint saved with model state only
            try:
                netG.load_state_dict(ckpt_dict)
            except Exception as e:
                st.error(f"Failed to load checkpoint: {e}")
                return None
        netG.eval()
        return netG
    else:
        st.warning("Checkpoint not found. Please place checkpoint in checkpoints/ and update path.")
        return None

netG = load_generator(ckpt_path, nz, device)

st.write("**Model checkpoint:**", ckpt_path)
if netG is None:
    st.stop()

# Generate images
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Generate"):
        if seed != 0:
            torch.manual_seed(seed)
        n = n_images
        out_imgs = []
        netG.eval()
        with torch.no_grad():
            for i in range((n + batch_size - 1) // batch_size):
                cur = min(batch_size, n - i*batch_size)
                noise = torch.randn(cur, nz, 1, 1, device=device)
                fake = netG(noise).cpu()  # range [-1,1]
                fake = (fake * 0.5) + 0.5  # to [0,1]
                out_imgs.append(fake)
        out = torch.cat(out_imgs, dim=0)
        grid = vutils.make_grid(out, nrow=min(8, n), padding=2)
        np_img = grid.permute(1,2,0).numpy()
        st.image(np_img, caption="Generated images", use_column_width=True)

        # Offer zip download
        buffered = io.BytesIO()
        with zipfile.ZipFile(buffered, "w") as z:
            for idx in range(n):
                img = (out[idx] * 255).permute(1,2,0).numpy().astype('uint8')
                pil = Image.fromarray(img)
                b = io.BytesIO()
                pil.save(b, format="PNG")
                z.writestr(f"img_{idx:03d}.png", b.getvalue())
        st.download_button("Download images (zip)", data=buffered.getvalue(), file_name="generated_images.zip", mime="application/zip")

with col2:
    st.markdown("### Instructions")
    st.markdown("""
    1. Put your checkpoint file (PyTorch `.pth`) into `checkpoints/` (or adjust path above).  
    2. Make sure the checkpoint contains `netG_state_dict`.  
    3. Adjust `nz` if you trained with a different latent vector size.  
    4. For GPU support, run Docker with `--gpus` or use `nvidia-container-runtime`.  
    """)
