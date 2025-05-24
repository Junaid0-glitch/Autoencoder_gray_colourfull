import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

# ========== Define the Autoencoder model ==========
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2),
        )
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x

# ========== Load the model ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
model.load_state_dict(torch.load("colorization_autoencoder.pth", map_location=device))
model.eval()

# ========== Streamlit Interface ==========
st.title("Grayscale Image Colorization using Autoencoder")
st.write("Upload a grayscale image (128x128), and the model will colorize it.")

uploaded_file = st.file_uploader("Choose a grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")  # Grayscale
    img = img.resize((128, 128))
    img_np = np.array(img).astype("float32") / 255.0
    input_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, 128, 128]

    with torch.no_grad():
        output = model(input_tensor)
    
    output_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Shape: [128, 128, 3]
    output_img = (output_img * 255).astype(np.uint8)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Input Grayscale Image", use_column_width=True)
    with col2:
        st.image(output_img, caption="Colorized Image", use_column_width=True)
