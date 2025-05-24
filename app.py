import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import io  # For image download

# Define the Autoencoder model class (same as your training code)
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 1024, (3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (3,3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, (3,3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, (3,3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, (3,3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, (3,3), stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x

# Function to load the model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder().to(device)
    
    # Load the saved weights (using the same name we saved with)
    model_path = "colorization_autoencoder.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.error(f"Model file not found at {model_path}")
        return None
    
    model.eval()
    return model

# Function to process the image
def process_image(image, model, img_size=128):
    device = next(model.parameters()).device
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array and resize
    gray_img = np.array(image)
    gray_img = cv2.resize(gray_img, (img_size, img_size))
    
    # Normalize and add batch dimension
    gray_img = gray_img.astype('float32') / 255.0
    gray_img = gray_img.reshape(1, 1, img_size, img_size)  # (batch, channel, height, width)
    
    # Convert to tensor and move to device
    gray_tensor = torch.from_numpy(gray_img).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(gray_tensor)
    
    # Convert output to numpy and denormalize
    color_img = output.cpu().numpy()[0]  # Remove batch dim
    color_img = np.transpose(color_img, (1, 2, 0))  # Change from C,H,W to H,W,C
    color_img = (color_img * 255).astype('uint8')
    
    # Convert original grayscale to 3 channels for display
    gray_display = cv2.cvtColor((gray_img[0][0] * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)
    
    return gray_display, color_img

# Main Streamlit app
def main():
    st.title("🎨 AI Image Colorization")
    st.write("Upload a grayscale image and watch the AI bring it to life with color!")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            if st.button("✨ Colorize Image"):
                with st.spinner("Working magic..."):
                    gray_img, color_img = process_image(image, model)
                    
                    with col1:
                        st.image(gray_img, caption="Grayscale Version", use_container_width=True)
                    with col2:
                        st.image(color_img, caption="Colorized Version", use_container_width=True)
                        
                    # Add download button for the colorized image
                    color_pil = Image.fromarray(color_img)
                    buf = io.BytesIO()
                    color_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download Colorized Image",
                        data=byte_im,
                        file_name="colorized.png",
                        mime="image/png"
                    )
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Add some info and examples
    st.markdown("---")
    st.subheader("How it works")
    st.write("""
    This app uses a deep learning autoencoder trained on celebrity faces to predict colors from grayscale images.
    The model was trained on 60,000 images from the CelebA dataset.
    """)
    
    st.subheader("Tips for best results")
    st.write("""
    - Use portrait images (works best with faces)
    - Higher resolution images give better results
    - Well-lit images work better than dark images
    """)

if __name__ == "__main__":
    main()