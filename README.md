# **🎨 AI Image Colorization Project**  
**Colorize Grayscale Images using Deep Learning Autoencoder**  

---

## **📌 Overview**  
This project trains an **Autoencoder** model to **convert grayscale images to color** using the **CelebA dataset**. The model was trained on **60,000 images** and evaluated using **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index Measure)** metrics.  

### **📊 Evaluation Results**  
| Metric | Score |
|--------|-------|
| **PSNR** | **25.74** |
| **SSIM** | **0.8178** |

These results indicate that the model produces **high-quality colorizations** with good structural similarity to the original images.

---

## **🚀 Features**  
✅ **Grayscale-to-Color Conversion** – Automatically adds realistic colors to B&W images  
✅ **Deep Autoencoder Architecture** – Uses convolutional and transpose layers for high-quality results  
✅ **Streamlit Web App** – Easy-to-use interface for testing the model  
✅ **Download Results** – Save colorized images in PNG format  

---![Screenshot 2025-05-24 210253](https://github.com/user-attachments/assets/5a7bbccf-923a-460b-b4c6-21c57a91a3e2)


  ![Screenshot 2025-05-24 135343](https://github.com/user-attachments/assets/fab8c653-9139-4a2d-a98f-8e1358004204)

## **🛠️ Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/ai-image-colorization.git
cd ai-image-colorization
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```
*(Sample `requirements.txt`)*  
```
streamlit==1.29.0
torch==2.1.0
torchvision==0.16.0
opencv-python==4.8.1
Pillow==10.0.0
numpy==1.24.0
```

### **3. Download the Pre-trained Model**  
Place the trained model (`colorization_autoencoder.pth`) in the project directory.  

*(If training from scratch, run:)*  
```bash
python train.py
```

### **4. Run the Streamlit App**  
```bash
streamlit run app.py
```
The app will open in your browser at **`http://localhost:8501`**.

---

## **🔧 Model Architecture**  
The Autoencoder consists of:  

### **📉 Encoder (Downsampling)**  
- **5 Convolutional Layers** (with BatchNorm & ReLU)  
- **MaxPooling** for dimensionality reduction  

### **📈 Decoder (Upsampling)**  
- **5 Transposed Convolutional Layers** (with ReLU)  
- **Sigmoid Activation** for final RGB output  

---

## **📂 Dataset**  
- **CelebA** (Celebrity Faces)  
- **60,000 training images** (128x128 resolution)  
- **Preprocessing**:  
  - Converted to grayscale (input)  
  - Normalized RGB (output)  

---

