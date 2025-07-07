import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os
import base64
import io
from PIL import Image
import time

# Initialize session state
def init_session_state():
    keys = ['encrypted_data', 'iv', 'key', 'latent_vector', 'original_img', 'device']
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None
            
    if st.session_state.device is None:
        st.session_state.device = torch.device('cpu')

# PyTorch Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Latent space
        self.latent = nn.Conv2d(128, 512, 1)
        
        # Decoder
        self.dec1 = nn.Conv2d(512, 128, 3, padding=1)
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec3 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec4 = nn.Conv2d(32, 1, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        
        # Latent space
        x = F.relu(self.latent(x))
        
        # Decoder
        x = F.relu(self.dec1(x))
        x = self.upsample(x)
        x = F.relu(self.dec2(x))
        x = self.upsample(x)
        x = F.relu(self.dec3(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.dec4(x))
        return x

# Initialize model
def init_model():
    model = ConvAutoencoder()
    model.eval()  # Set to inference mode
    return model

# Encrypt data using AES-GCM
def encrypt_data(data, key, iv):
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    
    encryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv),
        backend=default_backend()
    ).encryptor()
    
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return ciphertext, encryptor.tag

# Decrypt data using AES-GCM
def decrypt_data(ciphertext, key, iv, tag):
    decryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(iv, tag),
        backend=default_backend()
    ).decryptor()
    
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    
    return data

# Convert image to tensor
def process_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor

# Main application
def main():
    st.set_page_config(page_title="Secure Medical Image Transmission", layout="wide")
    st.title("🏥 Privacy-Preserving Medical Image Transmission")
    st.markdown("""
    **Secure workflow for transmitting X-rays between hospitals using autoencoders and AES-GCM encryption**
    """)
    
    init_session_state()
    model = init_model()
    
    # Hospital A Section
    st.header("🏥 Hospital A: Upload & Encryption")
    uploaded_file = st.file_uploader("Upload patient X-ray (128x128 grayscale)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        original_img, img_tensor = process_image(uploaded_file)
        st.session_state.original_img = original_img
        
        # Display original image
        st.subheader("1. Uploaded X-ray")
        st.image(original_img, caption="Original X-ray", use_column_width=False, width=300)
        
        # Compression
        if st.button("Compress and Encrypt"):
            with st.spinner("Compressing image..."):
                # Use PyTorch for compression
                latent_vector = model.encoder(img_tensor)
                st.session_state.latent_vector = latent_vector.detach().numpy()
            
            st.subheader("2. Compression Results")
            original_size = img_tensor.nelement() * img_tensor.element_size()
            latent_bytes = st.session_state.latent_vector.tobytes()
            latent_size = len(latent_bytes)
            st.write(f"Original Size: {original_size:,} bytes")
            st.write(f"Compressed Size: {latent_size:,} bytes (512D latent space)")
            
            # Generate crypto material
            key = os.urandom(32)  # AES-256
            iv = os.urandom(12)   # 96-bit IV for GCM
            
            # Encryption
            with st.spinner("Encrypting data..."):
                ciphertext, tag = encrypt_data(latent_bytes, key, iv)
                st.session_state.encrypted_data = ciphertext + tag
                st.session_state.key = key
                st.session_state.iv = iv
            
            st.subheader("3. Encryption Results")
            st.code(f"Key (hex): {key.hex()}", language="text")
            st.code(f"IV (hex): {iv.hex()}", language="text")
            
            encrypted_preview = base64.b64encode(st.session_state.encrypted_data).decode('utf-8')
            st.code(f"Encrypted Data (Base64): {encrypted_preview[:100]}...", language="text")
            
            st.success("✅ Encryption complete! Ready for transmission to cloud")
    
    # Cloud Transmission Section
    st.header("☁️ Cloud Transmission")
    if st.session_state.encrypted_data is not None:
        st.info("Encrypted data transmitted securely to cloud storage")
        st.code(f"Transmitted Data Size: {len(st.session_state.encrypted_data):,} bytes")
    else:
        st.warning("Waiting for encrypted data from Hospital A...")
    
    # Hospital B Section
    st.header("🏥 Hospital B: Decryption & Reconstruction")
    
    if st.button("Decrypt and Reconstruct") and st.session_state.encrypted_data is not None:
        # Separate tag from ciphertext (last 16 bytes)
        ciphertext = st.session_state.encrypted_data[:-16]
        tag = st.session_state.encrypted_data[-16:]
        
        # Decryption
        with st.spinner("Decrypting data..."):
            decrypted_data = decrypt_data(
                ciphertext, 
                st.session_state.key, 
                st.session_state.iv, 
                tag
            )
            
        st.subheader("4. Decryption Results")
        st.code(f"Used Key (hex): {st.session_state.key.hex()}", language="text")
        st.code(f"Used IV (hex): {st.session_state.iv.hex()}", language="text")
        st.success("✅ Decryption successful!")
        
        # Convert bytes back to numpy array
        latent_vector = np.frombuffer(decrypted_data, dtype=np.float32)
        latent_vector = latent_vector.reshape((1, 512, 16, 16))
        latent_tensor = torch.from_numpy(latent_vector)
        
        # Reconstruction
        with st.spinner("Reconstructing image..."):
            with torch.no_grad():
                reconstructed = model.decoder(latent_tensor)
                reconstructed_img = transforms.ToPILImage()(reconstructed.squeeze(0))
        
        st.subheader("5. Reconstructed X-ray")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.original_img, caption="Original Image", use_column_width=True)
        with col2:
            st.image(reconstructed_img, caption="Reconstructed Image", use_column_width=True)
        
        # Calculate metrics
        original_array = np.array(st.session_state.original_img)
        reconstructed_array = np.array(reconstructed_img)
        mse = np.mean((original_array - reconstructed_array) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        st.metric("Reconstruction Quality", f"PSNR: {psnr:.2f} dB")
        st.success("✅ Medical image successfully reconstructed at Hospital B!")

# Add encoder/decoder methods to the model class
def encoder(self, x):
    x = F.relu(self.enc1(x))
    x = self.pool(x)
    x = F.relu(self.enc2(x))
    x = self.pool(x)
    x = F.relu(self.enc3(x))
    x = self.pool(x)
    x = F.relu(self.latent(x))
    return x

def decoder(self, x):
    x = F.relu(self.dec1(x))
    x = self.upsample(x)
    x = F.relu(self.dec2(x))
    x = self.upsample(x)
    x = F.relu(self.dec3(x))
    x = self.upsample(x)
    x = torch.sigmoid(self.dec4(x))
    return x

# Add methods to the model class
ConvAutoencoder.encoder = encoder
ConvAutoencoder.decoder = decoder

if __name__ == "__main__":
    main()
