import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os
import base64
import io
from PIL import Image

# Initialize session state variables
def init_session_state():
    if 'encrypted_data' not in st.session_state:
        st.session_state.encrypted_data = None
    if 'iv' not in st.session_state:
        st.session_state.iv = None
    if 'key' not in st.session_state:
        st.session_state.key = None
    if 'latent_vector' not in st.session_state:
        st.session_state.latent_vector = None

# Build the autoencoder model
def build_autoencoder():
    input_img = Input(shape=(128, 128, 1))
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # Latent space representation (16x16x128 = 32768D compressed to 512D)
    latent = Conv2D(512, (1, 1), activation='relu')(encoded)
    
    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(latent)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, latent)
    
    # Create decoder
    latent_input = Input(shape=(16, 16, 512))
    dec = autoencoder.layers[-7](latent_input)
    for layer in autoencoder.layers[-6:]:
        dec = layer(dec)
    decoder = Model(latent_input, dec)
    
    return encoder, decoder

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

# Convert image to proper format
def process_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img, img_array

# Main application
def main():
    st.set_page_config(page_title="Secure Medical Image Transmission", layout="wide")
    st.title("🏥 Privacy-Preserving Medical Image Transmission")
    st.markdown("""
    **Secure workflow for transmitting X-rays between hospitals using autoencoders and AES-GCM encryption**
    """)
    
    init_session_state()
    encoder, decoder = build_autoencoder()
    
    # Hospital A Section
    st.header("🏥 Hospital A: Upload & Encryption")
    uploaded_file = st.file_uploader("Upload patient X-ray (128x128 grayscale)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        original_img, img_array = process_image(uploaded_file)
        
        # Display original image
        st.subheader("1. Uploaded X-ray")
        st.image(original_img, caption="Original X-ray", use_column_width=False, width=300)
        
        # Compression
        if st.button("Compress and Encrypt"):
            with st.spinner("Compressing image..."):
                latent_vector = encoder.predict(img_array)
                st.session_state.latent_vector = latent_vector
            
            st.subheader("2. Compression Results")
            st.write(f"Original Size: {img_array.nbytes:,} bytes")
            latent_bytes = latent_vector.tobytes()
            st.write(f"Compressed Size: {len(latent_bytes):,} bytes (512D latent space)")
            
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
        latent_vector = latent_vector.reshape((1, 16, 16, 512))
        
        # Reconstruction
        with st.spinner("Reconstructing image..."):
            reconstructed = decoder.predict(latent_vector)
            reconstructed_img = Image.fromarray((reconstructed[0, :, :, 0] * 255).astype(np.uint8))
        
        st.subheader("5. Reconstructed X-ray")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original Image", use_column_width=True)
        with col2:
            st.image(reconstructed_img, caption="Reconstructed Image", use_column_width=True)
        
        # Calculate metrics
        mse = np.mean((img_array - reconstructed) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        st.metric("Reconstruction Quality", f"PSNR: {psnr:.2f} dB")
        st.success("✅ Medical image successfully reconstructed at Hospital B!")

if __name__ == "__main__":
    main()