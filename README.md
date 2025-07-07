# Privacy-Preserving Medical Image Transmission

This Streamlit application demonstrates a secure workflow for transmitting medical images between hospitals using convolutional autoencoders and AES-GCM encryption.

## How It Works
1. **Hospital A**: 
   - Upload a medical image (X-ray)
   - Compress image using convolutional autoencoder (to 512D latent space)
   - Encrypt compressed data with AES-GCM
   - Transmit encrypted data to cloud

2. **Cloud**:
   - Securely stores encrypted data
   - Transmits data to Hospital B

3. **Hospital B**:
   - Decrypts received data
   - Reconstructs original image using autoencoder
   - Displays recovered medical image

## Deployment on Streamlit Cloud
1. Create a GitHub repository with these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and connect your GitHub repository
4. Set:
   - Repository: Your repository
   - Branch: `main`
   - Main file path: `app.py`
5. Click "Deploy"

## Usage
1. At Hospital A: Upload an X-ray image
2. Click "Compress and Encrypt"
3. View encryption details and keys
4. Click "Decrypt and Reconstruct" at Hospital B section
5. Compare original and reconstructed images

Note: This is a demonstration system using simulated components.