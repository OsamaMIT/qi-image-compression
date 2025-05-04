import streamlit as st
import numpy as np
from PIL import Image
import tensorly as tl
from tensorly.decomposition import tensor_train
from io import BytesIO
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

# Use NumPy backend for TensorLy
tl.set_backend('numpy')

class QuantumImageCompressor:
    def __init__(self, patch_size=32, rank=4):
        self.patch_size = patch_size
        self.rank = rank

    def _split_into_patches(self, img):
        h, w, c = img.shape
        patches = []
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = img[i:i+self.patch_size, j:j+self.patch_size, :]
                if patch.shape[:2] != (self.patch_size, self.patch_size):
                    patch = np.pad(
                        patch,
                        ((0, self.patch_size - patch.shape[0]),
                         (0, self.patch_size - patch.shape[1]),
                         (0, 0)),
                        mode='constant', constant_values=0
                    )
                patches.append(patch)
        return patches

    def compress_image(self, img_array):
        patches = self._split_into_patches(img_array)
        compressed_patches = []
        for patch in patches:
            tt = tensor_train(patch, rank=[1, self.rank, self.rank, 1])
            compressed_patches.append(tt.factors)
        reconstructed = self._reconstruct_image(compressed_patches, img_array.shape)
        return reconstructed, compressed_patches

    def _reconstruct_image(self, compressed_patches, original_shape):
        h, w, c = original_shape
        recon = np.zeros(original_shape, dtype=float)
        idx = 0
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                factors = compressed_patches[idx]
                block = tl.tt_to_tensor(factors)
                ph = min(self.patch_size, h - i)
                pw = min(self.patch_size, w - j)
                recon[i:i+ph, j:j+pw] = block[:ph, :pw]
                idx += 1
        return recon

# Metric calculations
def calculate_tt_metrics(original: np.ndarray, compressed_factors: list):
    orig_bytes = original.nbytes
    tt_bytes = sum(f.nbytes for factors in compressed_factors for f in factors)
    tt_ratio = orig_bytes / tt_bytes if tt_bytes else float('inf')
    return orig_bytes, tt_bytes, tt_ratio

def calculate_mse(original: np.ndarray, reconstructed: np.ndarray):
    return np.mean((original - reconstructed) ** 2)

def calculate_compression_gain(original_size, compressed_size):
    return 100 * (1 - compressed_size / original_size)


def main():
    st.set_page_config(page_title="Quantum Image Compressor", page_icon="🖼️")
    st.title("🖼️ Quantum-Inspired Image Compression")

    with st.sidebar:
        st.title("Compression Settings")
        patch_size = st.slider("Patch Size (~Tile size for compression)", 16, 64, 32, 8)
        rank = st.slider("Compression Rank (~Building blocks per tile)", 2, 8, 4, 1)

        st.info("If you want sharp edges and fine textures, use small patches + higher rank (slower but better quality).")
        st.info("If you just need a quick, “good enough” thumbnail, try bigger patches + lower rank (faster and smaller, with more smoothing).")

        st.success("The quantum-inspired algorithm uses Tensor Train decomposition to compress images.\
                   Its strongest practical advantage is reducing the required memory usage (RAM) of an image")


    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        return

    # Compute original upload size
    uploaded_file.seek(0)
    orig_file_bytes = len(uploaded_file.read())
    uploaded_file.seek(0)

    original_image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(original_image, dtype=float) / 255.0
    h, w, _ = img_array.shape

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)

    if st.button("Compress Image"):
        reconstructed, comp_factors = QuantumImageCompressor(
            patch_size=patch_size,
            rank=rank
        ).compress_image(img_array)

        # Clip and convert to uint8 for display
        recon_clipped = np.clip(reconstructed, 0.0, 1.0)
        display_img = (recon_clipped * 255).astype(np.uint8)

        # Save JPEG into buffer
        buf = BytesIO()
        Image.fromarray(display_img).save(buf, format='JPEG')
        jpeg_bytes = buf.getvalue()
        download_size = len(jpeg_bytes)

        # Compute metrics
        mse_val = calculate_mse(img_array, recon_clipped)
        orig_bytes, tt_bytes, tt_ratio = calculate_tt_metrics(img_array, comp_factors)
        on_disk_ratio = orig_file_bytes / download_size if download_size else float('inf')
        compression_gain = calculate_compression_gain(orig_file_bytes, download_size)

        with col2:
            # Display using uint8 image
            st.image(display_img, caption="Compressed Image", use_container_width=True)
            
        st.subheader("Compression Metrics")

        met1, met2 = st.columns(2)
        with met1:
            st.metric("MSE (Loss)", f"{mse_val:.6f}")
            st.metric("On-disk Compression Ratio (in Storage)", f"{on_disk_ratio:.1f}×")

        with met2:
            st.metric("TT Storage Ratio (in-Memory/RAM)", f"{tt_ratio:.1f}×")
            st.metric("Compression Reduction", f"{compression_gain:.1f}%")

        st.download_button(
            label="Download Compressed Image",
            data=jpeg_bytes,
            file_name="compressed.jpg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()
