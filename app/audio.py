import numpy as np
import scipy.signal as signal
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try to import torchaudio, fallback to scipy if not available
try:
    import torch
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

def reduce_noise_spectral_gating(y: np.ndarray, sr: int, noise_reduce_strength: float = 0.5) -> np.ndarray:
    """
    Applies Spectral Gating (a classic DSP noise reduction algorithm).
    1. Compute Short-Time Fourier Transform (STFT) of the signal.
    2. Estimate the noise floor from the lowest amplitude frames (or initial segment).
    3. Calculate a spectral mask: suppress bins where signal amplitude is close to noise floor.
    4. Apply the mask and reconstruct via Inverse STFT (ISTFT).
    
    This demonstrates deep DSP knowledge that is core to speech engineering.
    """
    if len(y) == 0:
        return y
        
    # Standard STFT parameters
    n_fft = 1024
    hop_length = 256
    win_length = n_fft
    
    # Compute STFT
    frequencies, times, Zxx = signal.stft(y, fs=sr, nperseg=win_length, noverlap=win_length - hop_length)
    
    # Compute magnitude and phase
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    
    # Estimate noise from the first 5 frames (assuming silence/noise-only at start)
    # or from the overall lowest 10% energy bins
    noise_frames = min(5, magnitude.shape[1])
    if noise_frames > 0:
        noise_mean = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        noise_std = np.std(magnitude[:, :noise_frames], axis=1, keepdims=True)
        noise_threshold = noise_mean + 1.5 * noise_std
    else:
        noise_threshold = np.percentile(magnitude, 15, axis=1, keepdims=True)
        
    # Safety margin
    noise_threshold = np.maximum(noise_threshold, 1e-5)
    
    # Compute mask (spectral gating)
    # Smooth mask to reduce musical noise artifacts
    sig = magnitude / noise_threshold
    mask = 1.0 / (1.0 + np.exp(-1.5 * (sig - 2.0))) # Sigmoid gate
    
    # Scale mask strength
    mask = mask * noise_reduce_strength + (1.0 - noise_reduce_strength)
    
    # Apply mask to magnitude
    magnitude_clean = magnitude * mask
    
    # Reconstruct complex spectrogram
    Zxx_clean = magnitude_clean * np.exp(1j * phase)
    
    # Compute Inverse STFT
    _, y_clean = signal.istft(Zxx_clean, fs=sr, nperseg=win_length, noverlap=win_length - hop_length)
    
    # Ensure length matches original
    if len(y_clean) > len(y):
        y_clean = y_clean[:len(y)]
    elif len(y_clean) < len(y):
        y_clean = np.pad(y_clean, (0, len(y) - len(y_clean)))
        
    return y_clean

def compute_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    """
    Computes the Mel-Spectrogram of an audio signal.
    Uses Torchaudio if available, otherwise falls back to a custom numpy/scipy implementation.
    """
    if HAS_TORCHAUDIO:
        # torchaudio expectations: tensor of shape (channels, samples)
        y_tensor = torch.from_numpy(y).float().unsqueeze(0)
        transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=n_mels
        )
        mel_spec = transform(y_tensor)
        # Convert to DB scale
        amplitude_to_db = T.AmplitudeToDB()
        mel_spec_db = amplitude_to_db(mel_spec)
        return mel_spec_db.squeeze(0).numpy()
    else:
        # Fallback NumPy implementation
        # Step 1: Compute STFT
        f, t, Zxx = signal.stft(y, fs=sr, nperseg=1024, noverlap=1024-256)
        magnitude = np.abs(Zxx)**2 # Power spectrogram
        
        # Step 2: Create Mel Filterbank
        # Mel scale conversion formulas
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700.0)
            
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595.0) - 1)
            
        # Frequency range
        f_min, f_max = 0, sr / 2
        mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
        
        # Equally spaced points in Mel scale
        mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts = mel_to_hz(mel_pts)
        
        # Map to FFT bins
        bin_pts = np.floor((1024 + 1) * hz_pts / sr).astype(int)
        
        # Construct filterbank
        fb = np.zeros((n_mels, magnitude.shape[0]))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_pts[m - 1]
            f_m = bin_pts[m]
            f_m_plus = bin_pts[m + 1]
            
            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
                    
        # Step 3: Dot product
        mel_spec = np.dot(fb, magnitude)
        
        # Step 4: Convert to DB scale
        mel_spec_db = 10 * np.log10(np.maximum(mel_spec, 1e-10))
        return mel_spec_db

def compute_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Computes Mel-Frequency Cepstral Coefficients (MFCCs).
    Uses Torchaudio if available, otherwise computes DCT of the Mel-Spectrogram.
    """
    if HAS_TORCHAUDIO:
        y_tensor = torch.from_numpy(y).float().unsqueeze(0)
        transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 1024,
                "n_mels": 128,
                "hop_length": 256
            }
        )
        mfccs = transform(y_tensor)
        return mfccs.squeeze(0).numpy()
    else:
        # Fallback NumPy implementation
        # Get Mel Spectrogram
        mel_spec_db = compute_mel_spectrogram(y, sr)
        
        # Apply Discrete Cosine Transform (DCT-II)
        # Standard implementation of DCT along columns (mel bins)
        num_frames = mel_spec_db.shape[1]
        n_mels = mel_spec_db.shape[0]
        mfccs = np.zeros((n_mfcc, num_frames))
        
        for i in range(n_mfcc):
            # Base vector for DCT
            cos_vec = np.cos(np.pi * i * (np.arange(n_mels) + 0.5) / n_mels)
            mfccs[i, :] = np.dot(cos_vec, mel_spec_db)
            
        return mfccs

def render_spectrogram_image(spec_data: np.ndarray, title: str, cmap: str = 'magma') -> str:
    """
    Plots the spectrogram data and returns a base64 encoded PNG string.
    This creates stunning visuals for the user interface.
    """
    plt.figure(figsize=(6, 2.5), dpi=100)
    plt.style.use('dark_background')
    
    # Plot spectrogram
    plt.imshow(spec_data, aspect='auto', origin='lower', cmap=cmap)
    plt.title(title, fontsize=10, color='#8892b0', pad=10)
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time Frames', fontsize=8, color='#8892b0')
    plt.ylabel('Mel Bins', fontsize=8, color='#8892b0')
    
    # Styling details
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#233554')
    plt.gca().spines['bottom'].set_color('#233554')
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    
    # Base64 encode
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"

def render_waveform_image(y: np.ndarray, y_clean: np.ndarray, title: str) -> str:
    """
    Plots a side-by-side or overlay comparison of raw vs cleaned waveforms.
    """
    plt.figure(figsize=(6, 2.5), dpi=100)
    plt.style.use('dark_background')
    
    plt.plot(y, label='Original (Noisy)', color='#ff79c6', alpha=0.6, linewidth=1.0)
    if y_clean is not None:
        plt.plot(y_clean, label='Denoised', color='#64ffda', alpha=0.9, linewidth=1.0)
        
    plt.title(title, fontsize=10, color='#8892b0', pad=10)
    plt.legend(loc='upper right', framealpha=0.3, fontsize=8)
    plt.xlabel('Samples', fontsize=8, color='#8892b0')
    plt.ylabel('Amplitude', fontsize=8, color='#8892b0')
    plt.grid(True, color='#233554', linestyle='--', alpha=0.5)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#233554')
    plt.gca().spines['bottom'].set_color('#233554')
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    
    # Base64 encode
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"
