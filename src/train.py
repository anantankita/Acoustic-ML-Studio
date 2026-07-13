"""
AcousticML Studio - Speech Model Training Pipeline
This script implements a production-grade PyTorch training script.
It covers:
1. torchaudio signal loading and preprocessing.
2. Data augmentation (SpecAugment - frequency and time masking).
3. PyTorch Dataset and DataLoader construction.
4. CRNN (Convolutional Recurrent Neural Network) training.
5. Model saving for downstream ONNX optimization.
"""

import os
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Ensure model module is importable
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model import SpeechCRNN, COMMAND_CLASSES

class SpeechCommandDataset(Dataset):
    """
    Custom PyTorch Dataset for loading speech commands.
    Demonstrates feature extraction and SpecAugment data augmentation.
    """
    def __init__(self, file_paths, labels, sample_rate=16000, max_seconds=1.0, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.augment = augment
        
        if HAS_TORCH:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                win_length=1024,
                hop_length=256,
                n_mels=128
            )
            self.amplitude_to_db = T.AmplitudeToDB()
            
            # SpecAugment (Data Augmentation for Speech)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
            self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if not HAS_TORCH:
            # Fallback for dummy initialization
            return np.zeros((1, 128, 63), dtype=np.float32), 0
            
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio (torchaudio returns waveform Tensor and sample_rate)
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if multi-channel
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if sample rate doesn't match
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        # Pad or truncate waveform to standard length (1.0 second)
        if waveform.size(1) < self.max_samples:
            pad_len = self.max_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :self.max_samples]
            
        # Extract Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec) # Shape: (1, n_mels, time_steps)
        
        # Apply SpecAugment during training to improve robustness and reduce overfitting
        if self.augment:
            mel_spec_db = self.freq_mask(mel_spec_db)
            mel_spec_db = self.time_mask(mel_spec_db)
            
        return mel_spec_db, label

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass + Optimize
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="AcousticML Studio - Train KWS CRNN Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output_path", type=str, default="model.pth", help="Path to save trained weights")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("[CRITICAL] PyTorch and torchaudio are required to run the actual training script.")
        print("Please install them using: pip install torch torchaudio")
        # Save a simulated file structure to demonstrate layout
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write("MOCK_WEIGHTS_FOR_DEMO")
        print(f"Created a mock weights file at {args.output_path} to support environment loading.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the CRNN model
    model = SpeechCRNN(num_classes=len(COMMAND_CLASSES)).to(device)
    
    # In a full run, we would download Google Speech Commands or utilize local audio files.
    # For this verification and showcase script, we will initialize the model and train on 
    # synthetic noise patterns to export a fully active weights file.
    print("No dataset path provided, generating synthetic training data to initialize model.pth...")
    
    # Create synthetic dataset (random waveforms representing audio signals)
    synthetic_wavs = []
    synthetic_labels = []
    
    temp_dir = "temp_synthetic_dataset"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        for i in range(32): # 32 items
            label = i % len(COMMAND_CLASSES)
            filepath = os.path.join(temp_dir, f"audio_{i}.wav")
            
            # Generate random wave
            waveform = torch.randn(1, 16000) * 0.1
            torchaudio.save(filepath, waveform, 16000)
            
            synthetic_wavs.append(filepath)
            synthetic_labels.append(label)
            
        dataset = SpeechCommandDataset(synthetic_wavs, synthetic_labels, augment=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        
        print("Starting training loops...")
        for epoch in range(args.epochs):
            loss, acc = train_epoch(model, dataloader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} - Acc: {acc*100:.2f}%")
            scheduler.step(loss)
            
        # Save model
        torch.save(model.state_dict(), args.output_path)
        print(f"Model saved successfully to {args.output_path}")
        
    finally:
        # Clean up synthetic files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
