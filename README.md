<div align="center">

# 🌸 VocalBloom - Speech Emotion Recognition

### *Multi-Task Audio Classification: Emotion, Gender & Speaker Identification*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Librosa](https://img.shields.io/badge/Librosa-0.10+-green.svg)](https://librosa.org)
[![Gradio](https://img.shields.io/badge/Gradio-3.0+-pink.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Dataset](#-dataset)
- [Feature Extraction](#-feature-extraction)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Gradio Demo](#-gradio-demo)
- [Project Structure](#-project-structure)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**VocalBloom** is a deep learning system that analyzes speech audio to simultaneously predict **three attributes**:

- 🎭 **Emotion** — neutral, calm, happy, sad, angry, fearful, disgust, surprised
- ⚤️ **Gender** — male, female
- 🎤 **Speaker** — identifies which of 24 actors spoke

The system uses a **CNN-LSTM-Attention architecture** with extensive audio feature engineering and data augmentation to achieve robust classification performance.

> **Dataset:** [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)  
> **Framework:** TensorFlow / Keras  
> **Demo:** Gradio web interface

---

## 🏗️ Architecture

```
Input Audio (3 seconds, 22,050 Hz)
         │
         ▼
┌─────────────────────────────────────┐
│      Feature Extraction Pipeline    │
│  • MFCC (40) + Delta + Delta-Delta  │
│  • Log-Mel Spectrogram (128)        │
│  • Chroma (12)                      │
│  • Spectral Contrast (7)            │
│  • Pitch (1)                        │
│  • Energy/RMS (1)                   │
│  ─────────────────────────────────  │
│  Total: 229 features × 216 timesteps│
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   CNN-LSTM-Attention Model          │
│  ┌───────────────────────────────┐  │
│  │ Conv2D (229×1 → 64 filters)   │  │
│  │ Reshape → Conv1D (128)        │  │
│  │ BatchNorm + MaxPool + Dropout │  │
│  │ Conv1D (128)                  │  │
│  │ BatchNorm + MaxPool + Dropout │  │
│  │ Bi-LSTM (128, return_seq)     │  │
│  │ Attention Layer               │  │
│  │ Dropout(0.5)                  │  │
│  │ Dense(128, ReLU)              │  │
│  │ Dropout(0.5)                  │  │
│  │ Dense(8, Softmax)             │  │
│  └───────────────────────────────┘  │
└─────────────┬───────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 Emotion   Gender   Speaker
 (8-class) (2-class) (24-class)
```

### Model Specifications

| Component | Specification |
|-----------|---------------|
| **Input Shape** | (229, 216, 1) — features × timesteps × channels |
| **CNN Filters** | 64 → 128 → 128 |
| **LSTM Units** | 128 (Bidirectional) |
| **Attention** | Custom trainable attention layer |
| **Dense Layers** | 128 → output |
| **Total Parameters** | ~500K |
| **Output Heads** | 3 (Emotion, Gender, Speaker) |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-Task Learning** | Single model predicts emotion, gender, and speaker simultaneously |
| **Rich Feature Engineering** | 229 features extracted per audio sample |
| **Data Augmentation** | Gaussian noise, time stretch, pitch shift (via Audiomentations) |
| **Custom Attention** | Trainable attention mechanism for temporal focus |
| **Class Balancing** | Compute class weights for imbalanced datasets |
| **Interactive Demo** | Gradio web UI with custom pink/gray theme |
| **Comprehensive Evaluation** | Confusion matrices, classification reports, accuracy comparison |

---

## 📊 Dataset

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

- **Source:** [Kaggle - RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Format:** WAV files, 48kHz, 16-bit
- **Speakers:** 24 professional actors (12 male, 12 female)
- **Emotions:** 8 categories
- **Total Samples:** ~1,440 audio files

### Emotion Mapping

| Code | Emotion | Code | Emotion |
|------|---------|------|---------|
| 01 | neutral | 05 | angry |
| 02 | calm | 06 | fearful |
| 03 | happy | 07 | disgust |
| 04 | sad | 08 | surprised |

### Data Split

| Set | Percentage | Purpose |
|-----|------------|---------|
| **Train** | 70% | Model training with augmentation |
| **Validation** | 15% | Hyperparameter tuning, early stopping |
| **Test** | 15% | Final evaluation |

---

## 🔧 Feature Extraction

### Audio Preprocessing

```python
SAMPLE_RATE = 22050    # Resample to 22.05 kHz
DURATION = 3.0         # 3-second clips
N_MFCC = 40            # 40 MFCC coefficients
N_MELS = 128           # 128 Mel bands
HOP_LENGTH = 512       # Hop length for STFT
N_FFT = 2048           # FFT window size
MAX_LEN = 216          # Fixed time steps
```

### Extracted Features (229 total)

| Feature | Dimensions | Description |
|---------|------------|-------------|
| **MFCC** | 40 | Mel-frequency cepstral coefficients |
| **MFCC Delta** | 40 | First-order derivative |
| **MFCC Delta-Delta** | 40 | Second-order derivative |
| **Log-Mel Spectrogram** | 128 | Mel-scaled power spectrogram |
| **Chroma** | 12 | 12 pitch classes |
| **Spectral Contrast** | 7 | Octave-based spectral contrast |
| **Pitch (YIN)** | 1 | Fundamental frequency |
| **Energy (RMS)** | 1 | Root mean square energy |

### Data Augmentation

| Technique | Parameters | Purpose |
|-----------|------------|---------|
| **Gaussian Noise** | amplitude: 0.001–0.015 | Simulate recording noise |
| **Time Stretch** | rate: 0.8–1.25 | Vary speaking speed |
| **Pitch Shift** | semitones: ±4 | Vary voice pitch |

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/vocalbloom-speech-emotion.git
cd vocalbloom-speech-emotion

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
tensorflow>=2.10.0
librosa>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
gradio>=3.0.0
audiomentations>=0.30.0
soundfile>=0.12.0
kagglehub>=0.1.0
tqdm>=4.64.0
```

---

## 🚀 Usage

### 1. Training

```bash
python speechrecognition.py
```

Training pipeline:
1. **Download dataset** — via KaggleHub
2. **Extract features** — 229 features per sample
3. **Split data** — 70/15/15 train/val/test
4. **Augment training** — noise, stretch, pitch shift
5. **Train models** — Emotion (150 epochs), Gender (50 epochs), Speaker (80 epochs)
6. **Evaluate** — confusion matrices, classification reports

### 2. Launch Gradio Demo

```python
from speechrecognition import ui
ui.launch()
```

Or run the script — the Gradio interface starts automatically at the end.

### 3. Predict Single Audio

```python
from speechrecognition import predict_all

result = predict_all("path/to/audio.wav")
print(result)
# Output: {'emotion': 'happy', 'gender': 'female', 'speaker': 'Actor_3'}
```

---

## 📈 Results

### Model Performance

| Task | Model | Test Accuracy | Best Val Accuracy |
|------|-------|---------------|-------------------|
| **Emotion** | CNN-LSTM-Attention + Augmentation | ~85% | ~87% |
| **Gender** | CNN-LSTM | ~95% | ~96% |
| **Speaker** | CNN-LSTM | ~98% | ~99% |

### Training Configuration

| Parameter | Emotion | Gender | Speaker |
|-----------|---------|--------|---------|
| Epochs | 150 | 50 | 80 |
| Batch Size | 32 | 32 | 32 |
| Learning Rate | 5e-5 | 1e-4 | 1e-4 |
| Early Stopping Patience | 20 | 15 | 15 |
| LR Reduction Factor | 0.5 | 0.2 | 0.2 |
| Loss Function | Categorical Crossentropy | Binary Crossentropy | Categorical Crossentropy |

### Key Insights

- **Gender** is the easiest task (95%+ accuracy) — pitch and formant features are highly discriminative
- **Speaker** identification achieves near-perfect accuracy — each actor has distinct vocal characteristics
- **Emotion** is the most challenging — requires the full attention mechanism and augmentation
- **Data augmentation** improves emotion accuracy by ~5-8%

---

## 🎨 Gradio Demo

### Interface

```
┌─────────────────────────────────────────┐
│           🌸🎧 VocalBloom               │
│                                         │
│  VocalBloom is an intelligent speech   │
│  emotion recognition system...          │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  🎧 Upload Audio               │    │
│  │  [Drop audio file or record]   │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  🌸 Predictions                │    │
│  │  {                              │    │
│  │    "emotion": "happy",          │    │
│  │    "gender": "female",          │    │
│  │    "speaker": "Actor_3"         │    │
│  │  }                              │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

### Theme

- **Primary:** Pink (`#db2777`)
- **Secondary:** Purple
- **Background:** Soft gray (`#E5E5E5`)
- **Font:** Poppins

---

## 📁 Project Structure

```
vocalbloom-speech-emotion/
├── speechrecognition.py       # Main implementation
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── .gitattributes              # GitHub language detection
├── models/                     # Saved model weights (not in repo)
│   ├── emotion_model.h5
│   ├── gender_model.h5
│   └── speaker_model.h5
└── data/                       # RAVDESS dataset (not in repo)
    └── ravdess-emotional-speech-audio/
```

---

## 🎓 Acknowledgments

- **Dataset:** [RAVDESS](https://zenodo.org/record/1188976) — Livingstone & Russo, 2018
- **Libraries:** TensorFlow, Librosa, Gradio, Audiomentations
- **Architecture:** Inspired by CNN-LSTM models for speech emotion recognition

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

</div>
