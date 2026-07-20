# 🎙️ Speech Emotion Recognition with Deep Learning

## Project Description

This project introduces a deep learning framework capable of understanding human speech by analyzing audio recordings and predicting multiple characteristics simultaneously. Instead of focusing only on emotion recognition, the model also identifies the speaker's gender and identity, making it a comprehensive speech analysis system.

The solution combines advanced audio preprocessing techniques with neural networks to learn meaningful patterns from speech signals and provides predictions through an easy-to-use interactive web application.

---

# Objectives

The project aims to:

- Recognize human emotions from speech recordings.
- Identify the speaker's gender.
- Recognize individual speakers.
- Improve classification performance using feature engineering and audio augmentation.
- Provide a simple interface for testing custom audio files.

---

# Technologies

- Python
- TensorFlow & Keras
- Librosa
- NumPy
- Scikit-learn
- Gradio

---

# Model Overview

The proposed system consists of several stages:

### Audio Processing

Each audio sample is normalized and transformed into numerical representations suitable for deep learning.

### Feature Extraction

Multiple acoustic descriptors are extracted, including:

- MFCC
- Delta MFCC
- Delta-Delta MFCC
- Mel Spectrogram
- Chroma Features
- Spectral Contrast
- Pitch
- RMS Energy

These features provide both spectral and temporal information about speech signals.

### Deep Learning Network

The extracted features are processed by a hybrid neural network composed of:

- Convolutional Neural Networks (CNN)
- Bidirectional LSTM layers
- Attention Mechanism
- Fully Connected Layers

This architecture enables the model to capture both local acoustic features and long-term temporal dependencies.

---

# Dataset

The experiments are conducted using the **RAVDESS Emotional Speech Dataset**.

Dataset characteristics include:

- 24 professional speakers
- Male and female voices
- Eight emotion categories
- High-quality WAV audio recordings

The dataset is divided into:

- Training Set
- Validation Set
- Testing Set

---

# Data Augmentation

To improve model generalization, several augmentation techniques are applied during training:

- Random Noise Injection
- Time Stretching
- Pitch Shifting

These techniques increase data diversity and reduce overfitting.

---

# Project Pipeline

```text
Audio Input
      │
      ▼
Audio Preprocessing
      │
      ▼
Feature Extraction
      │
      ▼
CNN Layers
      │
      ▼
Bi-LSTM
      │
      ▼
Attention Layer
      │
      ▼
Prediction
 ┌──────────┬──────────┬──────────┐
 │ Emotion  │ Gender   │ Speaker  │
 └──────────┴──────────┴──────────┘
```

---

# Performance Evaluation

The model is evaluated using several classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Separate evaluation is performed for:

- Emotion Recognition
- Gender Classification
- Speaker Identification

---

# User Interface

The project includes an interactive **Gradio** application that allows users to upload speech recordings and instantly receive predictions.

The interface displays:

- Predicted Emotion
- Predicted Gender
- Identified Speaker

making the system suitable for demonstrations and real-time testing.

---

# Applications

This project can be applied in various domains, including:

- Human–Computer Interaction
- Virtual Assistants
- Call Center Analytics
- Healthcare Monitoring
- Smart Voice Systems
- Educational AI Applications

---

# Future Improvements

Future enhancements may include:

- Support for multilingual speech
- Real-time microphone input
- Larger emotional speech datasets
- Transformer-based speech models
- Mobile and cloud deployment

---

# Project Structure

```text
Speech-Emotion-Recognition/
│
├── speechrecognition.py
├── requirements.txt
├── README.md
│
├── models/
├── datasets/
├── interface/
└── outputs/
```

---

# Conclusion

This project demonstrates an effective deep learning solution for speech analysis by combining advanced acoustic feature extraction with CNN, Bi-LSTM, and Attention mechanisms. The developed system successfully recognizes emotions, predicts speaker gender, and identifies individual speakers while providing an interactive interface for practical use.

---

# Author

**Shimaa Said**
