# Deepfake Audio and Video Detection using Deep Multi-Modal Approaches

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue.svg)](https://www.kaggle.com/)

> **Master's Thesis Project**  
> School of Computer and Information Sciences  
> University of Hyderabad  
> July 2025

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## 🎯 Overview

This project presents a comprehensive **multi-modal deepfake detection framework** that integrates three specialized deep learning architectures to detect sophisticated audio-visual manipulations. The system achieves state-of-the-art performance by analyzing audio, video, and audio-visual synchronization patterns simultaneously.

### Key Contributions
- **Multi-modal Architecture**: Unified detection system combining ViT, ViViT, and AV-HuBERT
- **High Accuracy**: 98.6% on audio, 97.8% on video, 96.0% on audio-visual detection
- **Ensemble Fusion**: Majority voting mechanism achieving 97.0% overall accuracy
- **Comprehensive Evaluation**: Tested on FakeAVCeleb, ASVspoof2019, DFDC, and DeepFake-TIMIT datasets

## ✨ Key Features

### 🎵 Audio Modal (ViT-based)
- MFCC feature extraction pipeline (7-step process)
- Vision Transformer architecture for audio spectrograms
- **98.6% accuracy** on FakeAVCeleb dataset
- Detects spectral anomalies and temporal inconsistencies

### 🎬 Video Modal (ViViT-based)
- Video Vision Transformer for spatio-temporal analysis
- MTCNN-based face detection and alignment
- **97.8% accuracy** on FakeAVCeleb dataset
- Identifies facial manipulation artifacts

### 🔊👁️ Audio-Visual Modal (AV-HuBERT)
- Lip-sync consistency analysis
- Cross-modal synchronization detection
- **96.0% accuracy** on FakeAVCeleb dataset
- Detects audio-visual misalignment

### 🤝 Ensemble Fusion
- Majority voting strategy
- **97.0% combined accuracy**
- Robust against sophisticated manipulations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Audio-Visual Content               │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼───┐   ┌───▼────┐  ┌───▼─────┐
   │ Audio  │   │ Video  │  │  A-V    │
   │ Modal  │   │ Modal  │  │ Modal   │
   │ (ViT)  │   │(ViViT) │  │(HuBERT) │
   └────┬───┘   └───┬────┘  └───┬─────┘
        │           │            │
        │    98.6%  │   97.8%    │  96.0%
        │           │            │
        └───────────┼────────────┘
                    │
            ┌───────▼────────┐
            │ Majority Voting│
            │     Fusion     │
            └───────┬────────┘
                    │
                  97.0%
                    │
            ┌───────▼────────┐
            │ Final Prediction│
            │  Real / Fake   │
            └────────────────┘
```

## 📊 Performance

### Individual Modal Results

| Modal | Dataset | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| **Audio (ViT)** | FakeAVCeleb | **98.6%** | 0.986 | 0.986 | 0.986 |
| **Video (ViViT)** | FakeAVCeleb | **97.8%** | 0.978 | 0.978 | 0.978 |
| **Audio-Visual** | FakeAVCeleb | **96.0%** | 0.960 | 0.960 | 0.960 |
| **Ensemble** | FakeAVCeleb | **97.0%** | 0.970 | 0.970 | 0.970 |

### Cross-Dataset Performance
- ASVspoof2019 (Audio): 98%+ accuracy
- DeepFake-TIMIT (Video): 92.4% accuracy
- DFDC (Video): High generalization capability

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended
- GPU with 16GB+ VRAM recommended

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Pre-trained Models
```bash
# Download model checkpoints
python scripts/download_models.py
```

## 💻 Usage

### Quick Start - Inference

```python
from deepfake_detector import MultiModalDetector

# Initialize detector
detector = MultiModalDetector(
    audio_model_path='models/audio_vit.pth',
    video_model_path='models/video_vivit.pth',
    av_model_path='models/av_hubert.pth'
)

# Detect deepfake
result = detector.predict('path/to/video.mp4')
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Audio Score: {result['audio_score']:.2%}")
print(f"Video Score: {result['video_score']:.2%}")
print(f"A-V Score: {result['av_score']:.2%}")
```

### Training from Scratch

```python
from train import train_multimodal_detector

# Configure training
config = {
    'audio': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 50
    },
    'video': {
        'batch_size': 16,
        'learning_rate': 5e-5,
        'epochs': 30
    },
    'audio_visual': {
        'batch_size': 8,
        'learning_rate': 1e-5,
        'epochs': 40
    }
}

# Train models
train_multimodal_detector(config, data_dir='data/', output_dir='output/')
```

### Batch Processing

```bash
# Process multiple videos
python scripts/batch_detect.py \
    --input_dir videos/ \
    --output_csv results.csv \
    --batch_size 16
```

## 📁 Datasets

### Primary Dataset
**FakeAVCeleb** (Main evaluation dataset)
- 21,000 videos (500 real + 20,500 fake)
- 5 deepfake generation methods
- Diverse ethnic backgrounds and balanced gender
- Download: [FakeAVCeleb Dataset](https://github.com/DASH-Lab/FakeAVCeleb)

### Augmentation Dataset
**VoxCeleb2**
- 10,000 additional real videos
- Enhanced dataset balance
- Download: [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)

### Training Datasets
1. **ASVspoof2019** (Audio Modal)
   - 79,380 training samples
   - 71,237 test samples
   - Download: [ASVspoof2019](https://www.asvspoof.org/)

2. **DFDC** (Video Modal)
   - 95,323 training videos
   - 4,846 test videos
   - Download: [DFDC](https://ai.facebook.com/datasets/dfdc/)

### Dataset Structure
```
data/
├── FakeAVCeleb/
│   ├── RealVideo-RealAudio/
│   ├── FakeVideo-RealAudio/
│   ├── RealVideo-FakeAudio/
│   └── FakeVideo-FakeAudio/
├── VoxCeleb2/
├── ASVspoof2019/
│   ├── LA/
│   └── PA/
└── DFDC/
    ├── train/
    └── test/
```

## 📈 Results

### Confusion Matrix Analysis
Each modal demonstrates strong classification performance with minimal false positives and false negatives:

- **Audio Modal**: Highest precision in detecting synthetic speech
- **Video Modal**: Excellent at identifying facial manipulations
- **Audio-Visual Modal**: Superior at detecting synchronization issues
- **Ensemble**: Balanced performance across all manipulation types

### ROC Curve Comparison
- Audio Modal: AUC = 0.986
- Video Modal: AUC = 0.978
- Audio-Visual Modal: AUC = 0.960

### Cross-Validation Results
5-fold cross-validation confirms robustness across different data splits with consistent performance metrics.

## 📂 Project Structure

```
deepfake-detection/
├── data/                          # Dataset directory
├── models/                        # Pre-trained models
│   ├── audio_vit.pth
│   ├── video_vivit.pth
│   └── av_hubert.pth
├── src/
│   ├── audio_modal/              # Audio detection module
│   │   ├── mfcc_extractor.py
│   │   ├── vit_model.py
│   │   └── train_audio.py
│   ├── video_modal/              # Video detection module
│   │   ├── face_detector.py
│   │   ├── vivit_model.py
│   │   └── train_video.py
│   ├── audiovisual_modal/        # Audio-visual module
│   │   ├── av_hubert_model.py
│   │   ├── lipsync_analyzer.py
│   │   └── train_av.py
│   ├── fusion/                   # Ensemble fusion
│   │   └── majority_voting.py
│   └── utils/                    # Utility functions
│       ├── preprocessing.py
│       ├── metrics.py
│       └── visualization.py
├── scripts/                      # Utility scripts
│   ├── download_models.py
│   ├── batch_detect.py
│   └── evaluate.py
├── notebooks/                    # Jupyter notebooks
│   ├── audio_analysis.ipynb
│   ├── video_analysis.ipynb
│   └── full_pipeline.ipynb
├── tests/                        # Unit tests
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── LICENSE                       # License file
```

## 🔧 Configuration

### Hardware Requirements
- **Minimum**: 8GB RAM, GPU with 8GB VRAM
- **Recommended**: 16GB+ RAM, GPU with 16GB+ VRAM (Tesla P100 or better)

### Software Environment
```
Python: 3.10.12
PyTorch: 2.0.1
CUDA: 11.8
```

Key Libraries:
- `torch`, `torchvision`, `torchaudio`
- `transformers` (Hugging Face)
- `librosa`, `opencv-python`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{govada2025deepfake,
  title={Deepfake Audio and Video Detection using Deep Multi-Modal Approaches},
  author={Govada, Dhana Lakshmi},
  year={2025},
  school={University of Hyderabad},
  type={Master's Thesis},
  department={School of Computer and Information Sciences}
}
```

## 🙏 Acknowledgments

I express my profound gratitude to:
- **Dr. Digambar Pawar** - Research Supervisor, for invaluable guidance and mentorship
- **University of Hyderabad** - School of Computer and Information Sciences
- **FakeAVCeleb Dataset** creators for providing comprehensive evaluation resources
- The **open-source community** for essential tools and libraries

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

**Dhana Lakshmi Govada**  
Master's in Artificial Intelligence  
University of Hyderabad  
Email: 23mcmi08@uohyd.ac.in

---

## 🔗 Additional Resources

- [Thesis Document](23MCMI08_Thesis.pdf)
- [Project Report](docs/project_report.pdf)
- [Presentation Slides](docs/presentation.pdf)
- [Demo Video](https://youtu.be/your-demo-video)

---

## 📊 System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel i5 / Ryzen 5 | Intel Xeon / Ryzen 7+ |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | GTX 1060 (6GB) | Tesla P100 (16GB) |
| **Storage** | 50 GB | 100 GB+ SSD |
| **OS** | Ubuntu 18.04+ | Ubuntu 20.04+ |

---

<div align="center">
  
### ⭐ If you find this project useful, please consider giving it a star!

**Made with ❤️ by Dhana Lakshmi Govada**

</div>
