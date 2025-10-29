# Multi-Modal Knowledge Graph-Augmented Retrieval for Explainable Mycetoma Diagnosis

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2025.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)

Official implementation of **"Multi-Modal Knowledge Graph-Augmented Retrieval for Explainable Mycetoma Diagnosis"** accepted at MICAD 2025.

**Authors:** Safi Shamsi¹, Laraib Hasan¹, Azizur Rahman¹, Paras Nigam²

¹University of Birmingham, UK | ²IIIT Guwahati, India

## 🔬 Overview

This repository contains the complete implementation of our KG-RAG system for mycetoma diagnosis, achieving **94.8% accuracy** with clinically grounded explanations rated **4.7/5** by expert pathologists.

### Key Features

- ✅ **InceptionV3-based histopathology image classification**
- ✅ **Multi-modal Knowledge Graph** (5,247 entities, 15,832 relationships)
- ✅ **5-modality retrieval system** (Visual, Clinical, Lab, Geographic, Literature)
- ✅ **RAG-based explanation generation** with GPT-4
- ✅ **Complete evaluation framework** with ablation studies
- ✅ **Reproducible experiments** with all code and configurations

## 📊 Main Results

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| InceptionV3 (baseline) | 88.5% | 0.891 | 0.874 | 0.882 |
| + Visual KG retrieval | 89.8% | 0.903 | 0.889 | 0.896 |
| + Clinical notes | 91.2% | 0.918 | 0.905 | 0.911 |
| + Lab results | 93.5% | 0.941 | 0.928 | 0.934 |
| + Geographic data | 94.1% | 0.945 | 0.936 | 0.941 |
| **Full KG-RAG System** | **94.8%** | **0.952** | **0.944** | **0.948** |

**Improvement: +6.3%** over CNN-only baseline ✨

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/safishamsi/mycetoma-kg-rag.git
cd mycetoma-kg-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from src.models.inception_v3 import InceptionV3Classifier

# Load model
model = InceptionV3Classifier()
model.load_weights("checkpoints/inception_v3_best.pth")

# Predict
result = model.predict("data/test/case_001.jpg")
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📁 Repository Structure

```
mycetoma-kg-rag/
├── src/                    # Source code
│   ├── models/            # CNN models
│   ├── knowledge_graph/   # KG construction
│   ├── retrieval/         # Multi-modal retrieval
│   ├── rag/               # Explanation generation
│   └── pipeline/          # Full system
├── data/                  # Dataset and samples
├── config/                # Configuration files
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🔑 API Keys Required

### OpenAI (for RAG explanations)

```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Or add to config/config.yaml
openai:
  api_key: "sk-..."
```

**Note:** The system works without OpenAI (no explanations generated), but explanations are a key contribution of the paper.

## 📈 Experimental Results

All experimental results are available in `results/`:

### Main Results (Table 2)

| Method | Accuracy | Precision | Recall | F1 | AUC |
|--------|----------|-----------|--------|-----|-----|
| CNN only | 0.885 | 0.891 | 0.874 | 0.882 | 0.943 |
| Full KG-RAG | 0.948 | 0.952 | 0.944 | 0.948 | 0.987 |

### Ablation Study (Table 3)

See `results/tables/table3_ablation_study.csv`

### Expert Evaluation

- **Explanation quality:** 4.7/5 (our method) vs 2.6/5 (Grad-CAM)
- **Clinical relevance:** 4.8/5
- **Trustworthiness:** 4.6/5

See `results/expert_evaluation/` for details.

## 🎓 Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@inproceedings{shamsi2025mycetoma,
  title={Multi-Modal Knowledge Graph-Augmented Retrieval for Explainable Mycetoma Diagnosis},
  author={Shamsi, Safi and Hasan, Laraib and Rahman, Azizur and Nigam, Paras},
  booktitle={Medical Imaging and Computer-Aided Diagnosis (MICAD)},
  year={2025},
  organization={Springer}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

- **Safi Shamsi** - mxs1923@alumni.bham.ac.uk
- **Laraib Hasan** - Laraib.hasan45@gmail.com
- **Azizur Rahman** - er.azizurrahman@gmail.com
- **Paras Nigam** - paras.nigam@iiitg.ac.in
- **GitHub Issues**: [Report bugs or request features](https://github.com/safishamsi/mycetoma-kg-rag/issues)

## 🙏 Acknowledgments

- **Mycetoma Research Centre, Khartoum, Sudan** for providing the dataset and clinical expertise
- **Expert pathologists** who participated in explanation quality evaluation

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Note:** The MycetoMIC 2024 dataset has its own license (CC BY-NC-SA 4.0) and should be cited separately.

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**📄 Paper:** [arXiv:2025.xxxxx](https://arxiv.org/abs/2025.xxxxx)  
**🏛️ Conference:** MICAD 2025  
**💻 Code:** https://github.com/safishamsi/mycetoma-kg-rag  
**📊 Dataset:** CC BY-NC-SA 4.0
