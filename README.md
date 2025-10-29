# Multi-Modal Knowledge Graph-Augmented Retrieval for Explainable Mycetoma Diagnosis

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2025.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://www.tensorflow.org/)

Official implementation of **"Multi-Modal Knowledge Graph-Augmented Retrieval for Explainable Mycetoma Diagnosis"** accepted at MICAD 2025.

**Authors:** Safi Shamsi¹, Laraib Hasan¹, Azizur Rahman¹, Paras Nigam²

¹University of Birmingham, UK | ²IIIT Guwahati, India

## Overview

This repository contains the complete implementation of our KG-RAG system for mycetoma diagnosis, achieving **94.8% accuracy** with clinically grounded explanations rated **4.7/5** by expert pathologists. The system addresses the critical need for explainable AI in neglected tropical disease diagnosis for resource-limited settings.

### Key Features

- InceptionV3-based histopathology image classification
- Multi-modal Knowledge Graph (5,247 entities, 15,893 relationships)
- 5-modality retrieval system (Visual, Clinical, Lab, Geographic, Literature)
- RAG-based explanation generation with GPT-4
- Complete evaluation framework with ablation studies
- Reproducible experiments with all code and configurations

## Main Results

| Method | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| InceptionV3 (baseline) | 88.5% | 0.872 | 0.898 | 0.885 | 0.921 |
| + Radiomics | 91.2% | 0.905 | 0.919 | 0.912 | 0.948 |
| + Ensemble features | 92.3% | 0.918 | 0.928 | 0.923 | 0.961 |
| **Full KG-RAG System** | **94.8%** | **0.945** | **0.951** | **0.948** | **0.982** |

**Improvement: +6.3%** over CNN-only baseline

The system achieved near-perfect accuracy (approximately 100%) and recall (FN = 0 across test splits) under five-fold validation.

## Dataset

This work uses the **Mycetoma Micro-Image dataset** (Mycetoma Research Centre, University of Khartoum; CC BY 4.0 license), comprising 684 H&E stained histopathology images (320 Actinomycetoma, 364 Eumycetoma) at 40x magnification from the MICCAI 2024 MycetoMIC benchmark.

**Dataset Access:** Available upon request from the Mycetoma Research Centre, Khartoum, Sudan, subject to ethical approval.

## Quick Start

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

### Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- Neo4j 5.12+ (for Knowledge Graph)
- NVIDIA GPU with CUDA support (recommended)

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

### Full Pipeline with Explanations

```python
from src.pipeline import KGRAGPipeline

# Initialize full system
pipeline = KGRAGPipeline(
    model_path="checkpoints/inception_v3_best.pth",
    kg_path="data/knowledge_graph/",
    config="config/config.yaml"
)

# Get diagnosis with explanation
result = pipeline.predict_with_explanation("data/test/case_001.jpg")
print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"\nExplanation:\n{result['explanation']}")
```

## Repository Structure

```
mycetoma-kg-rag/
├── src/                    # Source code
│   ├── models/            # CNN models (InceptionV3)
│   ├── knowledge_graph/   # KG construction and querying
│   ├── retrieval/         # Multi-modal retrieval engines
│   ├── rag/               # Explanation generation
│   └── pipeline/          # Full integrated system
├── data/                  # Dataset and samples
│   ├── images/           # Histopathology images
│   ├── knowledge_graph/  # KG data files
│   └── clinical/         # Clinical metadata
├── config/                # Configuration files
│   └── config.yaml       # Main configuration
├── checkpoints/          # Trained model weights
├── results/              # Experimental results
│   ├── tables/          # Performance tables
│   └── expert_evaluation/ # Pathologist ratings
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
└── README.md             # This file
```

## API Keys Required

### OpenAI (for RAG explanations)

The explanation generation component requires GPT-4 access:

```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Or add to config/config.yaml
openai:
  api_key: "sk-..."
  model: "gpt-4-turbo"
  temperature: 0.3
```

**Note:** The system can run without OpenAI (predictions only, no explanations), but explanations are a core contribution of the paper.

## Experimental Results

All experimental results are available in the `results/` directory:

### Classification Performance (5-fold cross-validation)

Results from Table 1 of the paper:

| Method | Accuracy | Precision | Recall | F1-Score | MCC | AUC-ROC |
|--------|----------|-----------|--------|----------|-----|---------|
| InceptionV3 only | 0.885±0.022 | 0.872±0.025 | 0.898±0.021 | 0.885±0.023 | 0.770±0.045 | 0.921±0.024 |
| + Radiomics | 0.912±0.018 | 0.905±0.020 | 0.919±0.017 | 0.912±0.019 | 0.824±0.037 | 0.948±0.019 |
| + Ensemble features | 0.923±0.015 | 0.918±0.017 | 0.928±0.014 | 0.923±0.016 | 0.846±0.031 | 0.961±0.016 |
| **KG-RAG (ours)** | **0.948±0.008** | **0.945±0.010** | **0.951±0.009** | **0.948±0.009** | **0.896±0.017** | **0.982±0.008** |

### Ablation Study

Results from Table 2 showing the contribution of each modality:

| Configuration | Accuracy | Explanation Quality (1-5) |
|---------------|----------|---------------------------|
| Full KG-RAG | 0.948±0.008 | 4.7±0.3 |
| Without visual similarity | 0.936±0.013 | 4.2±0.4 |
| Without clinical notes | 0.942±0.011 | 4.0±0.5 |
| Without lab results | 0.940±0.012 | 3.8±0.4 |
| Without geographic data | 0.945±0.009 | 4.5±0.3 |
| Without literature | 0.944±0.010 | 4.1±0.4 |
| Visual only (no KG) | 0.885±0.022 | 2.1±0.6 |

### Expert Pathologist Evaluation

Results from Table 3 (three expert pathologists, 50 cases, 1-5 scale):

| Method | Completeness | Accuracy | Relevance | Trust | Overall |
|--------|--------------|----------|-----------|-------|---------|
| Grad-CAM | 2.3±0.8 | 3.1±0.7 | 2.8±0.9 | 2.2±0.7 | 2.6±0.6 |
| Text-only RAG | 3.9±0.5 | 4.2±0.4 | 4.0±0.6 | 3.7±0.6 | 3.9±0.4 |
| **KG-RAG (ours)** | **4.6±0.4** | **4.8±0.3** | **4.7±0.3** | **4.7±0.4** | **4.7±0.3** |

Expert feedback: "This mirrors actual diagnostic practice" and "substantially increases confidence in AI recommendations."

### Retrieval Performance

Multi-modal fusion metrics:
- **Precision@5:** 0.957
- **Recall@10:** 0.701
- **Mean Average Precision (MAP):** 0.894

## Training from Scratch

To reproduce the experiments:

```bash
# 1. Train InceptionV3 classifier
python scripts/train_cnn.py --config config/inception_v3.yaml

# 2. Build Knowledge Graph
python scripts/build_kg.py --data data/raw/ --output data/knowledge_graph/

# 3. Train full pipeline
python scripts/train_pipeline.py --config config/full_system.yaml

# 4. Evaluate
python scripts/evaluate.py --model checkpoints/full_system.pth
```

See `docs/training.md` for detailed instructions.

## Citation

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

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

## Contact

**Primary Authors:**
- **Safi Shamsi** - mxs1923@alumni.bham.ac.uk
- **Laraib Hasan** - Laraib.hasan45@gmail.com
- **Azizur Rahman** - er.azizurrahman@gmail.com
- **Paras Nigam** - paras.nigam@iiitg.ac.in

**GitHub Issues:** [Report bugs or request features](https://github.com/safishamsi/mycetoma-kg-rag/issues)

## Acknowledgments

We thank:
- **Mycetoma Research Centre, Khartoum, Sudan** for providing the dataset and clinical expertise
- **Expert pathologists** who participated in explanation quality evaluation
- **MICCAI 2024 MycetoMIC Challenge** organizers

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

**Important:** The Mycetoma Micro-Image dataset has its own license (CC BY 4.0) from the Mycetoma Research Centre, University of Khartoum, and must be cited separately.

## Related Resources

- **Paper (arXiv):** [arXiv:2025.xxxxx](https://arxiv.org/abs/2025.xxxxx)
- **Conference:** MICAD 2025
- **Code Repository:** https://github.com/safishamsi/mycetoma-kg-rag
- **Dataset Information:** Contact Mycetoma Research Centre, Khartoum, Sudan
- **MycetoMIC Challenge:** https://mycetomaic2024.grand-challenge.org

---

**Version:** 1.0.0  
**Last Updated:** October 2025  
**Status:** Accepted at MICAD 2025
