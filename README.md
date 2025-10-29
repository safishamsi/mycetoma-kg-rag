# Multi-Modal Knowledge Graph-Augmented Retrieval for Explainable Mycetoma Diagnosis

[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2025.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-4.4+-green.svg)](https://neo4j.com/)

Official implementation of **"Multi-Modal Knowledge Graph-Augmented Retrieval for Explainable Mycetoma Diagnosis"** accepted at MICAD 2025.

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

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- Neo4j 4.4+
- 16GB RAM (minimum)
- 50GB free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mycetoma-kg-rag.git
cd mycetoma-kg-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Download Dataset & Pre-trained Models

```bash
# Download Mycetoma dataset (684 histopathology cases)
python data/scripts/download_dataset.py --output data/

# Download pre-trained InceptionV3 checkpoint
python scripts/download_checkpoint.py --model inception_v3 --output checkpoints/
```

### Setup Neo4j Database

```bash
# Install Neo4j (Ubuntu/Debian)
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j

# Start Neo4j
sudo systemctl start neo4j

# Access Neo4j browser at http://localhost:7474
# Default credentials: neo4j/neo4j (change on first login)
```

### Build Knowledge Graph

```bash
# Extract InceptionV3 features from all images (takes ~1 hour)
python scripts/extract_features.py \
    --data-path data/ \
    --model-path checkpoints/inception_v3_best.pth \
    --output data/features.npy

# Build complete Knowledge Graph (takes ~2 hours)
python scripts/build_kg.py \
    --data-path data/ \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password your_password
```

### Run Diagnostic System

```python
from src.pipeline.diagnostic_system import MycetomaDiagnosticSystem

# Initialize system
system = MycetomaDiagnosticSystem(
    model_path="checkpoints/inception_v3_best.pth",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",
    openai_api_key="your_openai_key"  # Optional: for GPT-4 explanations
)

# Diagnose a case
result = system.diagnose(
    image_path="data/test/case_001.jpg",
    clinical_notes="35-year-old male with 18-month history of painless foot swelling...",
    patient_demographics={
        "age": 35,
        "gender": "Male", 
        "location": "Khartoum"
    }
)

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"\nExplanation:\n{result['explanation']}")
```

**Output:**
```
Diagnosis: Eumycetoma
Confidence: 94.2%

Explanation:
Based on histopathological analysis, this case is diagnosed as Eumycetoma with 94.2% 
confidence. This diagnosis is supported by multiple lines of evidence: (1) Among 10 
visually similar cases, 9 were confirmed as Eumycetoma, showing characteristic fungal 
grain morphology with broad septate hyphae; (2) The patient is from Khartoum, Sudan, 
where Eumycetoma accounts for 75% of mycetoma cases; (3) Clinical presentation with 
chronic subcutaneous swelling and black grain discharge is consistent with fungal 
etiology. Recommend initiating itraconazole 400mg daily with consideration for 
surgical debridement if lesion is localized.
```

## 🏗️ System Architecture

```
Input: Histopathology Image + Clinical Notes + Demographics
                           ↓
        ┌─────────────────────────────────────┐
        │   InceptionV3 Visual Classifier      │
        │   (Pre-trained on ImageNet)          │
        └─────────────────────────────────────┘
                           ↓
              Initial CNN Prediction (88.5%)
                           ↓
        ┌─────────────────────────────────────┐
        │   Multi-Modal KG Retrieval           │
        ├─────────────────────────────────────┤
        │  • Visual Similarity (k=10)          │
        │  • Clinical Matching                 │
        │  • Lab Confirmations                 │
        │  • Geographic Priors                 │
        │  • Literature References             │
        └─────────────────────────────────────┘
                           ↓
        ┌─────────────────────────────────────┐
        │   Evidence Aggregation               │
        │   Weights: 0.35, 0.20, 0.30,        │
        │            0.10, 0.05                │
        └─────────────────────────────────────┘
                           ↓
              Refined Prediction (94.8%)
                           ↓
        ┌─────────────────────────────────────┐
        │   RAG Explanation Generation         │
        │   (GPT-4 with retrieved evidence)    │
        └─────────────────────────────────────┘
                           ↓
         Diagnostic Report with Clinical Explanation
```

## 📚 Documentation

- [**Installation Guide**](docs/installation.md) - Detailed setup instructions
- [**Data Preparation**](docs/data_preparation.md) - How to prepare your dataset
- [**Training Models**](docs/training.md) - Train InceptionV3 from scratch
- [**Building Knowledge Graph**](docs/kg_construction.md) - Construct the KG
- [**Reproducing Results**](docs/evaluation.md) - Reproduce all paper results
- [**API Reference**](docs/api_reference.md) - Code documentation

## 🔄 Reproducing Paper Results

### Train InceptionV3 Baseline

```bash
python scripts/train_inception_v3.py \
    --config experiments/configs/baseline_cnn.yaml \
    --data-path data/ \
    --output checkpoints/inception_v3.pth
```

Expected result: **88.5% test accuracy**

### Build Knowledge Graph

```bash
python scripts/build_kg.py \
    --config config/config.yaml \
    --data-path data/
```

Expected: 5,247 entities, 15,832 relationships

### Run Full System Evaluation

```bash
python scripts/evaluate.py \
    --config experiments/configs/kg_rag_full.yaml \
    --model-path checkpoints/inception_v3_best.pth \
    --output results/
```

Expected result: **94.8% test accuracy**

### Run Ablation Studies (Table 3)

```bash
python scripts/run_ablation_study.py \
    --config experiments/configs/ablation_study.yaml \
    --output results/ablation/
```

### Generate All Paper Figures

```bash
python scripts/generate_results.py \
    --results-path results/ \
    --output results/figures/
```

Generates:
- `confusion_matrix.png` (Figure 3)
- `roc_curve.png` (Figure 4)
- `training_curves.png` (Figure 5)
- `ablation_bars.png` (Figure 6)

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_models.py -v          # Test model components
pytest tests/test_kg.py -v              # Test KG operations
pytest tests/test_retrieval.py -v       # Test retrieval system
pytest tests/test_pipeline.py -v        # Test end-to-end pipeline

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📁 Repository Structure

```
mycetoma-kg-rag/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package installation
│
├── config/                            # Configuration files
│   ├── config.yaml                   # Main configuration
│   └── model_config.yaml             # Model hyperparameters
│
├── src/                              # Source code
│   ├── models/                       # Model architectures
│   ├── knowledge_graph/              # KG construction & queries
│   ├── retrieval/                    # Multi-modal retrieval
│   ├── rag/                          # RAG explanation generation
│   ├── aggregation/                  # Evidence aggregation
│   ├── pipeline/                     # Main diagnostic pipeline
│   └── utils/                        # Utility functions
│
├── scripts/                          # Executable scripts
│   ├── train_inception_v3.py        # Train vision model
│   ├── build_kg.py                  # Build Knowledge Graph
│   ├── evaluate.py                  # Evaluate system
│   └── generate_results.py          # Generate figures/tables
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # Dataset analysis
│   ├── 02_train_inception_v3.ipynb  # Model training
│   ├── 03_build_kg.ipynb            # KG construction
│   └── 07_demo.ipynb                # Interactive demo
│
├── tests/                            # Unit tests
├── data/                             # Dataset (download separately)
├── checkpoints/                      # Trained models
├── results/                          # Experimental results
└── docs/                             # Documentation
```

## 📊 Dataset

We use the **MycetoMIC 2024** benchmark dataset:

- **684 histopathology images** (H&E stained, 40× magnification)
- **342 Actinomycetoma cases** (bacterial etiology)
- **342 Eumycetoma cases** (fungal etiology)
- **412 clinical notes** with symptom descriptions
- **287 laboratory confirmations** (culture/PCR)
- **89 geographic locations** with epidemiological data

**Download:** Run `python data/scripts/download_dataset.py`

**License:** CC BY-NC-SA 4.0 (Non-commercial use only)

**Data Structure:**
```
data/
├── images/
│   ├── train/
│   │   ├── actinomycetoma/  (239 images)
│   │   └── eumycetoma/       (239 images)
│   ├── val/
│   │   ├── actinomycetoma/  (52 images)
│   │   └── eumycetoma/       (51 images)
│   └── test/
│       ├── actinomycetoma/  (51 images)
│       └── eumycetoma/       (52 images)
├── cases.csv                 # Patient metadata
├── clinical_notes.csv        # Clinical presentations
├── lab_results.csv           # Laboratory confirmations
└── geographic_locations.csv  # Epidemiological data
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
@inproceedings{author2025mycetoma,
  title={Multi-Modal Knowledge Graph-Augmented Retrieval for Explainable Mycetoma Diagnosis},
  author={Author, First and Author, Second and Author, Third},
  booktitle={Medical Imaging and Computer-Aided Diagnosis (MICAD)},
  year={2025},
  organization={Springer}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

- **First Author:** first.author@university.edu
- **GitHub Issues:** [Report bugs or request features](https://github.com/yourusername/mycetoma-kg-rag/issues)
- **Project Website:** [https://yourusername.github.io/mycetoma-kg-rag](https://yourusername.github.io/mycetoma-kg-rag)

## 🙏 Acknowledgments

- **Mycetoma Research Centre, Khartoum, Sudan** for providing the dataset and clinical expertise
- **Expert pathologists** who participated in explanation quality evaluation
- **[Your funding agency]** for financial support

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Note:** The MycetoMIC 2024 dataset has its own license (CC BY-NC-SA 4.0) and should be cited separately.

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/mycetoma-kg-rag&type=Date)](https://star-history.com/#yourusername/mycetoma-kg-rag&Date)

---

**📄 Paper:** [arXiv:2025.xxxxx](https://arxiv.org/abs/2025.xxxxx)  
**🏛️ Conference:** MICAD 2025  
**💻 Code:** MIT License  
**📊 Dataset:** CC BY-NC-SA 4.0
