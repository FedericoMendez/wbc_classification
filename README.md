# 🧬 White Blood Cell (WBC) Classification

This repository implements a complete pipeline for **white blood cell classification** using deep learning, including data preprocessing, model training, feature extraction, and ensemble methods.

The project explores multiple architectures—**ConvNeXt, ResNet50, and Swin Transformer**—and emphasizes **ensemble strategies** to improve generalization and robustness across classes.

---

## 📂 Repository Structure

```
wbc_classification-main/
│
├── configs/                    # examples of YAML configuration files for experiments
│   ├── convnext.yaml
│   ├── lymphoid.yaml
│   ├── myeloid.yaml
│   ├── neutrophil.yaml
│   └── weightedsampler.yaml
│
├── notebooks/ (implicit)      # Exploration & experimentation
│   ├── cropping.ipynb
│   ├── data_cleaning.ipynb
│   ├── feature_extraction.ipynb
│   ├── models_and_submissions.ipynb
│   ├── specialist_datasets.ipynb
│   ├── errors_analysis.ipynb
│   └── tuning.ipynb
│
├── core scripts
│   ├── convnext.py            # ConvNeXt model implementation
│   ├── tuning.py              # Training / hyperparameter tuning
│   ├── ensemble.py            # Global ensemble methods
│   ├── ensemble_classwise.py  # Class-wise ensemble optimization
│   ├── extract_probs.py       # Extract model prediction probabilities
│   ├── create_denoised_dataset.py
│   ├── utils.py               # Utility functions
│   └── config.py              # Global configuration
│
├── data splits
│   ├── train_metadata.csv
│   ├── test_metadata.csv
│   ├── train_split.csv
│   └── val_split.csv
│
├── environment_full.yml       # Conda environment
├── Report.pdf                 # Project report
└── .gitignore
```

---

## 🚀 Features

* 🔬 **Deep learning models**

  * ConvNeXt (primary architecture)
  * ResNet50 (baseline CNN)
  * Swin Transformer (hierarchical vision transformer)

* 🧹 **Data preprocessing & denoising**

* 🧠 **Feature extraction pipelines**

* 📊 **Error analysis tools**

* ⚖️ **Advanced ensembling**

  * Global weighted ensembles
  * Class-wise optimized combinations (e.g. Dirichlet-based search)

* 🎯 **Specialist models per class group**

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd wbc_classification-main
```

### 2. Create environment

```bash
conda env create -f environment_full.yml
conda activate wbc-env
```

---

## 🧪 Workflow

### 1. Data Preparation

* Use notebooks such as:

  * `data_cleaning.ipynb`
  * `cropping.ipynb`

* Optionally create a denoised dataset:

```bash
python create_denoised_dataset.py
```

---

### 2. Model Training

Run training / tuning:

```bash
python tuning.py --config configs/convnext.yaml
```

Configurations control:

* model architecture (ConvNeXt / ResNet50 / Swin)
* training hyperparameters
* sampling strategies

---

### 3. Feature Extraction

```bash
python extract_probs.py
```

Used for:

* ensembling
* error analysis
* downstream optimization

---

### 4. Ensembling

#### Global ensemble

```bash
python ensemble.py
```

#### Class-wise ensemble (recommended)

```bash
python ensemble_classwise.py
```

This allows:

* different model weights per class
* improved handling of class imbalance
* better exploitation of **error diversity across architectures**

---

## 📊 Key Insights

* Architecturally different models (CNNs vs transformers) learn **complementary representations**
* Models with similar overall performance often fail on **different subsets of data**
* Combining **ConvNeXt, ResNet50, and Swin Transformer** improves robustness
* **Class-wise ensembling** significantly outperforms global weighting
* Simple, well-regularized search strategies (e.g. Dirichlet-based weighting) generalize better than more complex optimization methods

---

## 📈 Results

See:

* `Report.pdf` for methodology and quantitative results

