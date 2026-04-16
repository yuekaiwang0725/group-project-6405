# GRU with Attention Model for Text Classification

This folder contains the implementation of a **Bi-directional GRU with Bahdanau Attention** model for text classification across three NLP tasks: emotion classification, IMDB sentiment analysis, and SST-2 sentiment analysis.

## Overview

The `gru_attention` module provides a complete pipeline for text classification using a neural network architecture that combines:
- **Bi-directional GRU** for capturing contextual information from both directions
- **Bahdanau Attention** for focusing on important tokens in the input sequence
- **Task-specific adaptation** for multi-class and binary classification problems

## Folder Structure

```
gru_attention/
├── __pycache__/                    # Python bytecode cache (auto-generated)
├── report/                         # Evaluation reports (generated after testing)
│   ├── emotion_report.json         # Emotion dataset evaluation results
│   ├── imdb_report.json           # IMDB dataset evaluation results  
│   └── sst2_report.json           # SST-2 dataset evaluation results
├── trained_models/                 # Pre-trained model weights
│   ├── gru_emotion_model.pth      # Emotion classification model (6 classes)
│   ├── gru_imdb_model.pth         # IMDB sentiment model (binary)
│   └── gru_sst2_model.pth         # SST-2 sentiment model (binary)
├── cleanup.py                      # Utility to delete generated files
├── config.py                       # Configuration management
├── dataset.py                      # Dataset loading and preprocessing
├── gru_demo.py                     # Streamlit interactive demo
├── model.py                        # GRU + Attention model architecture
├── test.py                         # Model evaluation script
└── train.py                        # Model training script
```

## File Descriptions

### Core Files

1. **`config.py`** - Dynamic configuration management
   - Automatically detects project structure and builds paths
   - Supports three tasks: `emotion` (6-class), `imdb` (binary), `sst2` (binary)
   - Task-specific parameters: sequence lengths, class counts, data paths
   - Common hyperparameters: vocab_size=8000, embedding_dim=128, hidden_dim=256

2. **`model.py`** - Neural network architecture
   - Implements `BahdanauAttention` mechanism with learnable weight matrices
   - Implements `GRUAttention` main model with embedding → Bi-GRU → Attention → FC layers
   - Includes padding masking to ignore padding tokens during attention computation

3. **`dataset.py`** - Data processing utilities
   - `Vocab` class for vocabulary building with text cleaning and tokenization
   - `EmotionDataset` PyTorch Dataset class for text classification
   - `get_dataloader()` function to create train/validation DataLoaders
   - Text cleaning: lowercasing, punctuation removal, whitespace normalization

4. **`train.py`** - Training pipeline
   - `train_single_task()` function to train model for a specific task
   - Automatic sequential training of all three tasks when run as main
   - Training loop with loss calculation, backpropagation, and validation
   - Model saving to `trained_models/` directory

5. **`test.py`** - Model evaluation
   - `run_final_evaluation()` function to evaluate models and generate performance reports
   - Batch processing for all three tasks sequentially
   - Generates JSON reports with accuracy, sample counts, and architecture info

6. **`gru_demo.py`** - Interactive web demo
   - Streamlit-based web application with four tabs
   - Real-time text inference with immediate prediction
   - Attention visualization heatmap showing token importance weights
   - Performance metrics display from JSON reports

7. **`cleanup.py`** - Cleanup utility
   - Safe deletion of generated files from `trained_models/` and `report/` folders
   - Requires user confirmation before deletion
   - Verifies file existence and handles exceptions gracefully

## How to Train the Models

### Prerequisites
1. Ensure you have Python 3.10+ installed
2. Install required dependencies:
```bash
pip install torch streamlit pandas numpy matplotlib seaborn scikit-learn
```
Or install from the project root `requirements.txt`:
```bash
pip install -r ../requirements.txt
```

### Prepare data 
The data is from the .\GROUP-PROJECT-6045\data\processed
If you not find or miss any '.csv' data file, you can try this way to rebuild them:

```bash
python -m src.main prepare-data
```
Tip: when run this code make sure your TERMINAL location is: `group-project-6405/`
and run other code please make your TERMINAL location is: `group-project-6405/gru_attention/`


### Training Process
To train all three models sequentially:
```bash
python train.py
```

This will:
1. Train the emotion classification model (6 classes)
2. Train the IMDB sentiment analysis model (binary)
3. Train the SST-2 sentiment analysis model (binary)
4. Save trained models to `trained_models/` directory

**Training Details:**
- Batch size: 64
- Learning rate: 0.001
- Number of epochs: 20
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Device: Automatically uses CUDA if available, otherwise CPU


## How to Test the Models

To evaluate the trained models and generate performance reports:
```bash
python test.py
```

This will:
1. Load each trained model from `gru_attention/trained_models/` directory
2. Evaluate on the respective test datasets
3. Generate JSON reports in `gru_attention/report/` directory with:
   - Accuracy scores
   - Sample counts
   - Model architecture information
   - Task-specific metrics

**Expected Performance:**
- Emotion classification: ~91.10% accuracy
- IMDB sentiment analysis: ~86.56% accuracy  
- SST-2 sentiment analysis: ~82.11% accuracy


## How to Delete Previous Data

To safely delete generated files (trained models and evaluation reports):
```bash
python cleanup.py
```

The script will:
1. List all files to be deleted from `trained_models/` and `report/` folders
2. Ask for confirmation before proceeding
3. Delete only the generated files, preserving source code

**Files that will be deleted:**
- `trained_models/gru_emotion_model.pth`
- `trained_models/gru_imdb_model.pth`
- `trained_models/gru_sst2_model.pth`
- `report/emotion_report.json`
- `report/imdb_report.json`
- `report/sst2_report.json`


## How to Run the Demo

To launch the interactive web demo:
```bash
streamlit run gru_demo.py
```

The demo will open in your default web browser at `http://localhost:8501`.

### Demo Features

The demo includes four tabs:

1. **Project Report** - Overview of the project and model performance
2. **Interactive Demo** - Real-time text classification with attention visualization
   - Enter any text in the input box
   - Select the task (Emotion, IMDB, or SST-2)
   - View the predicted class and confidence score
   - See attention heatmap showing which words influenced the decision
3. **Source Code** - Browse and view the implementation code
4. **References** - Citations and resources used in the project

## Pre-trained Models

Three pre-trained models are included:

1. **`gru_emotion_model.pth`** (7.5MB)
   - Task: 6-class emotion classification
   - Classes: sadness, joy, love, anger, fear, surprise
   - Accuracy: 91.10% on test set

2. **`gru_imdb_model.pth`** (7.5MB)
   - Task: Binary sentiment analysis on IMDB reviews
   - Classes: negative, positive
   - Accuracy: 86.56% on test set

3. **`gru_sst2_model.pth`** (7.5MB)
   - Task: Binary sentiment analysis on SST-2 dataset
   - Classes: negative, positive
   - Accuracy: 82.11% on test set

## Model Architecture

### Bi-directional GRU
- Processes sequences in both forward and backward directions
- Hidden dimension: 256
- Number of layers: 2
- Dropout: 0.5

### Bahdanau Attention
- Attention mechanism with learnable weight matrices
- Computes context vector as weighted sum of hidden states
- Includes padding masking to ignore padding tokens

### Full Architecture
```
Input Text → Embedding (128-dim) → Bi-GRU (256-dim) → Attention → 
FC Layer (128-dim) → Output Layer (task-specific)
```

## Configuration

The `config.py` file provides task-specific configurations:

| Parameter | Emotion | IMDB | SST-2 |
|-----------|---------|------|-------|
| Sequence Length | 128 | 256 | 128 |
| Number of Classes | 6 | 2 | 2 |
| Vocabulary Size | 8000 | 8000 | 8000 |
| Embedding Dimension | 128 | 128 | 128 |
| Hidden Dimension | 256 | 256 | 256 |

## Dependencies

- `torch` >= 2.0.0
- `streamlit` >= 1.28.0
- `pandas` >= 2.0.0
- `numpy` >= 1.24.0
- `matplotlib` >= 3.7.0
- `seaborn` >= 0.12.0
- `scikit-learn` >= 1.3.0

## Quick Start Summary

1. **Install dependencies**: `pip install -r ../requirements.txt`
2. **Train models**: `python train.py`
3. **Test models**: `python test.py`
4. **Run demo**: `streamlit run gru_demo.py`
5. **Clean up**: `python cleanup.py`

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in `config.py`
2. **File not found errors**: Ensure data files are in the correct location
3. **Streamlit port already in use**: Change port with `streamlit run gru_demo.py --server.port 8502`
4. **Import errors**: Check Python version (requires 3.10+) and installed packages

### Getting Help

For issues with this module, check:
- The source code comments in each file
- The project root README for general setup instructions
- PyTorch and Streamlit documentation for framework-specific questions

## License

This code is part of the EE6405 NLP assignment project. See the project root for licensing information.

## Authors

- Model implementation and training pipeline
- Interactive demo with attention visualization
- Multi-task configuration system

---
*Last updated: April 2026*