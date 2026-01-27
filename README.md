# Computer Vision Classification

A lightweight PyTorch image classification project using machine learning models for multi-class classification tasks.

## Features

- Custom CSV-based dataset loader
- Data augmentation pipeline
- Modified LeNet-5 architecture
- GPU acceleration (CUDA/MPS/CPU)
- Training with validation metrics

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd computervision-classification

# Install dependencies
pip install -e .
```

## Setup

Create a `.env` file in the project root:

```env
ROOT_DIR=/path/to/your/data
TRAIN_CSV=train.csv
TARGET_COL_NAME=label
IMAGE_SHAPE=224
```

Your CSV file should have columns: `image_path` and your target column (e.g., `label`).

## Usage

Train a model:

```bash
python training/train_lenet.py --epochs 10 --batchsize 32 --augmentation True
```

### Options

- `-a, --augmentation`: Enable data augmentation (default: False)
- `-i, --imagesize`: Input image size (default: 224)
- `-s, --trainsplitratio`: Training data ratio (default: 0.8)
- `-b, --batchsize`: Batch size (default: 16)
- `-e, --epochs`: Number of epochs (default: 5)

## Project Structure

```
├── core/
│   └── config.py          # Configuration settings
├── utility/
│   ├── load_data.py       # Dataset loader
│   └── transform_data.py  # Data transformations
├── training/
│   └── train_lenet.py     # Training script
└── pyproject.toml         # Dependencies
```

## Requirements

- Python ≥ 3.13
- PyTorch ≥ 2.10.0
- See `pyproject.toml` for full dependencies

## License

MIT