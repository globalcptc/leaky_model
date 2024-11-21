# Leaky Model

This project demonstrates how language models can potentially leak sensitive training data. It provides a modular implementation of an LSTM-based text generation model, complete with preprocessing, training, and generation capabilities.

## Project Structure

```
leaky_model/
├── data/
│   ├── raw/             # Raw PDF files
│   ├── processed/       # Preprocessed markdown files
│   └── tmp/            # Temporary files during processing
├── model/              # Trained models and processors
│   ├── text_generation_model.keras
│   └── text_processor.pkl
├── src/
│   ├── preprocessing/  # PDF and image processing
│   │   ├── pdf_processor.py
│   │   ├── image_enhancer.py
│   │   └── text_cleaner.py
│   ├── training/      # Model training components
│   │   ├── model_builder.py
│   │   └── text_processor.py
│   ├── utils/         # Utility functions
│   │   ├── graceful_killer.py
│   │   ├── progress_tracker.py
│   │   └── text_file_reader.py
│   ├── config.py      # Configuration settings
│   └── prompts.txt    # Example prompts for generation
├── preprocess.py      # PDF preprocessing script
├── train.py          # Model training script
└── generate.py       # Text generation script
```

## Setup

This project uses poetry for dependency management. To get started:

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

Alternatively, you can use pip with the provided requirements.txt:

```bash
pip install -r requirements.txt
```

### System Requirements

- Python 3.12
- Tesseract OCR
- OpenCV

## Usage

The project is divided into three main steps:

### 1. Preprocessing PDFs

![an image showing example output of the preprocessing](https://github.com/globalcptc/leaky_model/blob/main/docs/img/preprocessing.png)

Convert PDF files to preprocessed markdown format:

```bash
python preprocess.py --input-dir data/raw --output-dir data/processed

# Additional options:
#   --temp-dir data/tmp     # Directory for temporary files
```

Features:
- Multi-threaded PDF processing
- OCR for scanned documents
- Image enhancement for better text extraction
- Progress tracking and resumable processing
- Graceful shutdown handling

### 2. Training the Model

![an image showing example output of the training step](https://github.com/globalcptc/leaky_model/blob/main/docs/img/training.png)

Train the LSTM model on preprocessed data:

```bash
python train.py --data-dir data/processed --model-dir model

# Additional options:
#   --sequence-length 50    # Length of input sequences
#   --embedding-dim 100     # Dimension of word embeddings
#   --batch-size 128       # Training batch size
```

The training process:
1. Processes markdown files
2. Fits tokenizer to vocabulary
3. Creates training sequences
4. Trains LSTM model with progress tracking
5. Saves model and processor files

### 3. Generating Text

Generate text using the trained model:

```bash
python generate.py --prompts-file src/prompts.txt

# Additional options:
#   --model-dir model      # Directory containing model files
#   --num-words 50        # Number of words to generate
#   --temperature 1.0     # Sampling temperature (higher = more random)
#   --top-k 0            # Top-k sampling parameter
#   --top-p 0.0          # Nucleus sampling parameter
```

## Key Components

### Preprocessing
- `PDFProcessor`: Handles PDF reading and text extraction
- `ImageEnhancer`: Improves image quality for OCR
- `TextCleaner`: Normalizes and cleans extracted text

### Training
- `ModelBuilder`: Creates and configures the LSTM model
- `TextProcessor`: Handles text tokenization and sequence creation

### Utils
- `GracefulKiller`: Manages graceful shutdown of long-running processes
- `ProgressTracker`: Tracks and saves processing progress
- `TextFileReader`: Efficient reading of text/markdown files

## Configuration

Global settings are managed in `src/config.py`:
- Path configurations
- Model parameters
- Processing settings
- Default values

## Example Output

Here are some examples of model outputs that illustrate potential data leakage:

### matching output to training data

![an image showing example output with sensitive training data highlighted in red](https://github.com/globalcptc/leaky_model/blob/main/docs/img/match-to-training-data.png)

### highlighted sensitive training data in output

![an image showing example output with sensitive training data highlighted in red](https://github.com/globalcptc/leaky_model/blob/main/docs/img/example-leaked-data.png)


## Important Note

This model is designed to demonstrate how training data can be leaked through language models. It should be used responsibly and only with data you have permission to use.

## License

Apache 2.0 License
