# Leaky Model

This is a simple [LSTM](https://www.sciencedirect.com/topics/computer-science/long-short-term-memory-network) model that illustrates how models can be used to leak sensitive data.

## Setup

This codebase uses poetry to manage dependencies. Install Poetry first, then:

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

Alternatively, you can use pip with the provided requirements.txt, though version compatibility isn't guaranteed:

```bash
pip install -r requirements.txt
```

Required system dependencies:
- Python 3.12
- Tesseract OCR
- OpenCV

## Data Preparation

The training process consists of two main steps:

1. **Data Preparation**: Converting PDFs to markdown files using OCR
2. **Model Training**: Training the LSTM model on the processed text

### Step 1: Preparing the Data

![an image showing example output of the preprocessing](https://github.com/globalcptc/leaky_model/blob/main/img/preprocessing.png)

The `data-prep.py` script handles the PDF to markdown conversion process:

1. Place your PDF files in the `training_data/pdf` directory
2. Run the data preparation script:
   ```bash
   python data-prep.py
   ```

The script will:
- Process PDFs in parallel using multiple worker threads
- Convert PDF pages to enhanced images
- Perform OCR using Tesseract with optimized settings
- Clean and normalize the extracted text
- Save the results as markdown files in `training_data/markdown`
- Track progress and can resume interrupted processing

Features:
- Graceful shutdown handling (Ctrl+C)
- Progress tracking for each worker
- Error handling and recovery
- Automated image enhancement for better OCR results
- Text cleaning and normalization

### Step 2: Training the Model

![an image showing example output of the training step](https://github.com/globalcptc/leaky_model/blob/main/img/training.png)

Once the data is prepared, use `train.py` to train the model:

```bash
python train.py
```

The script will:
1. Process the markdown files in `training_data/markdown`
2. Fit a tokenizer to the vocabulary
3. Create training sequences
4. Train the LSTM model

The training process generates two important files:

1. `text_processor.pkl`: Contains:
   - Fitted tokenizer with vocabulary
   - Sequence length configuration
   - Vocabulary size

2. `text_generation_model.keras`: The trained model including:
   - LSTM model architecture
   - Trained weights
   - Model configuration

## Generating Text

To generate content from the trained model:

1. Edit `prompts.txt` with your desired prompts (one per line)
2. Run the generator:
   ```bash
   python generate.py
   ```

Or specify a custom prompts file:
```bash
python generate.py your-prompts-file.txt
```

Generation parameters can be adjusted in `generate.py`:
- `num_words`: Number of words to generate
- `temperature`: Controls randomness (higher = more random)
- `top_k`: Limits to top K most likely tokens
- `top_p`: Uses nucleus sampling threshold


## Example Output Generations

## matching output to training data

![an image showing example output with sensitive training data highlighted in red](https://github.com/globalcptc/leaky_model/blob/main/img/match-to-training-data.png)

## highlighted sensitive training data in output

![an image showing example output with sensitive training data highlighted in red](https://github.com/globalcptc/leaky_model/blob/main/img/example-leaked-data.png)


## Important Note

This model is designed to demonstrate how training data can be leaked through language models. It should be used responsibly and only with data you have permission to use.
