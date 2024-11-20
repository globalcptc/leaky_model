import os
import string
import numpy as np
from pickle import dump, load
from typing import Iterator, Optional, List, Tuple
from keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm  # for cli
# from tqdm import tqdm_notebook as tqdm # for jupyter notebook
from tqdm.keras import TqdmCallback
from pathlib import Path
import re


class TextFileReader:
    """A class to handle reading text and markdown files from a directory efficiently"""

    def __init__(self, directory_path: str, file_type: str = "text"):
        self.directory_path = directory_path
        self.file_type = file_type.lower()
        if self.file_type not in ["text", "markdown"]:
            raise ValueError('file_type must be either "text" or "markdown"')
        self._files = None

    @property
    def files(self) -> list[str]:
        """Lazy loading of file list"""
        if self._files is None:
            extension = ".md" if self.file_type == "markdown" else ".txt"
            self._files = [
                f for f in os.listdir(self.directory_path)
                if f.endswith(extension) and os.path.isfile(os.path.join(self.directory_path, f))
            ]
        return self._files

    def clean_markdown(self, content: str) -> str:
        """Clean markdown content to extract meaningful text while preserving structure."""
        # Remove code blocks
        content = re.sub(r'```[\s\S]*?```', '', content)
        # Remove inline code
        content = re.sub(r'`[^`]*`', '', content)
        # Remove images
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        # Remove links but keep link text
        content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', content)
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        # Remove markdown tables
        content = re.sub(r'^\|.*\|$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\|-+\|$', '', content, flags=re.MULTILINE)
        # Convert headers to plain text
        content = re.sub(r'^#+\s*(.*?)$', r'\1', content, flags=re.MULTILINE)
        # Remove bold and italic markers
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
        content = re.sub(r'\*(.*?)\*', r'\1', content)
        content = re.sub(r'_(.*?)_', r'\1', content)
        # Remove horizontal rules
        content = re.sub(r'^\s*[-*_]{3,}\s*$', '', content, flags=re.MULTILINE)
        # Remove multiple newlines and spaces
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        return content.strip()

    def read_file(self, filename: str) -> Optional[str]:
        """Read and clean a specific file"""
        if filename not in self.files:
            raise ValueError(f"File {filename} not found in directory")

        file_path = os.path.join(self.directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            if self.file_type == "markdown":
                return self.clean_markdown(content)
            return content

        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            return None

    def iter_files(self) -> Iterator[tuple[str, str]]:
        """Generator that yields (filename, content) pairs"""
        for filename in tqdm(self.files, desc="Reading files", dynamic_ncols=True, position=0, leave=True):
            content = self.read_file(filename)
            if content is not None:
                yield filename, content

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Makes the class iterable"""
        return self.iter_files()


class TextProcessor:
    def __init__(self, sequence_length: int = 50):
        self.tokenizer = Tokenizer()
        self.sequence_length = sequence_length
        self.vocab_size = 0

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
        return text

    def fit_tokenizer(self, texts: List[str]):
        """Fit the tokenizer on the texts"""
        with tqdm(total=len(texts), desc="Fitting tokenizer", dynamic_ncols=True, position=0, leave=True) as pbar:
            cleaned_texts = []
            for text in texts:
                cleaned_texts.append(self.clean_text(text))
                pbar.update(1)
            self.tokenizer.fit_on_texts(cleaned_texts)
            self.vocab_size = len(self.tokenizer.word_index) + 1

    def create_sequences(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences from text"""
        cleaned_text = self.clean_text(text)
        encoded = self.tokenizer.texts_to_sequences([cleaned_text])[0]
        sequences = []

        for i in range(0, len(encoded) - self.sequence_length):
            sequence = encoded[i:i + self.sequence_length + 1]
            sequences.append(sequence)

        if not sequences:
            return np.array([]), np.array([])

        sequences = np.array(sequences)
        X = sequences[:, :-1]
        y = sequences[:, -1]
        y = to_categorical(y, num_classes=self.vocab_size)
        return X, y

    def save(self, filepath: str):
        """Save the tokenizer and parameters"""
        config = {
            'tokenizer': self.tokenizer,
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'wb') as f:
            dump(config, f)

    @classmethod
    def load(cls, filepath: str) -> 'TextProcessor':
        """Load a saved TextProcessor"""
        with open(filepath, 'rb') as f:
            config = load(f)
        processor = cls(sequence_length=config['sequence_length'])
        processor.tokenizer = config['tokenizer']
        processor.vocab_size = config['vocab_size']
        return processor


def create_model(vocab_size: int, sequence_length: int, embedding_dim: int = 100) -> Model:
    """Create the LSTM model"""
    inputs = Input(shape=(sequence_length,))
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = LSTM(150, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(150)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def main():
    """Main training function with improved progress tracking"""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train text generation model on text or markdown files.')
    parser.add_argument('--file-type', type=str, choices=['text', 'markdown'],
                        default='markdown',
                        help='Type of files to process (text or markdown)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing training files')
    args = parser.parse_args()

    # Set up paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        base_dir = Path("./training_data")
        data_dir = base_dir / \
            ('markdown' if args.file_type == 'markdown' else 'text')

    if not data_dir.exists():
        raise ValueError(f"Training data directory not found: {data_dir}")

    print(f"\nTraining on {args.file_type} files from: {data_dir}")

    # Initialize reader and processor
    reader = TextFileReader(data_dir, file_type=args.file_type)
    processor = TextProcessor(sequence_length=50)

    # First pass: fit tokenizer
    print("\nProcessing files and fitting tokenizer...")
    all_texts = []
    for filename, content in reader:
        if content:
            all_texts.append(content)

    if not all_texts:
        raise ValueError("No valid training data found")

    processor.fit_tokenizer(all_texts)
    print(f"\nVocabulary size: {processor.vocab_size:,} words")
    processor.save('text_processor.pkl')
    print("Saved text processor to text_processor.pkl")

    # Create model and show summary
    print("\nCreating model...")
    model = create_model(
        vocab_size=processor.vocab_size,
        sequence_length=processor.sequence_length
    )
    model.summary()

    # Second pass: create sequences and train
    print("\nTraining model...")
    total_sequences = 0
    total_files = 0

    progress_bar = tqdm(reader.files, desc="Training files",
                        dynamic_ncols=True, position=0, leave=True)

    class CustomCallback(TqdmCallback):
        def on_epoch_end(self, epoch, logs=None):
            pass  # Suppress per-epoch output

    for filename in progress_bar:
        content = reader.read_file(filename)
        if not content:
            continue

        X, y = processor.create_sequences(content)

        if len(X) > 0:
            model.fit(
                X, y,
                epochs=1,
                batch_size=128,
                verbose=0,
                callbacks=[CustomCallback(verbose=0)]
            )
            total_sequences += len(X)
            total_files += 1

        progress_bar.set_description(
            f"Training files - Processed: {total_files}, Sequences: {total_sequences:,}"
        )

    progress_bar.close()

    print(f"\nTraining completed:")
    print(f"- Processed {total_files:,} files")
    print(f"- Total sequences trained on: {total_sequences:,}")
    print(f"- Average sequences per file: {total_sequences/total_files:,.1f}")

    # Save the model
    model.save('text_generation_model.keras')
    print("\nModel saved to text_generation_model.keras")


if __name__ == "__main__":
    main()
