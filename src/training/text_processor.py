# src/training/text_processor.py
import string
from typing import List, Tuple
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from pickle import dump, load


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

    def generate_text(self, model, seed_text: str, num_words: int = 50,
                      temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> str:
        """Generate text based on a seed string."""
        try:
            current_sequence = self.clean_text(seed_text)
            if not current_sequence:
                return None

            result = current_sequence

            for _ in range(num_words):
                sequence = self.tokenizer.texts_to_sequences([current_sequence])[
                    0]
                # Keep only last sequence_length tokens
                sequence = sequence[-self.sequence_length:]

                # Pad sequence if needed
                if len(sequence) < self.sequence_length:
                    sequence = [0] * (self.sequence_length -
                                      len(sequence)) + sequence

                sequence = np.array([sequence])

                # Get raw predictions
                pred_probs = model.predict(sequence, verbose=0)[0]

                # Apply temperature scaling
                if temperature != 1.0:
                    pred_probs = np.log(pred_probs) / temperature
                    pred_probs = np.exp(pred_probs)
                    pred_probs = pred_probs / np.sum(pred_probs)

                # Apply top-k filtering
                if top_k > 0:
                    top_k_idx = np.argsort(pred_probs)[-top_k:]
                    mask = np.zeros_like(pred_probs)
                    mask[top_k_idx] = 1
                    pred_probs = pred_probs * mask
                    pred_probs = pred_probs / np.sum(pred_probs)

                # Apply nucleus (top-p) sampling
                if top_p > 0.0:
                    sorted_probs = np.sort(pred_probs)[::-1]
                    cumsum_probs = np.cumsum(sorted_probs)
                    cutoff = sorted_probs[np.argmax(cumsum_probs > top_p)]
                    mask = pred_probs >= cutoff
                    pred_probs = pred_probs * mask
                    pred_probs = pred_probs / np.sum(pred_probs)

                # Sample from the modified distribution
                pred_word_idx = np.random.choice(len(pred_probs), p=pred_probs)

                # Convert predicted word index back to word
                for word, index in self.tokenizer.word_index.items():
                    if index == pred_word_idx:
                        result += " " + word
                        current_sequence = " ".join(
                            result.split()[-self.sequence_length:])
                        break

            return result
        except Exception as e:
            print(f"Error generating text for seed '{seed_text}': {str(e)}")
            return None

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
