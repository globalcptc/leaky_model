
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
from text_processor import TextProcessor
from text_file_reader import TextFileReader
from model import create_model

def main():
    """Main training function with improved progress tracking"""
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
