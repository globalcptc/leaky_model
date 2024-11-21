# train.py
import argparse
from pathlib import Path
from src.config import Config
from src.training.model_builder import ModelBuilder
from src.training.text_processor import TextProcessor
from src.utils.text_file_reader import TextFileReader
from src.utils.progress_tracker import ProgressTracker
from src.utils.graceful_killer import GracefulKiller


def main():
    # Set up environment and directories
    Config.setup_environment()
    Config.setup_directories()

    parser = argparse.ArgumentParser(
        description='Train LSTM model on preprocessed markdown files.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(Config.PROCESSED_DATA_DIR),
        help='Directory containing preprocessed markdown files'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=str(Config.MODEL_DIR),
        help='Directory to save trained model and tokenizer'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=Config.DEFAULT_SEQUENCE_LENGTH,
        help='Length of input sequences'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=Config.DEFAULT_EMBEDDING_DIM,
        help='Dimension of word embeddings'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=Config.DEFAULT_BATCH_SIZE,
        help='Training batch size'
    )
    args = parser.parse_args()

    # Initialize components
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize progress tracker
    progress_file = model_dir / '.training_progress.json'
    progress_tracker = ProgressTracker(progress_file)

    # Initialize graceful shutdown handler
    killer = GracefulKiller()

    reader = TextFileReader(args.data_dir, file_type="markdown")
    processor = TextProcessor(sequence_length=args.sequence_length)

    print("\nStarting model training...")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Batch size: {args.batch_size}")

    # First pass: fit tokenizer if not already done
    if not progress_tracker.is_completed("tokenizer_fitting"):
        print("\nFitting tokenizer...")
        all_texts = []
        for filename, content in reader:
            if content and not killer.kill_now:
                all_texts.append(content)
                progress_tracker.update_file_progress(
                    Path(filename),
                    completed=True,
                    stage="tokenizer_fitting"
                )

        if not all_texts:
            raise ValueError("No valid training data found")

        processor.fit_tokenizer(all_texts)
        print(f"\nVocabulary size: {processor.vocab_size:,} words")

        # Save the processor
        processor_path = model_dir / 'text_processor.pkl'
        processor.save(processor_path)
        print(f"Saved text processor to {processor_path}")

        progress_tracker.update_file_progress(
            Path("tokenizer"),
            completed=True,
            stage="tokenizer_fitting"
        )
    else:
        print("\nLoading existing tokenizer...")
        processor = TextProcessor.load(model_dir / 'text_processor.pkl')
        print(f"Vocabulary size: {processor.vocab_size:,} words")

    # Create and configure the model
    print("\nCreating model...")
    model = ModelBuilder.create_model(
        vocab_size=processor.vocab_size,
        sequence_length=processor.sequence_length,
        embedding_dim=args.embedding_dim
    )
    model.summary()

    # Train the model
    print("\nTraining model...")
    total_sequences = 0
    total_files = 0

    # Get remaining files to process
    remaining_files = progress_tracker.get_remaining_files([
        Path(f) for f in reader.files
    ])

    print(f"Files to process: {len(remaining_files)}")

    for file_path in remaining_files:
        if killer.kill_now:
            print("\nGraceful shutdown requested...")
            break

        try:
            content = reader.read_file(file_path.name)
            if not content:
                continue

            X, y = processor.create_sequences(content)
            if len(X) > 0:
                model.fit(
                    X, y,
                    epochs=1,
                    batch_size=args.batch_size,
                    verbose=0
                )
                total_sequences += len(X)
                total_files += 1

                # Update progress
                progress_tracker.update_file_progress(
                    file_path,
                    completed=True,
                    metadata={
                        "sequences": len(X),
                        "total_sequences": total_sequences
                    }
                )

                # Print progress
                print(f"\rProcessed: {total_files} files, {total_sequences:,} sequences",
                      end="", flush=True)

        except Exception as e:
            print(f"\nError processing {file_path}: {str(e)}")
            continue

    print(f"\n\nTraining completed:")
    print(f"- Processed {total_files:,} files")
    print(f"- Total sequences trained on: {total_sequences:,}")
    if total_files > 0:
        print(
            f"- Average sequences per file: {total_sequences/total_files:,.1f}")

    # Save the model
    try:
        model_path = model_dir / 'text_generation_model.keras'
        model.save(model_path)
        print(f"\nModel saved to {model_path}")

        # Mark training as complete
        progress_tracker.update_file_progress(
            Path("model"),
            completed=True,
            stage="model_training",
            metadata={
                "total_files": total_files,
                "total_sequences": total_sequences,
                "avg_sequences_per_file": total_sequences/total_files if total_files > 0 else 0
            }
        )
    except Exception as e:
        print(f"\nError saving model: {str(e)}")
        raise
    finally:
        progress_tracker.save_progress()


if __name__ == "__main__":
    main()
