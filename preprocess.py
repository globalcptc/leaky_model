# preprocess.py
import argparse
from pathlib import Path
from src.preprocessing.pdf_processor import PDFProcessor
from src.utils.graceful_killer import GracefulKiller
from src.config import Config


def main():
    # Set up environment and directories
    Config.setup_environment()
    Config.setup_directories()

    parser = argparse.ArgumentParser(
        description='Preprocess PDF files into markdown for LSTM training.'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(Config.RAW_DATA_DIR),
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Config.PROCESSED_DATA_DIR),
        help='Directory to store processed markdown files'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=str(Config.TEMP_DIR),
        help='Directory for temporary files'
    )
    args = parser.parse_args()

    # Create directories if they don't exist
    for dir_path in [args.input_dir, args.output_dir, args.temp_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Initialize the processor
    processor = PDFProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir
    )

    print("\nStarting PDF preprocessing...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Temporary directory: {args.temp_dir}")

    # Process the files
    try:
        processor.process_files()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Saving progress...")
    except Exception as e:
        print(f"\nError during processing: {type(e).__name__}: {str(e)}")
    finally:
        print("\nPreprocessing completed. Check the output directory for results.")


if __name__ == "__main__":
    main()
