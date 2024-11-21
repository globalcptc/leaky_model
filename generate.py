# generate.py
import argparse
from datetime import datetime
from pathlib import Path
from typing import List
from tensorflow.keras.models import load_model
from src.config import Config
from src.training.text_processor import TextProcessor
from src.utils.progress_tracker import ProgressTracker
from src.utils.graceful_killer import GracefulKiller


def read_prompts(prompts_file: Path) -> List[str]:
    """Read prompts from a file, one per line."""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def generate_text(model, processor, seed_text: str, **params) -> str:
    """Generate text using the trained model."""
    return processor.generate_text(model, seed_text, **params)


def main():
    # Set up environment and directories
    Config.setup_environment()
    Config.setup_directories()

    parser = argparse.ArgumentParser(
        description='Generate text using trained LSTM model.'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=str(Config.MODEL_DIR),
        help='Directory containing trained model and processor'
    )
    parser.add_argument(
        '--prompts-file',
        type=str,
        default=str(Config.BASE_DIR / 'prompts.txt'),
        help='File containing prompts for text generation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Config.DATA_DIR / 'generated'),
        help='Directory to save generated text'
    )
    parser.add_argument(
        '--num-words',
        type=int,
        default=Config.DEFAULT_NUM_WORDS,
        help='Number of words to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=Config.DEFAULT_TEMPERATURE,
        help='Sampling temperature (higher = more random)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=Config.DEFAULT_TOP_K,
        help='Top-k sampling parameter (0 to disable)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=Config.DEFAULT_TOP_P,
        help='Nucleus sampling parameter (0.0 to disable)'
    )
    args = parser.parse_args()

    # Initialize directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize progress tracking
    progress_file = output_dir / '.generation_progress.json'
    progress_tracker = ProgressTracker(progress_file)

    # Initialize graceful shutdown handler
    killer = GracefulKiller()

    # Load model and processor
    try:
        model_path = Path(args.model_dir) / 'text_generation_model.keras'
        processor_path = Path(args.model_dir) / 'text_processor.pkl'

        print("\nLoading model and processor...")
        model = load_model(model_path)
        processor = TextProcessor.load(processor_path)
        print("Model and processor loaded successfully!")
    except Exception as e:
        print(f"\nError loading model or processor: {str(e)}")
        return

    # Read prompts
    try:
        prompts = read_prompts(Path(args.prompts_file))
        print(f"\nFound {len(prompts)} prompts to process")
    except Exception as e:
        print(f"\nError reading prompts file: {str(e)}")
        return

    # Generation parameters
    params = {
        'num_words': args.num_words,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p
    }

    print("\nGeneration parameters:")
    print("-" * 20)
    for param, value in params.items():
        print(f"{param:12} = {value}")

    # Generate text for each prompt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"generated_text_{timestamp}.txt"

    print(f"\nGenerating text...")
    print(f"Output will be saved to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, prompt in enumerate(prompts, 1):
            if killer.kill_now:
                print("\nGraceful shutdown requested...")
                break

            print(f"\n[{i}/{len(prompts)}] Prompt: '{prompt}'")
            print("-" * 50)

            try:
                if not progress_tracker.is_completed(prompt):
                    generated = generate_text(
                        model, processor, prompt, **params)
                    if generated:
                        # Save to file
                        f.write(f"Prompt: {prompt}\n")
                        f.write("-" * 50 + "\n")
                        f.write(generated + "\n")
                        f.write("=" * 50 + "\n\n")
                        f.flush()  # Ensure content is written immediately

                        # Print to console
                        print(generated)
                        print("-" * 50)

                        # Update progress
                        progress_tracker.update_file_progress(
                            Path(prompt),
                            completed=True,
                            metadata={
                                "output_file": str(output_file),
                                "parameters": params,
                                "length": len(generated.split())
                            }
                        )
                    else:
                        print("Failed to generate text for this prompt")
                else:
                    print("Prompt already processed, skipping...")

            except Exception as e:
                print(f"Error generating text: {str(e)}")
                progress_tracker.update_file_progress(
                    Path(prompt),
                    completed=False,
                    metadata={"error": str(e)}
                )
            finally:
                progress_tracker.save_progress()

    print("\nGeneration completed!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
