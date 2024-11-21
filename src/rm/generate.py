import sys
import numpy as np
from typing import Optional
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from text_processor import TextProcessor


def generate_text(model, processor, seed_text: str, num_words: int = 50,
                  temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> Optional[str]:
    """
    Generate text based on a seed string.

    Args:
        model: The trained Keras model
        processor: TextProcessor instance
        seed_text: Starting text to generate from
        num_words: Number of words to generate
        temperature: Controls randomness (higher = more random, lower = more deterministic)
        top_k: If > 0, only sample from the top k most likely tokens
        top_p: If > 0.0, sample from the smallest set of tokens whose cumulative probability exceeds p
    """
    try:
        current_sequence = processor.clean_text(seed_text)
        if not current_sequence:
            return None

        result = current_sequence

        for _ in range(num_words):
            sequence = processor.tokenizer.texts_to_sequences([current_sequence])[
                0]
            sequence = pad_sequences(
                [sequence], maxlen=processor.sequence_length)

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
            for word, index in processor.tokenizer.word_index.items():
                if index == pred_word_idx:
                    result += " " + word
                    current_sequence = " ".join(
                        result.split()[-processor.sequence_length:])
                    break

        return result
    except Exception as e:
        print(f"Error generating text for seed '{seed_text}': {str(e)}")
        return None

def main():
    # Use prompts.txt as default if no file provided
    prompts_file = sys.argv[1] if len(sys.argv) > 1 else 'prompts.txt'

    # Generation parameters
    params = {
        'num_words': 50,
        'temperature': 2.0,  # Higher = more random, 0.0 to disable
        'top_k': 0,          # Higher = more likely, 0 to disable
        'top_p': 0.0         # Higher = more likely, 0.0 to disable
    }

    # Load model and processor
    try:
        model = load_model('text_generation_model.keras')
        processor = TextProcessor.load('text_processor.pkl')
        print("Model and processor loaded successfully!")
    except Exception as e:
        print(f"Error loading model or processor: {str(e)}")
        sys.exit(1)

    # Read and process prompts
    try:
        with open(prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading prompts file: {str(e)}")
        sys.exit(1)

    print(f"Found {len(prompts)} prompts to process")

    print("\nGeneration parameters:")
    print("-" * 20)
    for param, value in params.items():
        print(f"{param:12} = {value}")

    # Process each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: '{prompt}'")
        print("-" * 50)

        generated = generate_text(
            model=model,
            processor=processor,
            seed_text=prompt,
            **params
        )

        if generated:
            print(generated)
            print("-" * 50)


if __name__ == "__main__":
    main()