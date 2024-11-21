"""
Simple python script to demonstate usage with a single prompt
"""

import tensorflow as tf
import pickle
import numpy as np

model_file = "model/text_generation_model.keras"
processor_file = "model/text_processor.pkl"

# Load model and processor
model = tf.keras.models.load_model(model_file)
with open(processor_file, 'rb') as f:
    processor = pickle.load(f)

# Generation parameters
prompt = "we discovered the user"
max_tokens = 100
temperature = 1.7    # Higher = more random, Lower = more focused (default: 0.7)
top_k = 50          # Limit to top k tokens (set to 0 to disable)
top_p = 0.9         # Nucleus sampling threshold (set to 1.0 to disable)

# Process the prompt
tokenizer = processor['tokenizer']
sequence_length = processor['sequence_length']
current_sequence = tokenizer.texts_to_sequences([prompt])[0]
current_sequence = [0] * (sequence_length - len(current_sequence)) + current_sequence
current_sequence = np.array([current_sequence])

# Generate text
generated_text = prompt
for _ in range(max_tokens):
    pred = model.predict(current_sequence, verbose=0)
    logits = pred[0] / temperature

    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = np.argsort(logits)[:-top_k]
        logits[indices_to_remove] = -float('inf')

    # Apply top-p filtering (nucleus sampling)
    if top_p < 1.0:
        sorted_logits = np.sort(logits)[::-1]
        cumulative_probs = np.cumsum(tf.nn.softmax(sorted_logits))
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = False
        indices_to_remove = np.argsort(logits)[::-1][sorted_indices_to_remove]
        logits[indices_to_remove] = -float('inf')

    # Sample from the filtered distribution
    probs = tf.nn.softmax(logits).numpy()
    next_token = np.random.choice(len(probs), p=probs)

    # Get the word for this token
    for word, index in tokenizer.word_index.items():
        if index == next_token:
            generated_text += ' ' + word
            break

    # Update sequence
    current_sequence = np.array([current_sequence[0, 1:].tolist() + [next_token]])

print(generated_text)
