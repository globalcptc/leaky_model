# src/training/model_builder.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense


class ModelBuilder:
    @staticmethod
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
