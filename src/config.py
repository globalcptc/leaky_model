# src/config.py
import os
import multiprocessing
from pathlib import Path


class Config:
    """Global configuration settings for the project."""

    # Environment configuration
    OMP_THREAD_LIMIT = str(multiprocessing.cpu_count())
    PYTESSERACT_THREADS = str(multiprocessing.cpu_count())

    # Tesseract configuration
    TESSERACT_CONFIG = '--oem 1 --psm 6 -l eng'

    # Default paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'model'

    # Data directories
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    TEMP_DIR = DATA_DIR / 'tmp' / 'pdf_ocr'

    # Model files
    MODEL_FILE = MODEL_DIR / 'text_generation_model.keras'
    PROCESSOR_FILE = MODEL_DIR / 'text_processor.pkl'

    # Processing configuration
    PROGRESS_FILE = PROCESSED_DATA_DIR / '.processing_progress.json'

    # Training configuration
    DEFAULT_SEQUENCE_LENGTH = 50
    DEFAULT_EMBEDDING_DIM = 100
    DEFAULT_BATCH_SIZE = 128

    # Generation configuration
    DEFAULT_NUM_WORDS = 50
    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_TOP_K = 0
    DEFAULT_TOP_P = 0.0

    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.TEMP_DIR,
            cls.MODEL_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_env_variables(cls):
        """Set up environment variables for external tools."""
        return {
            'OMP_THREAD_LIMIT': cls.OMP_THREAD_LIMIT,
            'PYTESSERACT_THREADS': cls.PYTESSERACT_THREADS
        }

    @classmethod
    def setup_environment(cls):
        """Set up the environment with necessary variables."""
        env_vars = cls.get_env_variables()
        for key, value in env_vars.items():
            os.environ[key] = value
