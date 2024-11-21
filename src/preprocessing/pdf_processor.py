# src/preprocessing/pdf_processor.py
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import multiprocessing
from pathlib import Path
import os
import fitz
import pytesseract
from PIL import Image
import io
import tempfile
import numpy as np
from tqdm import tqdm
import cv2
import json
from datetime import datetime
import re
import unicodedata
from ..utils.graceful_killer import GracefulKiller
from ..utils.progress_tracker import ProgressTracker
from .image_enhancer import ImageEnhancer
from .text_cleaner import TextCleaner


class PDFProcessor:
    def __init__(self, input_dir: str, output_dir: str, temp_dir: str = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir) if temp_dir else Path(
            tempfile.gettempdir()) / "pdf_ocr"
        self.killer = GracefulKiller()
        self._worker_bars = {}  # Store progress bars for workers

        # Initialize helpers
        self.image_enhancer = ImageEnhancer()
        self.text_cleaner = TextCleaner()

        # Create directories if they don't exist
        for directory in [self.input_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.progress_file = self.output_dir / '.processing_progress.json'
        self.progress_tracker = ProgressTracker(self.progress_file)

        # Optimize tesseract configuration
        self.tesseract_config = '--oem 1 --psm 6 -l eng'

    def process_page(self, page_pixmap: bytes) -> str:
        """Process a single page with improved error handling."""
        if not page_pixmap:
            return ""

        try:
            # Process the image using ImageEnhancer
            processed_img = self.image_enhancer.enhance_image(page_pixmap)

            # Perform OCR
            text = pytesseract.image_to_string(
                processed_img,
                config=self.tesseract_config
            )

            # Clean the text
            return self.text_cleaner.clean_text(text)

        except Exception as e:
            # More informative error message
            error_msg = f"Page processing failed: {type(e).__name__}: {str(e)}"
            print(error_msg)
            return ""

    def update_progress(self, worker_id: int, increment: int = 1):
        """Update progress for a specific worker's progress bar."""
        # This will be called from process_pdf to update the worker's progress bar
        if hasattr(self, '_worker_bars') and worker_id in self._worker_bars:
            self._worker_bars[worker_id].update(increment)

    def process_pdf(self, pdf_path: Path, worker_id: int = None) -> str:
        """Process a single PDF file with improved error handling."""
        if self.killer.kill_now:
            return None

        try:
            doc = fitz.open(str(pdf_path))
            content = [f"# {pdf_path.stem}\n\n"]
            total_pages = len(doc)

            # Process each page
            for page_num in range(total_pages):
                if self.killer.kill_now:
                    doc.close()
                    return None

                try:
                    page = doc[page_num]
                    text = page.get_text().strip()

                    if not text:
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                        img_data = pix.tobytes("png")
                        text = self.process_page(img_data)

                    if text:
                        if content[-1] != "\n":
                            content.append("\n")
                        content.append(text.strip())
                        content.append("\n---\n")

                    # Update progress if worker_id provided
                    if worker_id is not None:
                        self.update_progress(worker_id)

                except Exception as e:
                    print(
                        f"Error processing page {page_num} of {pdf_path}: {type(e).__name__}: {str(e)}"
                    )
                    continue

            doc.close()

            # Only save if we haven't been interrupted
            if not self.killer.kill_now:
                final_content = ''.join(content)
                final_content = re.sub(r'\n{3,}', '\n\n', final_content)
                final_content = final_content.strip() + '\n'

                output_path = self.output_dir / f"{pdf_path.stem}.md"
                output_path.write_text(
                    final_content, encoding='utf-8', errors='replace')

                self.progress_tracker.update_file_progress(
                    pdf_path, completed=True, output_path=output_path)

                return final_content

        except Exception as e:
            print(
                f"Error processing PDF {pdf_path}: {type(e).__name__}: {str(e)}")

        return None

    def process_files(self):
        """Process multiple PDF files using thread pool with graceful shutdown."""
        pdf_files = list(self.input_dir.glob('*.pdf'))
        if not pdf_files:
            print(f"No PDF files found in {self.input_dir}")
            return

        remaining_files = self.progress_tracker.get_remaining_files(pdf_files)
        total_files = len(pdf_files)
        remaining_count = len(remaining_files)

        print(f"\nFound {total_files} PDF files")
        print(f"Previously completed: {total_files - remaining_count}")
        print(f"Remaining to process: {remaining_count}")

        if not remaining_files:
            print("All files have been processed!")
            return

        max_workers = min(multiprocessing.cpu_count()
                          * 2, len(remaining_files))

        # Create progress bars for each worker
        self._worker_bars = {
            i: tqdm(total=0,  # Will be updated when task assigned
                    desc=f"Worker {i+1}",
                    position=i+1,
                    leave=True,
                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
            for i in range(max_workers)
        }

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Initialize the main progress bar above worker bars
                with tqdm(total=len(remaining_files),
                          desc="Overall Progress",
                          position=0,
                          leave=True,
                          bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}') as pbar:

                    # Keep track of worker assignments
                    worker_assignments = {}
                    available_workers = list(range(max_workers))

                    # Process files in chunks to maintain worker assignment
                    remaining = remaining_files.copy()
                    while remaining or worker_assignments:
                        # Submit new tasks if workers are available and files remain
                        while available_workers and remaining:
                            worker_id = available_workers.pop(0)
                            pdf_path = remaining.pop(0)

                            # Update progress bar for this worker
                            self._worker_bars[worker_id].reset()
                            doc = fitz.open(str(pdf_path))
                            total_pages = len(doc)
                            doc.close()

                            self._worker_bars[worker_id].total = total_pages
                            self._worker_bars[worker_id].desc = f"Worker {worker_id+1}: {pdf_path.name}"
                            self._worker_bars[worker_id].refresh()

                            # Submit the task
                            future = executor.submit(
                                self.process_pdf, pdf_path, worker_id)
                            worker_assignments[future] = (worker_id, pdf_path)

                        # Check completed tasks
                        done, _ = concurrent.futures.wait(
                            worker_assignments.keys(),
                            timeout=0.1,
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )

                        for future in done:
                            worker_id, pdf_path = worker_assignments[future]
                            try:
                                future.result()
                            except Exception as e:
                                print(
                                    f"\nError processing {pdf_path}: {type(e).__name__}: {str(e)}")
                            finally:
                                # Update progress and release worker
                                pbar.update(1)
                                available_workers.append(worker_id)
                                del worker_assignments[future]

                        if self.killer.kill_now:
                            print("\nCancelling remaining tasks...")
                            for f in worker_assignments:
                                f.cancel()
                            break

        except KeyboardInterrupt:
            print("\nShutdown requested... Cleaning up...")
        finally:
            print("\nSaving progress...")
            self.progress_tracker.save_progress()

            # Clean up progress bars
            for bar in self._worker_bars.values():
                bar.close()
            self._worker_bars.clear()
