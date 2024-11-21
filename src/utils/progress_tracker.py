# src/utils/progress_tracker.py
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict


class ProgressTracker:
    """Track processing progress and handle persistence."""

    def __init__(self, tracking_file: Path):
        self.tracking_file = tracking_file
        self.processed_files = self._load_progress()

    def _load_progress(self) -> dict:
        """Load progress from tracking file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Corrupt progress file found. Starting fresh.")
                return {}
        return {}

    def save_progress(self):
        """Save current progress to tracking file."""
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f,
                          indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Could not save progress: {type(e).__name__}: {str(e)}")

    def update_file_progress(self,
                             file_path: Path,
                             completed: bool = False,
                             output_path: Optional[Path] = None,
                             stage: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Update progress for a specific file.

        Args:
            file_path: Path to the file being processed
            completed: Whether processing is complete
            output_path: Path where output was saved (if any)
            stage: Processing stage (e.g., 'tokenizer_fitting', 'training')
            metadata: Additional metadata to store
        """
        try:
            file_key = str(file_path)
            if file_key not in self.processed_files:
                self.processed_files[file_key] = {
                    'started_at': datetime.now().isoformat(),
                    'completed': False,
                    'stages': {},
                }

            if stage:
                if 'stages' not in self.processed_files[file_key]:
                    self.processed_files[file_key]['stages'] = {}

                self.processed_files[file_key]['stages'][stage] = {
                    'completed': completed,
                    'timestamp': datetime.now().isoformat()
                }

            if completed:
                self.processed_files[file_key]['completed'] = True
                self.processed_files[file_key]['completed_at'] = datetime.now(
                ).isoformat()

            if output_path:
                self.processed_files[file_key]['output_path'] = str(
                    output_path)

            if metadata:
                if 'metadata' not in self.processed_files[file_key]:
                    self.processed_files[file_key]['metadata'] = {}
                self.processed_files[file_key]['metadata'].update(metadata)

            self.save_progress()
        except Exception as e:
            print(
                f"Could not update progress for {file_path}: {type(e).__name__}: {str(e)}")

    def is_completed(self, file_path: Path, stage: Optional[str] = None) -> bool:
        """Check if a file has been completely processed.

        Args:
            file_path: Path to check
            stage: Specific stage to check (if None, checks overall completion)
        """
        file_info = self.processed_files.get(str(file_path))
        if not file_info:
            return False

        if stage:
            stages = file_info.get('stages', {})
            stage_info = stages.get(stage, {})
            return stage_info.get('completed', False)

        if file_info.get('completed', False):
            output_path = file_info.get('output_path')
            if output_path:
                return Path(output_path).exists()
        return False

    def get_remaining_files(self, all_files: list[Path], stage: Optional[str] = None) -> list[Path]:
        """Get list of files that still need processing.

        Args:
            all_files: List of all files to process
            stage: Specific stage to check (if None, checks overall completion)
        """
        return [f for f in all_files if not self.is_completed(f, stage)]

    def get_stage_metadata(self, file_path: Path, stage: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific processing stage."""
        file_info = self.processed_files.get(str(file_path))
        if not file_info:
            return None

        stages = file_info.get('stages', {})
        stage_info = stages.get(stage, {})
        return stage_info.get('metadata')
