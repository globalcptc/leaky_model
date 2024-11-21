import json
from datetime import datetime
from pathlib import Path

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

    def update_file_progress(self, pdf_path: Path, completed: bool = False,
                             output_path: Path = None):
        """Update progress for a specific file."""
        try:
            file_key = str(pdf_path)
            if file_key not in self.processed_files:
                self.processed_files[file_key] = {
                    'started_at': datetime.now().isoformat(),
                    'completed': False,
                    'output_path': str(output_path) if output_path else None
                }

            if completed:
                self.processed_files[file_key]['completed'] = True
                self.processed_files[file_key]['completed_at'] = datetime.now(
                ).isoformat()
                if output_path:
                    self.processed_files[file_key]['output_path'] = str(
                        output_path)

            self.save_progress()
        except Exception as e:
            print(
                f"Could not update progress for {pdf_path}: {type(e).__name__}: {str(e)}")

    def is_completed(self, pdf_path: Path) -> bool:
        """Check if a file has been completely processed."""
        file_info = self.processed_files.get(str(pdf_path))
        if file_info and file_info.get('completed', False):
            output_path = Path(file_info.get('output_path', ''))
            return output_path.exists()
        return False

    def get_remaining_files(self, all_files: list[Path]) -> list[Path]:
        """Get list of files that still need processing."""
        return [f for f in all_files if not self.is_completed(f)]
