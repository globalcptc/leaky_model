import os
import re
from typing import Optional, Iterator
from tqdm import tqdm

class TextFileReader:
    """A class to handle reading text and markdown files from a directory efficiently"""

    def __init__(self, directory_path: str, file_type: str = "text"):
        self.directory_path = directory_path
        self.file_type = file_type.lower()
        if self.file_type not in ["text", "markdown"]:
            raise ValueError('file_type must be either "text" or "markdown"')
        self._files = None

    @property
    def files(self) -> list[str]:
        """Lazy loading of file list"""
        if self._files is None:
            extension = ".md" if self.file_type == "markdown" else ".txt"
            self._files = [
                f for f in os.listdir(self.directory_path)
                if f.endswith(extension) and os.path.isfile(os.path.join(self.directory_path, f))
            ]
        return self._files

    def clean_markdown(self, content: str) -> str:
        """Clean markdown content to extract meaningful text while preserving structure."""
        # Remove code blocks
        content = re.sub(r'```[\s\S]*?```', '', content)
        # Remove inline code
        content = re.sub(r'`[^`]*`', '', content)
        # Remove images
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        # Remove links but keep link text
        content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', content)
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        # Remove markdown tables
        content = re.sub(r'^\|.*\|$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\|-+\|$', '', content, flags=re.MULTILINE)
        # Convert headers to plain text
        content = re.sub(r'^#+\s*(.*?)$', r'\1', content, flags=re.MULTILINE)
        # Remove bold and italic markers
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
        content = re.sub(r'\*(.*?)\*', r'\1', content)
        content = re.sub(r'_(.*?)_', r'\1', content)
        # Remove horizontal rules
        content = re.sub(r'^\s*[-*_]{3,}\s*$', '', content, flags=re.MULTILINE)
        # Remove multiple newlines and spaces
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        return content.strip()

    def read_file(self, filename: str) -> Optional[str]:
        """Read and clean a specific file"""
        if filename not in self.files:
            raise ValueError(f"File {filename} not found in directory")

        file_path = os.path.join(self.directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            if self.file_type == "markdown":
                return self.clean_markdown(content)
            return content

        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            return None

    def iter_files(self) -> Iterator[tuple[str, str]]:
        """Generator that yields (filename, content) pairs"""
        for filename in tqdm(self.files, desc="Reading files", dynamic_ncols=True, position=0, leave=True):
            content = self.read_file(filename)
            if content is not None:
                yield filename, content

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """Makes the class iterable"""
        return self.iter_files()
