# src/preprocessing/text_cleaner.py
import re
import unicodedata


class TextCleaner:
    def clean_text(self, text: str) -> str:
        """Clean and normalize OCR text output."""
        if not text:
            return ""

        try:
            # First clean up line endings and remove any carriage returns
            text = text.replace('\r\n', '\n').replace('\r', '\n')

            # Split into lines and process each line
            lines = text.split('\n')
            cleaned_lines = []

            for line in lines:
                # Skip empty or garbage lines
                line = line.strip()
                if not line or len(line) <= 2:
                    continue

                # Remove lines that are mostly special characters
                char_ratio = sum(c.isalnum() or c.isspace()
                                 for c in line) / len(line)
                if char_ratio < 0.5:
                    continue

                # Clean up the line
                line = self._clean_line(line)

                # Only add non-empty lines
                if line.strip():
                    cleaned_lines.append(line)

            # Join lines with single newlines
            text = '\n'.join(cleaned_lines)

            # Remove Unicode control characters
            text = ''.join(
                c for c in text if unicodedata.category(c)[0] != 'C')

            # Normalize Unicode whitespace characters to regular spaces
            text = re.sub(r'\s+', ' ', text)

            return text.strip()

        except Exception as e:
            print(f"Text cleaning failed: {type(e).__name__}: {str(e)}")
            return text

    def _clean_line(self, line: str) -> str:
        """Clean a single line of text."""
        try:
            # Fix URLs
            url_fixes = [
                (r'https:f+', 'https://'),
                (r'http:f+', 'http://'),
                (r'www\.f+', 'www.'),
                (r'\.orgf', '.org/'),
                (r'\.comf', '.com/'),
                (r'\.eduf', '.edu/'),
                (r'\.govf', '.gov/'),
                (r'\.netf', '.net/')
            ]

            for pattern, replacement in url_fixes:
                line = re.sub(pattern, replacement, line)

            # Clean up whitespace
            return ' '.join(line.split())

        except Exception as e:
            print(f"Line cleaning failed: {type(e).__name__}: {str(e)}")
            return line
