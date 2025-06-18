import re
from typing import List


class Tokenizer:

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def clean_text(self, text: str) -> str:
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        text = self.clean_text(text)
        if self.lowercase:
            text = text.lower()
        tokens = self.pattern.findall(text)
        return tokens
