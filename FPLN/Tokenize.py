import numpy as np
import re
from string import digits, ascii_letters, punctuation, whitespace

def space_tokenize(text: str) -> list[str]:
    return text.replace("\n", " ").strip().split(" ")

def mark_tokenize(text: str) -> list[str]:
    unique = np.unique(list(text))
    markers = unique[np.isin(unique, list(ascii_letters + digits + "".join([chr(i) for i in range(193, 255)])), invert=True)]
    return [i for i in re.split(r"([{}])".format(re.escape("".join(markers))), text) if i.strip()]

def ngram_tokenize(text: str, n: int = 1) -> list[str]:
    pretokenized = space_tokenize(text)
    return [" ".join(pretokenized[i:i + n]) for i, _ in enumerate(pretokenized)]
