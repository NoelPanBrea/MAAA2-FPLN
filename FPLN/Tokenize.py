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

def feature_extraction(text: str) -> np.matrix:
    pretokenized = space_tokenize(text)
    input_matrix  = np.zeros((len(pretokenized), 4))
    label_matrix = np.zeros((len(pretokenized), 1))
    for i, e  in enumerate(pretokenized):
        # asignamos valores feature_vector(char, is_punct, is_num, next_char)
        feature_vector = np.zeros(4)
        feature_vector[0] = e
        feature_vector[3] = pretokenized[i+1]
        if np.isin(e, list(ascii_letters + digits), invert=True):      
            feature_vector[0] = 1
        if e.isnumeric():
            feature_vector[1] = 1
        input_matrix[i, :] = feature_vector
        # asignamos valor label (0=char no terminal, 1=char terminal)
        if feature_vector[3] == 0:
            label_matrix[i][0] = 1 
    
def classification_tokenize(text: str) -> list[str]:
    data_matrix = feature_extraction(text)