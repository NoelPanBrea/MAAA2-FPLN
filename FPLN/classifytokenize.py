import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from string import digits, punctuation

class ClassifyTokenize():
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def feature_extraction(self, text: str):
        features = []

        for i, character in enumerate(text):
            character_features = {}
            character_features[f"char={character}"] = 1
            character_features["is_punct"] = character in punctuation
            character_features["is_digit"] = character in digits
            if i < len(text) - 1:
                character_features[f"next_char={text[i+1]}"] = 1
            else:
                character_features["next_char=EOS"] = 1
            features.append(character_features)
    
        return features

    def label_extraction(self, text: str):
        labels = []

        for i in range(len(text)):
            if i == len(text) - 1:
                labels.append(1)
            elif text[i+1] == " ":
                labels.append(1)
            elif text[i+1] in punctuation:
                labels.append(1)
            else:
                labels.append(0)

        return labels
    
    def train_classification_tokenize(self, text):
        X_dict = self.feature_extraction(text)
        y = self.label_extraction(text)
    
        vectorizer = DictVectorizer(sparse=False)
        X = vectorizer.fit_transform(X_dict)

        logistic_model = LogisticRegression()
        logistic_model.fit(X, y)
        self.vectorizer = vectorizer
        self.model = logistic_model

        return logistic_model, vectorizer

    def test_classification_tokenize(self, sentence: str):
        X_dict = self.feature_extraction(sentence)
        X = self.vectorizer.transform(X_dict)
        predictions = self.model.predict(X)
    
        tokens = []
        current_token = ""
    
        for ch, end in zip(sentence, predictions):
            current_token += ch
            if end == 1:
                tokens.append(current_token)
                current_token = ""
    
        if current_token:
            tokens.append(current_token)
    
        return tokens

    def tokenize_text_by_lines(self, text: str):
        tokenized_sentences = []

        lines = text.split("\n")

        for line in lines:
            if line.strip():
                tokens = self.test_classification_tokenize(line)
                tokenized_sentences.append(tokens)

        return tokenized_sentences
   
           

def main():
    with open("training_sentences.txt", "r", encoding="utf-8") as file:
        train_text = file.read()
    with open("test_sentences.txt", "r", encoding="utf-8") as file:
        test_text = file.read()
    tokenizer = ClassifyTokenize()
    tokenizer.train_classification_tokenize(train_text)
    result = tokenizer.tokenize_text_by_lines(test_text)
    print(result)

if __name__ == "__main__":
    main()