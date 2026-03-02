from wordpiece import WordPiece
from classifytokenize import ClassifyTokenize

def main():
    with open("training_sentences.txt", "r", encoding="utf-8") as file:
        text = file.read()
    tokenizer = WordPiece()
    voc, corpus = tokenizer.train(text)
    print(corpus, len(voc))
    # with open("test_sentences.txt", "r", encoding="utf-8") as file:
    #     text = file.read()
    # print(tokenizer.tokenize(text, voc))

if __name__ == "__main__":
    main()