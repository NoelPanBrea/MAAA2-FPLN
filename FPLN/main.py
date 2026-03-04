import matplotlib.pyplot as plt
from wordpiece import WordPiece, chain
from classifytokenize import ClassifyTokenize
from Tokenize import space_tokenize, mark_tokenize, ngram_tokenize
from bpe import Bpe

def main():
    # with open("FPLN/training_sentences.txt", "r", encoding="utf-8") as file:
    #     text = file.read()
    # tokenizer = Bpe()
    # voc, corpus = tokenizer.train(text, 150)
    # print(len(voc))
    # with open("FPLN/test_sentences.txt", "r", encoding="utf-8") as file:
    #     text = file.readlines()
    # print(tokenizer.tokenize(text[2]))
    with open("FPLN/majesty_speeches.txt", "r", encoding="utf-8") as file:
        text = file.readlines()
    x, y = [], []
    for i in range(300, len(text), 300):
        tokenizer = Bpe()
        sent = list(chain.from_iterable(text[:i]))[0]
        print(sent)
        voc, _ = tokenizer.train(sent, 3000)
        x.append(len(sent))
        y.append(len(voc))
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()