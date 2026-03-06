import matplotlib.pyplot as plt
from sys import argv
from wordpiece import WordPiece, chain
from classifytokenize import ClassifyTokenize
from Tokenize import space_tokenize, mark_tokenize, ngram_tokenize
from bpe import Bpe

def test_def_tokenizer(tokenizer, test_lines: list[str], tokenizer_params = None):
    print("\n" + tokenizer.__name__)
    for line in test_lines:
        line = line.strip("\n")
        print("Input: ", line, end="->")
        print("Tokens: ", tokenizer(line) if tokenizer_params is None else tokenizer(line, tokenizer_params))

def test_class_tokenizer(tokenizer, test_lines: list[str]):
    print("\n" + str(tokenizer))
    for line in test_lines:
        line = line.strip("\n")
        print("Input: ", line, end="->")
        print("Tokens: ", tokenizer.tokenize(line))
    
def plot_test(text: list[str]):
    x, y = [], [[] for _ in range(6)]
    bins = range(500, len(text), 500)
    titles = ["space_tokenize", "mark_tokenize", "ngram_tokenize",
               "Bpe", "WordPiece", "ClassifyTokenize"]
    for i in bins:
        x.append(len(text[:i]))
        sent = "".join(text[:i]).strip("\n")

        tokenizer = Bpe()
        _, corp = tokenizer.train(sent, 3000)
        y[0].append(len(set(corp)))

        tokenizer = WordPiece()
        _, corp = tokenizer.train(sent, 3000)
        y[1].append(len(set(corp)))

        tokenizer = ClassifyTokenize()
        tokenizer.train(sent)
        corp = tokenizer.tokenize(sent)
        y[2].append(len(set(corp)))

        y[3].append(len(set(space_tokenize(sent))))
        y[4].append(len(set(mark_tokenize(sent))))
        y[5].append(len(set(ngram_tokenize(sent, 2))))

    for i, title in zip(y, titles): 
        plt.plot([0] + x, [0] + i, label = title)
    plt.xlabel("Nº de Oraciones")
    plt.ylabel("Nº de Tokens Únicos")
    plt.title("Comparación de Curvas")
    plt.legend()
    plt.show()

def main():
    if len(argv) != 2:
        print("Usage: python main.py path_to_files_folder")
        print("\nUsando rutas por defecto, FLPN/Files.txt")
        folder_path = "FPLN"
    else: folder_path = argv[1]
    with open(folder_path + "/training_sentences.txt", "r", encoding="utf-8") as file:
        train_lines = file.readlines()
    with open(folder_path + "/test_sentences.txt", "r", encoding="utf-8") as file:
        test_lines = file.readlines()
    with open(folder_path + "/majesty_speeches.txt", "r", encoding="utf-8") as file:
        long_text = file.readlines()
    train_text = "".join(train_lines)
    bpe = Bpe()
    wordpiece = WordPiece()
    classifytokenize = ClassifyTokenize()
    maxvoc = 150
    bpe.train(train_text, maxvoc)
    wordpiece.train(train_text, maxvoc)
    classifytokenize.train(train_text)

    test_lines = train_lines + test_lines
    test_def_tokenizer(space_tokenize, test_lines)
    test_def_tokenizer(mark_tokenize, test_lines)
    test_def_tokenizer(ngram_tokenize, test_lines, 2)

    test_class_tokenizer(bpe, test_lines)
    test_class_tokenizer(wordpiece, test_lines)
    test_class_tokenizer(classifytokenize, test_lines)

    plot_test(long_text)

if __name__ == "__main__":
    main()