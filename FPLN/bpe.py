from itertools import chain
from Tokenize import space_tokenize, ascii_letters, digits

class Bpe() :
    def __init__(self):
        self.voc = set(["[UNK]"])

    @property
    def vocab(self):
        return self.voc

    def train(self, text: str, max_vocab: int = 300) -> list[str]:
        pretokenized = space_tokenize(text)
        ascii_range = list(ascii_letters + digits + "".join([chr(i) for i in range(193, 255)]))
        corpus = []
        for i in pretokenized:
            word = []
            for j, e in enumerate(i):
                word.append(("##" + e if j > 0 else e) if e in ascii_range else "[UNK]")
            corpus.append(word)
        self.voc.update(chain.from_iterable(corpus))
        maxcnt = (True, True)
        while len(self.voc) < max_vocab and maxcnt[0]:
            paircnt = dict()
            maxcnt = (0, ("", ""))
            for word in corpus:
                for pair in zip(word, word[1:]):
                    paircnt[pair] = 1 + paircnt.get(pair, 0)
                    maxcnt = (paircnt[pair], pair) if paircnt[pair] > maxcnt[0] else maxcnt
            first, second = maxcnt[1]
            fusion = first + second.strip("#") if (first != "[UNK]" and second != "[UNK]") else "[UNK]"
            self.voc.add(fusion)
            for i, word in enumerate(corpus):
                j = 0
                while j < len(word) - 1:
                    if maxcnt[1] == (word[j], word[j + 1]):
                        word[j] = fusion
                        word.pop(j + 1)
                    j += 1
                corpus[i] = word
        return self.voc, list(chain.from_iterable(corpus))

    def tokenize(self, text: str):
        pretokenized = space_tokenize(text)
        for word in pretokenized:
            for j in range(len(word) - 1, 0, -1):
                head = word[:j]
                if head in self.voc:
                    word