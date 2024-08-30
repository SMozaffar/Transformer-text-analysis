from collections import defaultdict

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.reverse_vocab = {0: "[PAD]", 1: "[UNK]", 2: "[CLS]", 3: "[SEP]"}
        self.word_count = defaultdict(int)

    def build_vocab(self, texts):
        index = 4  # Start after special tokens
        for text in texts:
            for word in text.split():
                self.word_count[word] += 1
                if word not in self.vocab:
                    self.vocab[word] = index
                    self.reverse_vocab[index] = word
                    index += 1

    def encode(self, text):
        tokens = [self.vocab.get("[CLS]")]
        tokens += [self.vocab.get(word, 1) for word in text.split()]
        tokens.append(self.vocab.get("[SEP]"))
        return tokens

    def decode(self, tokens):
        return ' '.join([self.reverse_vocab.get(token, '[UNK]') for token in tokens])

    def pad_sequences(self, sequences, max_length):
        return [
            seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length]
            for seq in sequences
        ]
