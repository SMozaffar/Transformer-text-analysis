from collections import defaultdict

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.word_count = defaultdict(int)

    def build_vocab(self, texts):
        index = 2  # 0 reserved for padding, 1 for unknown tokens
        for text in texts:
            for word in text.split():
                self.word_count[word] += 1
                if word not in self.vocab:
                    self.vocab[word] = index
                    self.reverse_vocab[index] = word
                    index += 1

    def encode(self, text):
        return [self.vocab.get(word, 1) for word in text.split()]

    def decode(self, tokens):
        return ' '.join([self.reverse_vocab.get(token, '[UNK]') for token in tokens])

    def pad_sequences(self, sequences, max_length):
        return [seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in sequences]
