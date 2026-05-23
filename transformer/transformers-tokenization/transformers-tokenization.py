from typing import List, Dict

class SimpleTokenizer:

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.vocab = []

    def build_vocab(self, texts: List[str]) -> None:

        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab.append(word)

        self.vocab.sort()

        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }

        self.id_to_word = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.bos_token,
            3: self.eos_token
        }

        count = 4

        for word in self.vocab:
            self.word_to_id[word.lower()] = count
            self.id_to_word[count] = word.lower()
            count += 1

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:

        return [
            self.word_to_id.get(word, 1)
            for word in text.lower().split()
        ]

    def decode(self, ids: List[int]) -> str:

        return " ".join(
            self.id_to_word.get(i, self.unk_token)
            for i in ids
        )