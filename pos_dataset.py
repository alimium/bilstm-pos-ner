import torch
from torch.utils.data import Dataset


class POSDataset(Dataset):
    def __init__(
        self,
        sentences: list,
        tags: list,
        word2idx: dict[str, int],
        tag2idx: dict[str, int],
        max_len: int,
    ):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

        self.sentences = self.tokenize_and_pad(sentences, self.word2idx, self.max_len)
        self.tags = self.tokenize_and_pad(tags, self.tag2idx, self.max_len)

    def tokenize_and_pad(self, sequences, vocab, max_len):
        processed_sequences = []
        for seq in sequences:
            seq = seq.split(" ")
            seq = [vocab.get(word, vocab["UNK"]) for word in seq]
            seq = seq + [vocab["PAD"]] * (max_len - len(seq))
            processed_sequences.append(seq)
        return processed_sequences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx], dtype=torch.long), torch.tensor(
            self.tags[idx], dtype=torch.long
        )
