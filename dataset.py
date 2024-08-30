# dataset.py
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, task="sentiment"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

        if task == "sentiment":
            self.label_map, self.reverse_label_map = self._create_label_map(labels)
        elif task == "classification":
            # Explicitly define the label mapping for AGNews
            self.label_map = {1: 0, 2: 1, 3: 2, 4: 3}  # AGNews dataset's original integer labels mapped to 0-based indices
            self.reverse_label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

        # Debug: Print the first few mappings
        print(f"Label mapping for {self.task}: {self.label_map}")
        print(f"Reverse label mapping for {self.task}: {self.reverse_label_map}")

    def _create_label_map(self, labels):
        unique_labels = sorted(set(labels))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        reverse_label_map = {idx: label for label, idx in label_map.items()}
        return label_map, reverse_label_map

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_text = self.tokenizer.encode(text)
        padded_text = self.tokenizer.pad_sequences([encoded_text], self.max_length)[0]
        return torch.tensor(padded_text), torch.tensor(self.label_map[label])

    def get_label_name(self, index):
        return self.reverse_label_map[index]
