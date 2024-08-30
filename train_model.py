import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from dataset import TextDataset

def train_model(model, train_loader, task, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, task=task)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader.dataset)

def evaluate_model(model, test_loader, task, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, task=task)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    loss = running_loss / len(test_loader.dataset)
    return accuracy, loss


def evaluate_bert(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy


def load_data(imdb_file, agnews_train_file, agnews_test_file, tokenizer, max_length):
    # Load IMDB dataset
    imdb_df = pd.read_csv(imdb_file)
    print(f"IMDB Dataset Loaded: {imdb_df.shape[0]} records")

    # Load AGNews dataset
    agnews_train_df = pd.read_csv(agnews_train_file)
    agnews_test_df = pd.read_csv(agnews_test_file)
    print(f"AGNews Train Dataset Loaded: {agnews_train_df.shape[0]} records")
    print(f"AGNews Test Dataset Loaded: {agnews_test_df.shape[0]} records")

    # Create datasets
    imdb_dataset = TextDataset(imdb_df['review'].tolist(), imdb_df['sentiment'].tolist(), tokenizer, max_length, task="sentiment")
    agnews_train_dataset = TextDataset(agnews_train_df['Description'].tolist(), agnews_train_df['Class Index'].tolist(), tokenizer, max_length, task="classification")
    agnews_test_dataset = TextDataset(agnews_test_df['Description'].tolist(), agnews_test_df['Class Index'].tolist(), tokenizer, max_length, task="classification")

    # Example to show the mapping
    print(f"IMDB dataset label mapping: {imdb_dataset.label_map}")
    print(f"AGNews dataset label mapping: {agnews_train_dataset.label_map}")

    return imdb_dataset, agnews_train_dataset, agnews_test_dataset

