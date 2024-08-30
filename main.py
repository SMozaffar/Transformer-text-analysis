# main.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizer import SimpleTokenizer
from model import TransformerModel
from train_model import train_model, evaluate_model, load_data, evaluate_bert
#from transformers import BertTokenizer, BertForSequenceClassification

def main():
    # Configuration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    d_model = 64
    n_heads = 8
    d_ff = 256
    n_layers = 4
    num_classes_sentiment = 2  # Positive/Negative
    num_classes_classification = 4  # AG News classes
    batch_size = 128
    num_epochs = 10
    max_length = 50

    # Load Data
    imdb_file = '/Users/shawyan/Desktop/Data Portfolio/Transformer-text-analysis/data/IMDB.csv'  # Replace with actual path to IMDB dataset
    agnews_train_file = '/Users/shawyan/Desktop/Data Portfolio/Transformer-text-analysis/data/AGNews/train.csv'  # Replace with actual path to AG News train dataset
    agnews_test_file = '/Users/shawyan/Desktop/Data Portfolio/Transformer-text-analysis/data/AGNews/test.csv'  # Replace with actual path to AG News test dataset

    tokenizer = SimpleTokenizer()
    imdb_dataset, agnews_train_dataset, agnews_test_dataset = load_data(imdb_file, agnews_train_file, agnews_test_file, tokenizer, max_length)

    # Split IMDB dataset into train and test
    train_size = int(0.8 * len(imdb_dataset))
    test_size = len(imdb_dataset) - train_size
    imdb_train_dataset, imdb_test_dataset = torch.utils.data.random_split(imdb_dataset, [train_size, test_size])

    # DataLoaders
    imdb_train_loader = DataLoader(imdb_train_dataset, batch_size=batch_size, shuffle=True)
    imdb_test_loader = DataLoader(imdb_test_dataset, batch_size=batch_size, shuffle=False)
    agnews_train_loader = DataLoader(agnews_train_dataset, batch_size=batch_size, shuffle=True)
    agnews_test_loader = DataLoader(agnews_test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    vocab_size = len(tokenizer.vocab) + 2  # Including padding and unknown tokens
    model = TransformerModel(vocab_size, d_model, n_heads, d_ff, n_layers, num_classes_sentiment, num_classes_classification)
    model.to(device)

    # Loss and Optimizer
    sentiment_criterion = nn.CrossEntropyLoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    for epoch in range(num_epochs):
        sentiment_loss = train_model(model, imdb_train_loader, "sentiment", sentiment_criterion, optimizer, device)
        classification_loss = train_model(model, agnews_train_loader, "classification", classification_criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Sentiment Loss: {sentiment_loss:.4f}, Classification Loss: {classification_loss:.4f}")

    # Evaluation
    sentiment_accuracy, sentiment_test_loss = evaluate_model(model, imdb_test_loader, "sentiment", sentiment_criterion, device)
    classification_accuracy, classification_test_loss = evaluate_model(model, agnews_test_loader, "classification", classification_criterion, device)
    
    print(f"Sentiment Test Accuracy: {sentiment_accuracy:.2f}%, Test Loss: {sentiment_test_loss:.4f}")
    print(f"Classification Test Accuracy: {classification_accuracy:.2f}%, Test Loss: {classification_test_loss:.4f}")

    # Save the Model
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Model saved successfully.")

    # # Compare with Pretrained BERT Model
    # print("\n--- Comparing with Pretrained BERT Model ---")
    
    # # Load pretrained BERT model and tokenizer
    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # bert_model_sentiment = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    # bert_model_classification = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4).to(device)

    # # Evaluate BERT on IMDB dataset
    # imdb_accuracy = evaluate_bert(bert_model_sentiment, imdb_test_loader, device)
    # print(f"Pretrained BERT Model - Sentiment Analysis (IMDB) Accuracy: {imdb_accuracy * 100:.2f}%")

    # # Evaluate BERT on AGNews dataset
    # agnews_accuracy = evaluate_bert(bert_model_classification, agnews_test_loader, device)
    # print(f"Pretrained BERT Model - Text Classification (AGNews) Accuracy: {agnews_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
