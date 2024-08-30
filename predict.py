import torch
from tokenizer import SimpleTokenizer
from model import TransformerModel
from train_model import load_data

def predict_string(model, text, tokenizer, max_length, device):
    model.eval()
    
    # Tokenize and prepare the input
    tokens = tokenizer.encode(text)
    padded_tokens = tokenizer.pad_sequences([tokens], max_length)[0]
    input_tensor = torch.tensor(padded_tokens).unsqueeze(0).to(device)  # Add batch dimension
    
    # Sentiment Prediction
    with torch.no_grad():
        sentiment_output = model(input_tensor, task="sentiment")
        sentiment_pred = torch.argmax(sentiment_output, dim=1).item()
    
    # Text Classification Prediction
    with torch.no_grad():
        classification_output = model(input_tensor, task="classification")
        classification_pred = torch.argmax(classification_output, dim=1).item()

    # Convert predictions to human-readable labels
    sentiment_label = "positive" if sentiment_pred == 1 else "negative"
    classification_label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    classification_label = classification_label_map[classification_pred]
    
    # Print the input and predictions
    print(f"Input String: {text}")
    print(f"Predicted Sentiment: {sentiment_label}")
    print(f"Predicted Classification: {classification_label}")

if __name__ == "__main__":
    # Load and prepare the model
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tokenizer = SimpleTokenizer()
    model = TransformerModel(vocab_size=len(tokenizer.vocab)+2, d_model=64, n_heads=8, d_ff=256, n_layers=4, num_classes_sentiment=2, num_classes_classification=4)
    model.load_state_dict(torch.load('transformer_model.pth'))
    model.to(device)
    
    # Build the tokenizer vocabulary with raw text
    imdb_file = '/Users/shawyan/Desktop/Data Portfolio/Transformer-text-analysis/data/IMDB.csv'
    agnews_train_file = '/Users/shawyan/Desktop/Data Portfolio/Transformer-text-analysis/data/AGNews/train.csv'  # Replace with actual path to AG News train dataset
    agnews_test_file = '/Users/shawyan/Desktop/Data Portfolio/Transformer-text-analysis/data/AGNews/test.csv'
    imdb_dataset, _, _ = load_data(imdb_file, agnews_train_file, agnews_test_file, tokenizer, max_length=50)
    
    # Extract raw texts to build the vocabulary
    raw_texts = imdb_dataset.texts  # Assuming texts are stored in imdb_dataset.texts
    tokenizer.build_vocab(raw_texts)

    # Example text to predict
    example_text = "The business man was mad."
    predict_string(model, example_text, tokenizer, max_length=50, device=device)
