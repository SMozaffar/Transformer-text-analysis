# Transformer-Based NLI for Sentiment Analysis and Text Classification

## Overview

An NLP project that implements a Transformer model from scratch, designed to tackle two fundamental tasks:
1. **Sentiment Analysis**: Analyzing the sentiment (positive or negative) of a given piece of text.
2. **Text Classification**: Categorizing text into predefined classes.

This project is built on top of two popular datasets:
- **IMDB Reviews**: A dataset of 50,000 movie reviews, each labeled as either positive or negative.
- **AGNews**: A dataset comprising 120,000 news articles categorized into four classes: World, Sports, Business, and Sci/Tech.

The project serves as a practical demonstration of how to construct and train a Transformer model from the ground up, offering insights into the inner workings of the model, such as custom tokenization, data handling, and explicit label mapping. Additionally, the performance of the custom Transformer model is compared against a pretrained BERT model to evaluate how well it stacks up against a widely recognized state-of-the-art model.

## Background

Transformers have revolutionized NLP by enabling models to capture long-range dependencies and contextual information more effectively than traditional RNN-based architectures. This project offers a hands-on understanding of Transformers by building one from scratch and applying it to real-world tasks like sentiment analysis and text classification.

## Features

- **Custom Transformer Implementation**: The model is built from scratch, without relying on pre-built Transformer libraries, offering a deeper understanding of the architecture.
- **Dual-Task Learning**: Capable of performing both sentiment analysis and text classification.
- **Custom Tokenization**: Manual tokenization and padding to demonstrate data preprocessing steps.
- **Model Comparison**: Performance evaluation and comparison with a pretrained BERT model.

## Results and Discussion

This section discusses the performance of the custom Transformer model compared to a pretrained BERT model. Key metrics such as accuracy and loss are evaluated on both the IMDB and AGNews datasets. The comparison highlights the strengths and weaknesses of building a model from scratch versus using a pretrained model that has been fine-tuned on large datasets.

## Potential Improvements

While the project successfully demonstrates the construction and training of a Transformer model, there are several areas where further enhancements could be explored:

1. **Data Augmentation**: To improve generalization, additional data augmentation techniques could be applied during training.
2. **Hyperparameter Tuning**: Further experimentation with different hyperparameters, such as learning rates, batch sizes, and the number of Transformer layers, could optimize model performance.
3. **Model Regularization**: Introducing regularization techniques like dropout, weight decay, or layer normalization could help prevent overfitting.
4. **Advanced Architectures**: Experimenting with more advanced Transformer architectures, such as BERT or GPT, could provide insights into how different configurations affect performance.

