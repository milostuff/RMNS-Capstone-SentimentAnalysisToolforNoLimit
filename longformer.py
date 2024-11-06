import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch

# Load the dataset
df = pd.read_csv("adira.csv", delimiter=';', on_bad_lines='skip')

# Clean the sentiment labels by removing extra characters
df['sentiment'] = df['sentiment'].str.strip("'")

# Filter relevant columns
df_filtered = df[['body', 'sentiment']].dropna()

# Split the data into training and validation sets (80% training, 20% validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_filtered['body'].tolist(), df_filtered['sentiment'].tolist(), test_size=0.2, random_state=42
)

# Load the Longformer tokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# Tokenize the training and validation texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# Map sentiment labels to integers
label_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
train_labels = [label_mapping[label] for label in train_labels]
val_labels = [label_mapping[label] for label in val_labels]

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Longformer model for sequence classification
model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=3)

# Set training arguments with improvements
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",           # Evaluate at the end of each epoch
    save_strategy="epoch",                 # Save at the end of each epoch
    per_device_train_batch_size=8,         # Increase batch size
    per_device_eval_batch_size=8,          # Increase batch size
    num_train_epochs=5,                    # Increase the number of training epochs
    learning_rate=2e-5,                    # Decrease the learning rate
    weight_decay=0.01,                     # Add weight decay for regularization
    logging_dir='./logs',
    logging_steps=50,                      # Log less frequently
    save_total_limit=2,                    # Limit the total number of saved checkpoints
    load_best_model_at_end=True,           # Load the best model at the end of training
    metric_for_best_model='eval_loss',     # Save based on the best eval loss
    greater_is_better=False,               # Lower eval loss is better
    lr_scheduler_type="cosine",            # Use cosine learning rate schedule
)

# Define Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Stop early if no improvement
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

print(results)

import numpy as np
from sklearn.metrics import accuracy_score

# After training, get predictions on the validation dataset
predictions = trainer.predict(val_dataset)

# Get the predicted logits
logits = predictions.predictions

# Convert logits to predicted class labels
predicted_labels = np.argmax(logits, axis=1)

# Get the true labels
true_labels = val_labels

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.4f}")
