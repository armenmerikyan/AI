import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load and preprocess the dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    return df['text'].tolist()  # Ensure it's a list of strings
 
# Tokenize and prepare the dataset for training
# Tokenize and prepare the dataset for training
def tokenize_data(tokenizer, texts, max_length=128):
    # Ensure `texts` is a list of strings
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings.")
    print(f"Tokenizing {len(texts)} texts...")  # Debugging output
    
    # Tokenize the texts using the tokenizer's batch_encode_plus method
    encodings = tokenizer(
        texts,  # Pass the list of strings
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True  # Ensure special tokens are added
    )
    return TextDataset(encodings)


# Custom Dataset class for PyTorch
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

# Fine-tune the model
def fine_tune_model(csv_path, model_name="gpt2", output_dir="./results"):
    # Load data
    texts = load_data(csv_path)
    
    # Split data into training and validation sets
    if len(texts) < 2:
        print("Not enough samples to split. Using the entire dataset for training.")
        X_train, X_test = texts, []
    else:
        X_train, X_test = train_test_split(texts, test_size=0.1, random_state=42)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    # Tokenize data
    train_dataset = tokenize_data(tokenizer, X_train)
    test_dataset = tokenize_data(tokenizer, X_test) if X_test else None
    
    # Data collator with dynamic padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch" if test_dataset else "no",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Train and evaluate the model
    trainer.train()
    if test_dataset:
        trainer.evaluate()
    
    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved to {output_dir}")

# Run fine-tuning
if __name__ == "__main__":
    # Path to your CSV file with text data
    csv_path = 'prepared_data.csv'
    
    # Call the fine-tuning function
    fine_tune_model(csv_path)
