import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import psutil
print(f"Total memory: {psutil.virtual_memory().total / (1024 ** 3)} GB")
print(f"Used memory: {psutil.virtual_memory().used / (1024 ** 3)} GB")
print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3)} GB")

print(torch.cuda.memory_allocated())  # For CUDA GPUs
print(torch.cuda.memory_reserved())   # For CUDA GPUs
# Load model and tokenizer on MPS if available
#device = torch.device("mps" if torch.has_mps else "cpu")

#tokenizer = AutoTokenizer.from_pretrained("gpt-neo")
#model = AutoModelForCausalLM.from_pretrained("gpt-neo").to(device)

print(torch.has_mps)

print(torch.__version__)

# Load and preprocess the dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    return df['text'].tolist()

# Tokenize and prepare the dataset for training
def tokenize_data(tokenizer, texts, max_length=128):
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings.")
    print(f"Tokenizing {len(texts)} texts...")
    
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True
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
def fine_tune_model(csv_path, model_name="EleutherAI/gpt-neo-1.3B", output_dir="./results"):
    # Load data, tokenize, and train as before
    texts = load_data(csv_path)
    
    if len(texts) < 2:
        print("Not enough samples to split. Using the entire dataset for training.")
        X_train, X_test = texts, []
    else:
        X_train, X_test = train_test_split(texts, test_size=0.1, random_state=42)
    

    device = torch.device("mps" if torch.has_mps else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    train_dataset = tokenize_data(tokenizer, X_train)
    test_dataset = tokenize_data(tokenizer, X_test) if X_test else None
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch" if test_dataset else "no",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    if test_dataset:
        trainer.evaluate()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model fine-tuned and saved to {output_dir}")

# Run fine-tuning
if __name__ == "__main__":
    csv_path = 'prepared_data.csv'
    fine_tune_model(csv_path)



