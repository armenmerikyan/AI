import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

topics = [
    "Role-playing and fantasy exploration",
    "Sensory play (temperature, texture, etc.)",
    "Power dynamics (dominance and submission)",
    "Bondage and restraint techniques",
    "Impact of trust and communication in kink",
    "Exploring fetishes and personal interests",
    "Erotic massage and touch techniques",
    "Using toys and accessories for intimacy",
    "Mindfulness and sensual awareness",
    "Costumes and dressing up",
    "Erotic storytelling and fantasy writing",
    "Edging and delayed gratification",
    "Impact of kinks on relationships and intimacy",
    "Aftercare and emotional support",
    "Building trust and safe boundaries in kink"
]

topics = [
    "give me a tip for good sex"
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah", "Ivy", "Jack",
    "Kathy", "Liam", "Mona", "Nina", "Oscar", "Paul", "Quinn", "Rachel", "Sam", "Tina",
    "Uma", "Vince", "Walter", "Xander", "Yara", "Zane", "Amelia", "Benjamin", "Chloe",
    "Derek", "Emma", "Felix", "Gina", "Holly", "Ian", "Jade", "Kevin", "Laura", "Mason",
    "Nora", "Oliver", "Penny", "Quincy", "Rita", "Steve", "Terry", "Ursula", "Vera",
    "Wendy", "Ximena", "Yasmine", "Zachary"
]
 


response_output = "model_output_hf"

# Load the fine-tuned model and tokenizer
def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return tokenizer, model

# Generate text using the model
def generate_text(prompt, tokenizer, model, max_length=100, num_return_sequences=1, repetition_penalty=1.2):
    # Encode the input prompt with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
    
    # Get the end-of-sequence token ID (if available in the model)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else None

    # Generate text with adjusted settings for more natural sentence generation
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        temperature=0.7,  # Slightly higher for more variety
        top_k=50,         # Increase top_k for more token choices
        top_p=0.9,        # Nucleus sampling for more diversity
        do_sample=True,   # Enable sampling for diversity
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode and clean up the generated text, removing line breaks and unnecessary whitespace
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True).replace("\n", " ").strip() for output in outputs]
    
    # Optionally, apply a post-processing step to add periods or check for incomplete sentences
    # For example, you can append a period if one is missing
    final_texts = []
    for text in generated_texts:
        if not text.endswith('.'):
            text += '.'
        final_texts.append(text)
    
    return final_texts



index = 0

while index < len(topics):
# Main function to run the text generation
    model_dir = './results'  # Path to your saved model directory
    tokenizer, model = load_model_and_tokenizer(model_dir)

 

    prompt = "You are a friendly and helpful assistant. " + topics[index]    # Get the prompt from the command line
    
    # Generate text (you can change num_return_sequences to test both options)
    generated_texts = generate_text(prompt, tokenizer, model, max_length=80, num_return_sequences=1)  # Change to 1 for greedy

    # Print the generated texts
    #for idx, text in enumerate(generated_texts):
    #    print(f"Generated Text {idx + 1}: {text}")

    print(f"Generated Text: {generated_texts[0]}")

    filename = topics[index].replace(" ", "_")
    output_file_path = os.path.join(response_output, f"output_{filename}.txt")
    with open(output_file_path, "w") as f:
        f.write(generated_texts[0])

    index +=1