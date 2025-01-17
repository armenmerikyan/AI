import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import os
# Path to the directory where the model and tokenizer are saved
model_directory = "./results"


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

 

response_output = "model_output"

tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

# Set device for model inference
#device = torch.device("mps" if torch.has_mps else "cpu")
#model.to(device)

# Set device for model inference
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
model.to(device)


index = 0

def generate_text(prompt, max_new_tokens=100, temperature=0.7, repetition_penalty=1.5):

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate output
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=repetition_penalty
    )
    
    # Decode the output and return it as text
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Use a while loop to iterate through the string array
while index < len(topics):
    # Print the current string 

    # Load the tokenizer and model

    # Set device for model inference
    #device = torch.device("mps" if torch.has_mps else "cpu")
    #model.to(device)
    # Set device for model inference
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
    model.to(device)
    prompt = topics[index]            
    generated_text = generate_text(prompt)
    print("Generated Text:", generated_text)        
    filename = topics[index].replace(" ", "_")
    output_file_path = os.path.join(response_output, f"output_{filename}.txt")
    with open(output_file_path, "w") as f:
        f.write(generated_text)

    index +=1
