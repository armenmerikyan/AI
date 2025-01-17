import os
import re

# Path to the directory containing the text files
directory = 'gh_training_data'

# Function to clean up text by keeping only letters, periods, and spaces
#def clean_text(text):
#    return re.sub(r'[^a-zA-Z\s.]', '', text)

def clean_text(text):
    return re.sub(r"[^a-zA-Z\s.'`]", '', text)

# Process each .txt file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Clean the content
        cleaned_content = clean_text(content)
        
        # Write the cleaned content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        
        print(f"Processed file: {filename}")
