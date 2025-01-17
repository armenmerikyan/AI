import os
import re

# Define the directory where the .txt files are located
directory = 'gh_training_data'

# Define a function to clean the text by removing leading numbers and special characters
def clean_text(text):
    # Remove leading numbers at the beginning of the content
    text = re.sub(r'^\d+', '', text)  # Remove any leading digits
    # Remove specific patterns such as '>>', '<', '>', and '?'
    text = re.sub(r'[<>?]+', '', text)
    return text

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Clean the content
        cleaned_content = clean_text(content)

        # Save the cleaned content back to the file (or to a new file)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)

        print(f"Cleaned file: {filename}")
