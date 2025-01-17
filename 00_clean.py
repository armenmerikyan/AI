import os
import re

# Regular expression to detect valid English sentences
sentence_pattern = re.compile(r"^[A-Z][^.?!]*[.?!]$")

def is_valid_sentence(sentence):
    """Check if a sentence is valid based on the regular expression pattern."""
    return bool(sentence_pattern.match(sentence.strip()))

def process_txt_files(directory_path):
    """Process each .txt file in the directory, keeping only valid sentences."""
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Filter out only valid sentences
            valid_sentences = [line.strip() for line in lines if is_valid_sentence(line)]

            # If valid sentences are found, overwrite the file with them
            if valid_sentences:
                with open(file_path, 'w') as file:
                    for sentence in valid_sentences:
                        file.write(sentence + '\n')
            else:
                # Delete the file if no valid sentences are left
                os.remove(file_path)
                print(f"Deleted empty file: {file_path}")

# Directory containing the .txt files
directory_path = "gh_training_data"
process_txt_files(directory_path)
