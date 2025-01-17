import re
import unicodedata
import os 
import pandas as pd

def clean_text(text):
    # Remove non-UTF-8 characters (e.g., non-printable characters)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Cn')  # Cn means "Other, not assigned"
    
    # Normalize text to NFC form (composed form) to standardize accented characters
    text = unicodedata.normalize('NFC', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-word characters (keeping spaces and alphanumeric characters)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text

# Process each file in the directory
texts = []
labels = []  # If you have labels

directory_path = 'gh_training_data'

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)

    if filename.endswith(".txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

                # Clean the text
                cleaned_text = clean_text(text)
                texts.append(cleaned_text)

                # Extract label from the filename (optional)
                label = filename.split('_')[0]  # Adjust this logic as needed
                labels.append(label)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Create a DataFrame if you have labels
df = pd.DataFrame({
    'text': texts,
    'label': labels  # Uncomment if you have labels
})

# Save to CSV for model training
df.to_csv('prepared_data.csv', index=False)

print("Data has been processed and saved to 'prepared_data.csv'")
