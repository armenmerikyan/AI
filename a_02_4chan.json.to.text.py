import os
import json
import re
import html

# Function to remove HTML tags, links, and decode HTML entities
def clean_html(content):
    # Decode HTML entities (e.g., &gt; to >)
    decoded_content = html.unescape(content)
    
    # Remove HTML tags using regular expression
    clean_content = re.sub(r'<.*?>', '', decoded_content)
    
    # Remove links (URLs)
    clean_content = re.sub(r'http\S+|www\S+', '', clean_content)
    
    return clean_content.strip()

# Function to extract 'com' from JSON and save to a .txt file
def extract_post_content(json_directory):
    # Loop through all files in the directory
    for filename in os.listdir(json_directory):
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            json_file_path = os.path.join(json_directory, filename)
            
            # Open and load the JSON data
            with open(json_file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    
                    # Check if 'posts' key exists
                    if 'posts' in data:
                        for post in data['posts']:
                            # Extract the 'com' field from the post
                            post_content = post.get('com', None)
                            
                            # If 'com' is found, clean the content and create a .txt file
                            if post_content:
                                # Clean the HTML tags, links, and decode HTML entities
                                cleaned_content = clean_html(post_content)
                                
                                # Prepare the filename for the .txt file
                                txt_filename = f"{post['no']}.txt"
                                txt_file_path = os.path.join(json_directory, txt_filename)
                                
                                # Write the cleaned post content into the .txt file
                                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                                    txt_file.write(cleaned_content)
                                    print(f"Created: {txt_filename}")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {json_file_path}")
                except Exception as e:
                    print(f"Error processing file {json_file_path}: {e}")

# Specify the directory containing the JSON files
json_directory = 'gh_training_data'

# Call the function to process the files
extract_post_content(json_directory)
