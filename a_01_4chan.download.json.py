import os
import requests
import time
import json
# List of boards you want to fetch data from 
'''
boards_list = [
    "a", "b", "c", "d", "e", "f", "g", "gif", "h", "hr", "k", "m", "o", "p", "r", "s", "t", "u", "v", "vg", "vm", "vmg", "vr", "vrpg", "vst", "w", "wg",
    "i", "ic",
    "r9k", "s4s", "vip", "qa",
    "cm", "hm", "lgbt", "y",
    "3", "aco", "adv", "an", "bant", "biz", "cgl", "ck", "co", "diy", "fa", "fit", "gd", "hc", "his", "int", "jp", "lit", "mlp", "mu", "n", "news", "out", "po", "pol", "pw", "qst", "sci", "soc", "sp", "tg", "toy", "trv", "tv", "vp", "vt", "wsg", "wsr", "x", "xs"
]
'''
boards_list = [
    "o", "p", "r", "s", "t", "u", "v", "vg", "vm", "vmg", "vr", "vrpg", "vst", "w", "wg",
    "i", "ic",
    "r9k", "s4s", "vip", "qa",
    "cm", "hm", "lgbt", "y",
    "3", "aco", "adv", "an", "bant", "biz", "cgl", "ck", "co", "diy", "fa", "fit", "gd", "hc", "his", "int", "jp", "lit", "mlp", "mu", "n", "news", "out", "po", "pol", "pw", "qst", "sci", "soc", "sp", "tg", "toy", "trv", "tv", "vp", "vt", "wsg", "wsr", "x", "xs"
]

# Create the folder if it doesn't exist
if not os.path.exists("gh_training_data"):
    os.makedirs("gh_training_data")

def get_threads_for_board(board):
    # Construct the URL for fetching the catalog (list of threads) for the given board
    url = f"https://a.4cdn.org/{board}/catalog.json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        data = response.json()
        print(f"Fetched catalog data for board: /{board}")
        
        for page in data:
            for thread in page['threads']:
                try:
                    thread_id = thread['no']
                    fetch_thread_data(board, thread_id)
                except KeyError:
                    print(f"Thread ID missing or malformed in page: {page}")
                except Exception as e:
                    print(f"Error processing thread ID {thread_id}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching catalog for board {board}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def fetch_thread_data(board, thread_id):
    url = f"https://a.4cdn.org/{board}/thread/{thread_id}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        thread_data = response.json()
        print(f"Successfully fetched data for thread {thread_id}")
        
        # Save thread data to file
        file_path = f"gh_training_data/{board}_{thread_id}.json"
        with open(file_path, "w") as f:
            json.dump(thread_data, f)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for thread {thread_id}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while fetching thread {thread_id}: {e}")

# Iterate over each board
for board in boards_list:
    get_threads_for_board(board)
    time.sleep(1)  # Add a small delay between requests to avoid rate-limiting
