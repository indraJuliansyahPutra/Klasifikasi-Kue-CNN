import os
import requests
from PIL import Image
from io import BytesIO

def download_images_for_keywords(keywords, api_key, cse_id, num_images_per_keyword, save_folder):
    search_url = "https://www.googleapis.com/customsearch/v1"
    max_results_per_request = 10

    for query in keywords:
        query_folder = os.path.join(save_folder, query)
        if not os.path.exists(query_folder):
            os.makedirs(query_folder)
        
        count = 0
        for start in range(1, num_images_per_keyword + 1, max_results_per_request):
            print(f"Requesting images for '{query}' from {start} to {start + max_results_per_request - 1}")
            
            params = {
                'q': query,
                'key': api_key,
                'cx': cse_id,
                'searchType': 'image',
                'num': max_results_per_request,
                'start': start
            }
            
            response = requests.get(search_url, params=params)
            if response.status_code != 200:
                print(f"Error: {response.status_code}, {response.text}")
                break

            results = response.json()
            
            if 'items' not in results:
                print(f"No more images found for '{query}' at start index {start}.")
                break
            
            for item in results.get('items', []):
                try:
                    img_url = item['link']
                    img_data = requests.get(img_url).content
                    img = Image.open(BytesIO(img_data))
                    img_format = img.format.lower()
                    img.save(os.path.join(query_folder, f"{query}_{count}.{img_format}"))
                    count += 1
                    if count >= num_images_per_keyword:
                        break
                except Exception as e:
                    print(f"Error downloading image {count} for '{query}': {e}")
            
            if count >= num_images_per_keyword:
                break
        
        print(f"Downloaded {count} images for query '{query}'.")

# Contoh penggunaan
api_key = ""
cse_id = ""
keywords = [
    "kastengel",
    "indonesian kastengel",
    "cheese kastengel",
    "kastengel cake",
    "kastengel cookies",
    "traditional kastengel",
    "indonesian cheese stick",
    "kastengel recipe",
    "kastengel kue",
    "kastengel kering"
]
num_images_per_keyword = 100
save_folder = "kue kastengel - 1000"

download_images_for_keywords(keywords, api_key, cse_id, num_images_per_keyword, save_folder)