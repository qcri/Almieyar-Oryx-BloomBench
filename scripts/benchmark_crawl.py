from selenium import webdriver
from selenium_authenticated_proxy import SeleniumAuthenticatedProxy
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
try:
    from webdriver_manager.chrome import ChromeDriverManager
except Exception:
    ChromeDriverManager = None
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
import time
import random
import re
import urllib.parse
import hashlib
from multiprocessing import Pool, Lock
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
import torch
import numpy as np
import google.generativeai as genai

from scipy.spatial.distance import cosine


proxies = [

]

def setup_driver(proxy):
   
    chrome_options = webdriver.ChromeOptions()
    # Try to detect Chrome/Chromium binary. Honor CHROME_BIN env var if set.
    chrome_bin = os.environ.get('CHROME_BIN')
    possible_bins = [
        chrome_bin,
        '/usr/bin/google-chrome',
        '/usr/bin/google-chrome-stable',
        '/usr/bin/chromium',
        '/usr/bin/chromium-browser',
        '/snap/bin/chromium'
    ]
    for b in possible_bins:
        if b and os.path.exists(b):
            chrome_options.binary_location = b
            print(f"Using chrome binary: {b}")
            break
    # If a proxy is provided, configure proxy helper; otherwise run without proxy
    if proxy:
        proxy_helper = SeleniumAuthenticatedProxy(proxy_url=proxy, tmp_folder="")
        # print(proxy_helper.proxy_extension_dirname)
        hasher = hashlib.sha256()
        hasher.update(proxy.encode())

        digest = hasher.digest()

        hex_digest = digest.hex()
        
        # proxy_helper.enrich_chrome_options(chrome_options)
        
        # ext_path = os.path.join("/Users/omid/Desktop/tmp", hex_digest)
        # if os.path.exists(ext_path):
        #     chrome_options.add_argument(f"--load-extension={ext_path}")
    

        
        
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # print("Using extension from:", proxy_helper.extension_dir)

    print(chrome_options.arguments)

    # Prefer webdriver_manager if available to auto-download driver
    if ChromeDriverManager is not None:
        service = Service(ChromeDriverManager().install())
    else:
        service = Service()
    driver = webdriver.Chrome(options=chrome_options, service=service)
    
    return driver

def create_download_folder(folder, leaf, num_images):

    """Create a folder to store downloaded images"""
    folder_name = f"{folder}/{leaf}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    if len(os.listdir(folder_name)) == num_images:
        print(f"Folder {folder_name} already contains {num_images} images. Skipping...")
        return None
    return folder_name

def compute_image_embedding(image_path, model, preprocess):
    """Compute the normalized embedding of an image using a pre-trained model."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor).squeeze(0).numpy()
    # Normalize the embedding
    norm = np.linalg.norm(embedding)
    normalized_embedding = embedding / norm if norm > 0 else embedding
    return normalized_embedding

def is_image_unique(folder_path, given_image_path, similarity_threshold=0.7):
    """
    Check if the given image is unique in the folder based on cosine similarity.
    Returns True if unique, False otherwise.
    """
    # Load a pre-trained model (e.g., ResNet) and remove the classification head
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # Preprocessing pipeline for the images
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Compute embedding for the given image
    given_image_embedding = compute_image_embedding(given_image_path, model, preprocess)

    # Iterate through all images in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path != given_image_path:
            try:
                folder_image_embedding = compute_image_embedding(file_path, model, preprocess)
                similarity = 1 - cosine(given_image_embedding, folder_image_embedding)
                if similarity >= similarity_threshold:
                    print(f"Image {file_name} is similar to the given image with similarity {similarity:.2f}")
                    return False
            except Exception as e:
                print(f"Error processing image {file_name}: {str(e)}")

    return True


genai.configure(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-flash-preview")

image_evaluation_prompt = """
You are a vision-language model evaluating whether a crawled image is suitable for generating high-quality visual reasoning questions.

The goal is to determine whether the image meaningfully supports a specific visual capability grounded in Bloom’s Taxonomy, and whether it is visually rich enough to enable hard, complex, multi-step questions with reliable answers.

This evaluation focuses on task suitability and reasoning potential, not visual aesthetics.

---

Task Context

Each task is defined by a core visual concept (Leaf) and a cognitive ability (Bloom level).
The task description explains the exact visual capability that must be evaluated.

The image should naturally expose the visual evidence required to test this capability.

---

Input Fields

You are given the following task specification:

- Path: {{path}}
- Leaf: {{leaf}}
- Bloom Level: {{bloom_node}}
- Task Description: {{description}}

Use these inputs as follows:

- leaf defines *what* must be visually present and identifiable.
- bloom_node defines *how* the visual information should be mentally processed (e.g., identifying, differentiating, comparing, analyzing).
- description precisely explains *which visual distinctions, features, or relationships* must be observable and testable in the image.

---

Bloom’s Taxonomy Reference

1. Remembering – Recognizing, Recalling, Identifying
2. Understanding – Explaining, Interpreting, Classifying
3. Applying – Using, Executing, Solving
4. Analyzing – Comparing, Organizing, Differentiating
5. Evaluating – Judging, Critiquing, Validating
6. Creating – Designing, Constructing, Producing

---

Evaluation Focus

Evaluate whether the image:

- Clearly represents the target concept ({{leaf}})
- Supports the cognitive operation implied by {{bloom_node}}
- Avoids ambiguity caused by missing, unclear, or misleading visual information

---

Scoring Scale (1–5)

Assign a single overall score based on the definitions below:

5 – Excellent (Perfect Support)  
The image strongly and unambiguously supports the task. {{leaf}} is clearly represented, the visual distinctions required by {{description}} are explicit, and the image is rich enough to generate multiple hard, multi-step reasoning questions with clear answers.

4 – Good (Strong Support)  
The image supports the task well and allows challenging questions, but has minor limitations (e.g., slightly reduced complexity, partial viewpoints, or small ambiguities).

3 – Adequate (Limited Support)  
The image represents {{leaf}} but mainly supports simple or medium-difficulty questions. Visual richness or clarity is insufficient for consistently hard reasoning tasks.

2 – Weak (Poor Support)  
The image loosely relates to the task but lacks critical visual features described in description. Questions would be shallow, ambiguous, or unreliable.

1 – Unsuitable (No Support)  
The image does not support the task. {{leaf}} is unclear or absent, required visual evidence is missing, or the image is too simple or ambiguous to test the described capability.

---

Expected Output

Return only a valid JSON object in the following format:

{
  "score": <integer from 1 to 5>,
  "justification": "<concise explanation referencing leaf, bloom_node, and description>"
}

Do not include markdown, explanations, or any additional text outside the JSON.
"""



def download_image(url, folder_path, index, hierarchy):
    """Download an image from URL and save it"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:

            # Determine file extension from content-type
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = 'jpg'
            elif 'png' in content_type:
                ext = 'png'
            else:
                ext = 'jpg'  # Default to jpg
                
            file_path = os.path.join(folder_path, f"image_{index}.{ext}")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            if is_image_unique(folder_path, file_path):
                img = Image.open(file_path)
                prompt = image_evaluation_prompt.replace("{{leaf}}", hierarchy.get('leaf',''))
                prompt = prompt.replace("{{path}}", f"{hierarchy.get('lvl1','')} -> {hierarchy.get('leaf','')}")
                bloom_node = hierarchy['lvl1'].split('->')[0].strip()
                prompt = prompt.replace("{{bloom_node}}", bloom_node)
                prompt = prompt.replace("{{description}}", hierarchy['description'])
                response = model.generate_content([prompt, img])

                if response:
                    print(f"Image evaluation successful for {file_path}")
                    # append each results to end of log.json file
                    with open("log.json", "a") as log_file:
                        json.dump({"url": url, "response": response}, log_file)
                        log_file.write("\n")
                    json_text = response.text.strip().lstrip("```json").rstrip("```")
                    response_data = json.loads(json_text)
                    score = response_data['score']
                    if score >= 4:
                        print(f"Image evaluation successful for {file_path}")
                        # into jsonl file append url and file_path of successful image
                        with open("successful_images.jsonl", "a") as jsonl_file:
                            jsonl_file.write(json.dumps({"url": url, "file_path": file_path}) + "\n")
                        return True
                    else:
                        print(f"Image evaluation failed for {file_path}")
                        os.remove(file_path)
                        return False
                else:
                    os.remove(file_path)
                    print(f"Image evaluation failed for {file_path}")
                    return False
            else:
                os.remove(file_path)
                print(f"Removed duplicate image {file_path}")
                return False
            
    except Exception as e:
        print(f"Error downloading image {index}: {str(e)}")
    return False

def scrape_google_images_list(query_countryCode_terms, process_proxies, num_images, folder):
    # allow empty process_proxies: use a single None entry so modulo works
    use_proxies = process_proxies if process_proxies else [None]
    for idx, (query, folder_name, hierarchy) in enumerate(query_countryCode_terms):
        proxy = use_proxies[idx % len(use_proxies)]
        scrape_google_images(query, folder_name, proxy, num_images, folder, hierarchy)

def scrape_google_images(query, leaf, proxy, num_images=10, folder='google_images_scraped', hierarchy=None, retries=0):

    # Retries to deal with chrome driver issues
    if retries > 3:
        print(f"Failed to scrape {query} after {retries} retries")
        return

    folder_path = create_download_folder(folder, leaf, num_images)
    
    ### Query has been completed before
    if not folder_path:
        return
    
    try:
        driver = setup_driver(proxy)
        
        
    
    except Exception as e:
        print(f"Error setting up driver: {str(e)}\nretrying...")
        time.sleep(2)
        scrape_google_images(query, leaf, proxy, num_images, folder, hierarchy, retries + 1)
        return

    downloaded_count = 0

    try:

        # Construct the Google Images URL
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"
        
        # Add random delay to mimic human pacing and prevent being flagged as a bot
        time.sleep(random.uniform(2.0, 5.0))
        
        driver.get(url)
        
        driver.maximize_window()

        
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, "F0uyec"))
        )

        script_elements = driver.find_elements(By.TAG_NAME, 'script')

        for elem in script_elements:
            text = elem.get_attribute('innerHTML')

            if 'google.kEXPI=' in text:
                all_urls = re.findall(r'"https://[^"]+"', text)

                for url in all_urls:

                    if downloaded_count >= num_images:
                        break

                    if '.jpg' in url or '.png' in url or '.jpeg' in url:
                        url = url[1:-1]
                        url = url.split('&')[0].split('?')[0]
                        if url.endswith('jpg') or url.endswith('png') or url.endswith('jpeg'):

                            if download_image(url, folder_path, downloaded_count, hierarchy):
                                downloaded_count += 1
                                print(f"Downloaded image {downloaded_count}/{num_images} for {query}")  

                break      
                    
    except Exception as e:
        print("Error on ", query)
        print("Proxy used: ", proxy)
        print(f"Error scraping Google Images: {str(e)}")
        print("retrying...")
        time.sleep(2)
        scrape_google_images(query, leaf, proxy, num_images, folder, hierarchy, retries + 1)
        return
        
    try:
        print(f"\nDownload complete! {downloaded_count} images saved to {folder_path}")

    except Exception as e:
        print("Error on ", query)
        time.sleep(2)
        scrape_google_images(query, leaf, proxy, num_images, folder, hierarchy, retries + 1)
        return
    
    return      


# Example usage
if __name__ == "__main__":
    from tqdm import tqdm
    import csv
    import json

    query_countryCode_terms = []
    with open('Taxonomy.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        taxonomy_list = []
        for row in reader:
            # Handle possible unpacking variations just in case
            if len(row) >= 4:
                hierarchy_path = row[0]
                leaf = row[1]
                description = row[2]
                example = row[3]
            else:
                continue

            # Only process "Create" taxonomy items
            if not hierarchy_path.strip().startswith("Create"):
                continue

            # levels = hierarchy_path.strip().split("->")
            # lvl1 = levels[0].strip()  # Get only the first level
            taxonomy_list.append({
            "lvl1": hierarchy_path,
            "leaf": leaf.strip(),
            "description": description.strip(),
            "example": example.strip()
        })

    for i in taxonomy_list:
        # Build sanitized scenario filename to match generator output
        safe_name = f"{i['lvl1']} - {i['leaf']}"
        safe_name = re.sub(r'[<>:\\"/\\|?*\n\r]+', '_', safe_name).strip()
        scenarios_path = Path('scenarios') / f"{safe_name}.json"
        if not scenarios_path.exists():
            print(f"WARNING: scenario file not found: {scenarios_path}")
            continue
        with open(scenarios_path, 'r', encoding='utf-8') as f:
            keyword_data = json.load(f)
            keyword_data = keyword_data.get('keywords', [])
            for j in keyword_data:
                # sanitize query-derived folder name
                safe_query_part = re.sub(r'[^A-Za-z0-9._-]+', '_', j).strip('_')[:120]
                folder_name = f"{safe_name}_{safe_query_part}"
                query_countryCode_terms.append((j, folder_name, i))
        
    # query_templates = ['+Traditional +Clothing in +{}']


    ## Mapping from query template to folder name to store results of the query
    # folder_names = {'+Traditional +Clothing in +{}': 'Traditional Clothing'}

    # query_countryCode_terms = [(query_template.format(country), country_to_country_code_mapping[country], folder_names[query_template]) for query_template in query_templates for country in countries]

    ## Folder to store images
    output_folder = 'google_images_templated_queries_en'

    # Number of images to scrape for each query
    NUM_IMAGES = 10
    print(query_countryCode_terms)


    

    POOL_SIZE = 5

    inputs = [query_countryCode_terms[i::POOL_SIZE] for i in range(POOL_SIZE)]
    proxies_division = [proxies[i::POOL_SIZE] for i in range(POOL_SIZE)]

    # ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    if not proxies:
        print("WARNING: no proxies configured; setup_driver will be called with empty proxy list")

    with Pool(POOL_SIZE) as p:
        p.starmap(scrape_google_images_list, [(inputs[i], proxies_division[i], NUM_IMAGES, output_folder) for i in range(POOL_SIZE)])
