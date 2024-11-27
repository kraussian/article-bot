# pip install -U python-dotenv requests ollama feedparser tqdm
import os
import math
import requests
import feedparser
from   tqdm import tqdm
from   dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID    = os.getenv("ENDPOINT_ID")
if not RUNPOD_API_KEY:
    raise ValueError("API key not found! Ensure RUNPOD_API_KEY is set in the .env file.")
if not ENDPOINT_ID:
    raise ValueError("Endpoint ID not found! Ensure ENDPOINT_ID is set in the .env file.")

# Define Runpod API Endpoint
url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

# Define Runpod API Key
api_key = RUNPOD_API_KEY

# Define Atom Feed URL of arXiv articles to extract
# arXiv API User Manual: https://info.arxiv.org/help/api/user-manual.html
CATEGORY = "cs.AI" # cs.AI is the arXiv tag for Artificial Intelligence
MAX_RESULTS = 50
ATOM_FEED_URL = f"http://export.arxiv.org/api/query?search_query=cat:{CATEGORY}&max_results={MAX_RESULTS}"

# Define LLM Parameters
N_CTX = 16384
TEMPERATURE = 0.7 # Default 0.8
SEED = 42 # Default 0
TOP_P = 0.9 # Default 0.9
TOP_K = 40 # Default 40
NUM_PREDICT = -2 # Default 128, -1: Infinite, -2: Fill Context

# Function: Invoke LLM
def invoke_llm(input_text, num_ctx=N_CTX):
    # Define headers for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Define the payload for the request
    payload = {
        "input": {
            "method_name": "generate",
            "input": {
                "prompt": input_text,
                "options": {
                    "num_ctx": num_ctx,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "num_predict": NUM_PREDICT,
                }
            }
        }
    }

    # Send the POST request
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("output", {}).get("response", "No response found.")
    except requests.exceptions.RequestException as e:
        return f"Error communicating with the model: {e}"

# Function: Build one-liner summary of each paper
def build_one_liner(atom_feed_url=ATOM_FEED_URL, DEBUG=False):
    # Parse the feed
    if DEBUG:
        print("Extracting ATOM feeds")
    atom_feed = feedparser.parse(atom_feed_url)
    one_liners = []
    for entry in tqdm(atom_feed.entries, desc="Building one-liners"):
        # Ensure abstract fits into the LLM prompt size
        abstract = entry.summary.replace('\n', '')[:2000]  # Truncate to 2000 characters if needed
        text = (
            "Summarize the following research paper in **one sentence**. "
            "Focus on the main finding or purpose of the research. Avoid excessive details or context.\n"
            f"Title: {entry.title}\nAbstract: {abstract}"
        )
        response = invoke_llm(text)
        one_liners.append(f"Title: {entry.title}\nSummary: {response}")
    return one_liners

# Function: Extract insights from the list of one-liner summaries
def summarize_one_liners(one_liners, DEBUG=False):
    text = (
        "Below are one-line summaries of recent AI research papers. Your task is to identify exactly **3 key insights** "
        "or trends emerging from these summaries.\n\n"
        "**Instructions:**\n"
        "1. Group summaries into themes or research areas if they share a common methodology, application, or challenge.\n"
        "2. Identify the **top 3 insights** based on the most recurring and impactful themes.\n"
        "3. Provide a short explanation for each insight and reference specific paper titles to support your conclusion.\n\n"
        "**What NOT to do:**\n"
        "1. Do NOT summarize each individual paper.\n"
        "2. Do NOT include more than 3 insights.\n"
        "3. Do NOT list generic topics unless explicitly supported by multiple papers.\n\n"
        "**Output Format:**\n"
        "- **Insight 1**: [Short and clear description of the insight]\n"
        "  - Supporting Evidence: [Paper titles or quotes from summaries]\n"
        "- **Insight 2**: [Short and clear description of the insight]\n"
        "  - Supporting Evidence: [Paper titles or quotes from summaries]\n"
        "- **Insight 3**: [Short and clear description of the insight]\n"
        "  - Supporting Evidence: [Paper titles or quotes from summaries]\n\n"
        "**Summaries:**\n"
    )
    for record in one_liners:
        text += f"{record}\n----------\n"

    if DEBUG:
        print(f"Sending text to LLM:\n{text}")
    response = invoke_llm(text)
    return response

if __name__ == "__main__":
    one_liners = build_one_liner(ATOM_FEED_URL, DEBUG=True)
    insights = summarize_one_liners(one_liners)
    print(insights)
