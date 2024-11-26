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
                "num_ctx": num_ctx,
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
        text = "Give me a one-liner summary of what the following research paper is about:\n"
        text += f"Title: {entry.title.replace('\n','')}\nAbstract: {entry.summary.replace('\n','')}"
        response = invoke_llm(text)
        one_liners.append(f"Title: {entry.title.replace('\n','')}\n" + response)
    return one_liners

# Function: Build input text to send to LLM
def summarize_articles(atom_feed_url, DEBUG=False):
    # Parse the feed
    if DEBUG:
        print("Extracting ATOM feeds")
    atom_feed = feedparser.parse(atom_feed_url)

    # Build text
    text = """The following are abstracts from the latest research papers on Artificial Intelligence. Your task is to identify the **top 3 research trends** based on the abstracts.

**What to do:**
1. Analyze the abstracts and group them into overarching themes (e.g., specific methodologies, applications, or challenges).
2. Identify **exactly 3 trends** that are discussed or implied in **multiple papers**, and prioritize those that seem most prominent or impactful.
3. Provide a brief explanation for each trend and reference specific papers to justify your choice.

**What NOT to do:**
1. Do NOT summarize each individual paper.
2. Do NOT include more than 3 trends.
3. Do NOT list generic topics like "machine learning" or "optimization" unless explicitly supported by multiple papers.

**Output Format:**
- **Trend 1**: [Clear and concise description of the trend]
  - Supporting Evidence: [List specific paper titles or key phrases from abstracts that support this trend]
- **Trend 2**: [Clear and concise description of the trend]
  - Supporting Evidence: [List specific paper titles or key phrases from abstracts that support this trend]
- **Trend 3**: [Clear and concise description of the trend]
  - Supporting Evidence: [List specific paper titles or key phrases from abstracts that support this trend]

**Articles**:
"""
    for entry in atom_feed.entries:
        text += f"Title: {entry.title.replace('\n','')}\nAbstract: {entry.summary.replace('\n','')}\n----------\n"

    ctx_len = len(text) // 4
    new_ctx = int(math.ceil(ctx_len // 1000) * 1000) + 1000
    if DEBUG:
        print(f"Estimated Context Length: {ctx_len}") # Assuming ~4 characters per token on average
    if ctx_len > N_CTX:
        if DEBUG:
            print(f"Increasing N_CTX to {new_ctx}")
        response = invoke_llm(text, num_ctx=new_ctx)
    else:
        response = invoke_llm(text)
    return response

if __name__ == "__main__":
    summary_text = summarize_articles(ATOM_FEED_URL, DEBUG=True)
    print(summary_text)

# # Parse the feed
# feed = feedparser.parse(atom_feed_url)

# # Access feed metadata
# print("Feed Title:", feed.feed.title)
# print("Feed Link:", feed.feed.link)

# # Iterate through feed entries
# for entry in feed.entries:
#     print("Title:", entry.title.replace('\n',''))
#     print("ID:", entry.id)
#     print("Published:", entry.published)
#     print("Author:", entry.author)
#     print("Category:", entry.category)
#     print("Summary:", entry.summary.replace('\n',''))
#     print()
