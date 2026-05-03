# Importing the packages
from datetime import datetime, date,timedelta
import pandas as pd
import psycopg2
from pathlib import Path
from psycopg2.extras import execute_values
import numpy as np
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import bisect
import os
from dotenv import load_dotenv
import warnings
import logging
import sys
import getpass

# Trying to get user name of this system (to be used in promt design)
try:
    user_name = getpass.getuser().capitalize()
except:
    user_name = "User"  # If not fetched, we would use this default name.
print("Hello {}".format(user_name))

## Update the name of directory in which those content are stored
personal_diary_dir = 'rohan_diary_contents'
print("Trying to use diary in: {}.\nChecking if it exists...".format(personal_diary_dir))
## checking if it exists
while True:
    if Path(personal_diary_dir).exists():
        print("Ok, this directory exists")
        break
    else:
        print("Cannot find this directory, please enter a valid directory")
        personal_diary_dir = input("Enter a Valid directory: ")
    

# Filling in new values to our Vector DB
## Connecting to PostgreSQL DB
load_dotenv(dotenv_path='Coding/mind_mirror_config.env')  # Loading database credentials saved in .env file
db_config = {
    "host": os.getenv('host'),
    "database": os.getenv('database'),
    "user": os.getenv('user'), 
    "password": os.getenv('password') 
}

conn = psycopg2.connect(**db_config)
cur = conn.cursor()

## Getting the last date till which we have data in DB
cur.execute("select created_at from personal_diary order by 1 desc limit 1;") # change table name as per your choice.
update_check = False
row = cur.fetchone()

if row is None:
    print("No records in DB")
else:
    print("Latest updated date from the DB:",row[0])
    latest_db_update = row[0]

# Data Preprocessing
## Reading the files in diary directory
### Creating list of data to be feeded in dataframe
diary_contents = []  # List of contents
diary_dates = []  # List of dates
for i in Path(personal_diary_dir).iterdir():
    # print(i.name)
    date_obj = datetime.strptime(i.name, "%Y-%m-%d").date()  # Converting the filename into python datetime obj for further processing
    if row is None:  # in case no records fetched from DB. Meaning, its the first time user running this script.
        diary_dates.append(date_obj)
        with open(i, encoding='utf-8') as f:
            diary_contents.append(f.read())
    else:
        if latest_db_update < date_obj:  # We would only want those files which are new to 
            diary_dates.append(date_obj)
            with open(i, encoding='utf-8') as f:
                diary_contents.append(f.read())

# Finding the last updated date of the diary.
## Getting list of dates (as created_date) from the metadata of all the notes fetched (in case we have new entries to be pushed into DB)
if len(diary_dates)==0:
    print("Nothing to be processed!. Moving on to query processing...")  # in case of no new entries, we would directly jump to query search.
else:
    print("Last updated date of the diary:",sorted(diary_dates, reverse=True)[0])

## Suppressing Huggingface, BERT and other warnings..
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TQDM"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
for logger_name in [
    "transformers", "transformers.modeling_utils", "transformers.configuration_utils",
    "huggingface_hub", "huggingface_hub.repocard", "sentence_transformers",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

class SuppressOutput:
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self._stderr
warnings.filterwarnings("ignore")

# Initialising the embedding model
with SuppressOutput():
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Creating Semantic chunks (in case of new entry push)
if len(diary_dates) > 0:
    # Here, we're keeping similarity threshold to 97%. Meaning only when two chunks in same note have similarity of below 97%, we would split
    chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=97)
    processed_data = []  # These woud have chunks from the original content list
    processed_dates = []  # These would have dates derived for those chunks

    for date, content in zip(diary_dates, diary_contents):
        chunks = chunker.split_text(content)  # creating semantic chunks out of original contents
        processed_data.extend(chunks)
        processed_dates.extend([date] * len(chunks))  # we would repeat and use the same date for each of the chunks for this 
    processed_embeddings = embed_model.embed_documents(processed_data)

# Pushing the data to our local Postgres DB
## Ensuring pgvector extension is enabled
if len(diary_dates) > 0:
    print("Updating the new values to the DB")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    insert_query = "insert into personal_diary (created_at, content, embeddings) values %s"
    values = list(zip(processed_dates, processed_data, processed_embeddings))
    
    execute_values(cur, insert_query, values)
    conn.commit()
    cur.close()
    conn.close()

# Doing query search to our vector DB
query = input("Enter the question or query you want to search: ")
query_embedding = embed_model.embed_query(query)

## We would fetch top 5 most similar chunks from the DB
print("Great!. Searching for the contexts...")
conn = psycopg2.connect(**db_config)
cur = conn.cursor()
cur.execute("""
    SELECT id, created_at, content,
           1 - (embeddings <=> %s::vector) AS similarity
    FROM personal_diary
ORDER BY created_at DESC, similarity DESC
limit 5;
""", (query_embedding,))

similar_rows = cur.fetchall()
conn.commit()
cur.close()
conn.close()
matching_contents = ["As on : "+str(i[1]) + "; " + i[2] for i in similar_rows]  # this would have list of matched chunks of data from DB, with dates

print("Do you want to use your local ollama model?, or Gemini API for your answer: ")
while True:
    user_inp = input('Type "G" for using Gemini or "O" for Ollama (g/o). Type "S" for just displaying the top 5 matched entries: ')
    if user_inp.lower() == 'g' or user_inp.lower() == 'o' or user_inp.lower() == 's':
        break
    else:
        print("Try again..")

prompt = """Based on the following snippets from {}'s career: {}.
Answer the following question: {}.
start responding by greeting me. Your answer should always be on the point, short (like 2-3 paragraphs) easy to understand. Avoid using jargons""".format(user_name,matching_contents,query)

if user_inp.lower() == 'g':
# Connecting to Google Gemini.
    from google import genai
    client = genai.Client(api_key=os.getenv('gem_api_key'))  # Getting API key from the env file
    # Trying to connect to Google API
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        print(response.text)
    except:
        print("There was some problem connecting to LLM. But below are the 5 most similar chunks from your database:")
        for i in matching_contents:
            print(i)
            print("####")
    
if user_inp.lower() == 'o':
    import platform
    import requests
    ollama_url = "http://localhost:11434"
    OS = platform.system()
    # Checking if ollama is running
    def ollama_running():
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=3)
            return r.status_code == 200
        except:
            return False

    def get_installed_models():
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        models = r.json().get("models", [])
        return [m["name"] for m in models]

    def choose_model(models: list[str]):  # Must have a list of models as parameter
        print("Installed models:")
        for i, name in enumerate(models, start=1):
            print(" {}: {}".format(i, name))
        print()
        while True:
            choice = input(f"Select a model (1-{len(models)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                return models[int(choice) - 1]
            print("Invalid choice. Please enter a number from the list.")        

    def run_prompt(model:str, user_prompt: str):
        r = requests.post(f"{ollama_url}/api/generate",
            json={"model": model, "prompt": user_prompt, "stream": False},
            timeout = 600)
        return r.json().get("response", "").strip() 
        
    if not ollama_running():
        print("Ollama server is not running. Please start it first")
        print("For Linux: sudo systemctl start ollama")
        print("For macOS: open -a Ollama")
        print("For Windows (from command prompt): ollama serve")
        sys.exit(1)
    
    models = get_installed_models()
    if not models:
        print("No models installed. Pull one first. Refer: https://ollama.com/library")
        sys.exit(1)

    selected_model = choose_model(models)
    print("Ok!, I'll use {} model".format(selected_model))
    print("-"*30)
    print("RESPONSE (Generating response. Please wait for sometime. Depending on system, it might take about 2-6 mins.): ")
    print(run_prompt(selected_model, prompt))

if user_inp.lower() == 's':
    print("Ok!. Here are top 5 matching chunks from your database:")
    for i in matching_contents:
        print(i)
        print("####")
