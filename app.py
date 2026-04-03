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
from google import genai
import os
from dotenv import load_dotenv

## Update the name of directory in which those content are stored
personal_diary_dir = '/home/rohan/Documents/rohan_diary_contents'
## checking if it exists
if Path(personal_diary_dir).exists():
    print("Ok, this directory exists")
else:
    print("Cannot find this directory, please enter a valid directory")

# Filling in new values to our Vector DB
## Connecting to PostgreSQL DB
load_dotenv(dotenv_path='/home/rohan/Documents/Coding/mind_mirror_config.env')  # Loading database credentials saved in .env file
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

# Creating Semantic chunks (in case of new entry push)
if len(diary_dates) > 0:
    # Initialising the embedding model
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    insert_query = "insert into personal_diary (created_at, content, embeddings) values %s"
    values = list(zip(processed_dates, processed_data, processed_embeddings))
    
    execute_values(cur, insert_query, values)
    
    conn.commit()
    cur.close()
    conn.close()

# Doing query search to our vector DB
query = input("Enter the question or query you want to search: ")
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query_embedding = embed_model.embed_query(query)

## We would fetch top 5 most similar chunks from the DB
print("Great!. Searching for the contexts...")
cur.execute("""
    SELECT id, created_at, content,
           1 - (embeddings <=> %s::vector) AS similarity
    FROM personal_diary
ORDER BY similarity DESC
limit 5;
""", (query_embedding,))
similar_rows = cur.fetchall()

# Connecting to Google Gemini.
client = genai.Client(api_key=os.getenv('gem_api_key'))  # Getting API key from the env file

import getpass
## Trying to get user name of this system (to be used in promt design)
try:
    user_name = getpass.getuser().capitalize()
except:
    user_name = "User"  # If not fetched, we would use this default name.

matching_contents = [i[2] for i in similar_rows]  # this would have list of matched chunks of data from DB
prompt = """Based on the following snippets from {}'s career: {}.
Answer the following question: {}.
start responding by greeting me. Your answer should always be on the point, short (like 2-3 paragraphs) easy to understand. Avoid using jargons""".format(user_name,matching_contents,query)
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt
)
print(response.text)
