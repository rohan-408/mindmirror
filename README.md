# 🪞 MindMirror

> *Talk to your journal. Rediscover yourself.*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791)](https://github.com/pgvector/pgvector)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange)](https://aistudio.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Motivation](#-motivation)
- [How It Works — RAG Explained](#-how-it-works--rag-explained)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Step 1 — Set Up Python Virtual Environment](#step-1--set-up-python-virtual-environment)
- [Step 2 — Install Required Packages](#step-2--install-required-packages)
- [Step 3 — Set Up PostgreSQL with pgvector](#step-3--set-up-postgresql-with-pgvector)
- [Step 4 — Get Your Google Gemini API Key](#step-4--get-your-google-gemini-api-key)
- [Step 5 — Configure the .env File](#step-5--configure-the-env-file)
- [Step 6 — Running the App](#step-6--running-the-app)
- [Example Queries](#-example-queries)
- [Future Roadmap](#-future-roadmap)

---

## 📌 About the Project

**MindMirror** is a local, privacy-first semantic search engine for your personal documents — starting with journal entries from [CloudScript Journal](https://github.com/rohan-408/cloudscript_journal). Instead of grepping through dozens of plain text files hoping to find something, MindMirror lets you simply *ask a question* and get a thoughtful, context-aware answer drawn directly from what you have written.

CloudScript Journal saves your diary entries as neatly dated plain text files on your local system. MindMirror reads those files, understands their meaning using AI embeddings, stores them in a local vector database, and then uses Google Gemini to answer your questions in natural language — all running on your own machine, with no data sent to any third-party storage.

In the future, MindMirror is designed to expand beyond journal files and connect with other personal documents such as spreadsheet logs, PDF resumes, university research papers, notes, and more — essentially becoming a single place to query everything important in your life. No more forgetting what you learned, decided, or felt.

---

## 💡 Motivation

Most journaling apps available online — even the premium ones — lack proper **semantic search**. You can keyword-search, tag, or scroll through entries, but none of them truly understand the *meaning* of what you have written. The few platforms that do offer intelligent search tend to lock it behind expensive subscription tiers, require you to upload your most personal thoughts to their servers, and offer little transparency about how your data is handled.

This felt like a gap worth filling. Journaling is deeply personal, and the value of a journal grows as it gets longer — but so does the difficulty of actually using it to recall insights, patterns, and lessons.

That frustration led to two companion projects:

1. **[CloudScript Journal](https://github.com/rohan-408/cloudscript_journal)** — A simple, distraction-free terminal-based journaling tool that stores your entries as local text files, dated and organised automatically.
2. **MindMirror** *(this project)* — A semantic search and Q&A layer on top of those files, powered entirely by open-source models and your own local infrastructure.

Together, they give you a complete, private, and intelligent journaling experience — for free.

---

## 🧠 How It Works — RAG Explained

MindMirror uses a technique called **Retrieval-Augmented Generation (RAG)**. Here is an intuitive breakdown of each stage:

```
Your Journal Files
      │
      ▼
  Read all diary files from the directory
      │
      ▼
  Compare latest file dates against the PostgreSQL DB
  (Only new entries are processed — no redundant work)
      │
      ▼
  Split each new entry into Semantic Chunks
  (using SemanticChunker — splits on meaning, not character count)
      │
      ▼
  Generate Vector Embeddings for each chunk
  (using all-MiniLM-L6-v2, a compact but powerful local model)
      │
      ▼
  Store chunks + embeddings in local PostgreSQL (pgvector)
      │
      ▼
  User asks a question
      │
      ▼
[R] RETRIEVAL — Embed the question, run cosine similarity search in PostgreSQL
      │  Returns top 5 most semantically similar chunks
      ▼
[A] AUGMENTED — The question + retrieved chunks are combined into a structured prompt
      │
      ▼
[G] GENERATION — (Optional) Google Gemini reads the prompt and generates a natural language answer.
      │
      ▼
  Answer printed to terminal
```

### The Three Pillars of RAG

**[R] Retrieval** — The user's question is converted into a vector embedding just like the stored chunks were. The database then performs a cosine similarity search, returning the 5 chunks that are closest in *meaning* — not just keyword matches. This is what makes the search semantic: asking *"what did I learn from failures?"* will surface relevant entries even if the word "failure" never appears explicitly.

**[A] Augmented** — Those retrieved chunks are injected into a prompt alongside the original question. The LLM is not asked to answer from memory or general knowledge; it is given the actual relevant context from your own journal. This grounds the answer in your real data and prevents hallucination.

**[G] Generation** — Google Gemini reads the enriched prompt and produces a coherent, summarised answer. Since it is reading your own words, it can identify patterns, summarise learnings, and connect dots across multiple diary entries spanning months or years. It is set to optional now. If user wants to privately search the database, it simply outputs the top 5 most similar chunks of data without calling the API.

---

## 📁 Project Structure

```
mindmirror/
│
├── app.py                    # Main application — runs everything
├── mind_mirror_config.env    # Your private credentials (never commit this!)
└── README.md                 # This file
```

### `app.py`
The single entry point for the entire application. It handles:
- Reading your diary directory and detecting new entries
- Comparing against the database to avoid reprocessing old data
- Chunking and embedding new content
- Pushing embeddings to PostgreSQL
- Accepting your question via terminal input
- Performing vector similarity search
- Calling Gemini with the retrieved context and printing the answer

### `mind_mirror_config.env`
A local environment file that stores your credentials securely.

---

## ✅ Prerequisites

Before you begin, make sure the following are installed on your system:

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.8 or higher | Use `python3 --version` to check |
| PostgreSQL | 14 or higher | Must support the pgvector extension |
| pip | Latest | Comes with Python |
| Git | Any | For cloning the repo |
| Internet connection | — | For Gemini API calls and first-time model download |

---

## Step 1 — Set Up Python Virtual Environment

It is strongly recommended to run this project inside a **Python virtual environment**. This keeps the project's dependencies isolated from your system Python and prevents version conflicts with other projects.

### On macOS / Linux

```bash
# Navigate to the project folder
cd mindmirror

# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your terminal prompt should now show (venv) at the start
# Example: (venv) user@machine:~/mindmirror$
```

### On Windows

```cmd
# Navigate to the project folder
cd mindmirror

# Create a virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

### To Deactivate (when you are done)

```bash
deactivate
```

> 💡 **Important:** Every time you open a new terminal session to run MindMirror, you need to re-activate the virtual environment first using `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows). The app will not find its packages otherwise.

---

## Step 2 — Install Required Packages

With your virtual environment activated, install all dependencies using pip:

```bash
pip install google-genai
pip install psycopg2-binary
pip install pandas
pip install numpy
pip install langchain-experimental
pip install langchain-huggingface
pip install sentence-transformers
```

Or install them all in a single command:

```bash
pip install google-genai psycopg2-binary pandas numpy langchain-experimental langchain-huggingface sentence-transformers
```

### What Each Package Does

| Package | Purpose |
|---|---|
| `google-genai` | Official Python SDK for Google Gemini — sends prompts and receives answers |
| `psycopg2-binary` | PostgreSQL adapter for Python — connects to your local database |
| `pandas` | Data manipulation — manages diary content, dates, and dataframes |
| `numpy` | Numerical operations — used for vector math |
| `langchain-experimental` | Provides `SemanticChunker` for splitting text by meaning rather than fixed size |
| `langchain-huggingface` | Bridges LangChain with HuggingFace embedding models |
| `sentence-transformers` | Runs the `all-MiniLM-L6-v2` model locally to generate 384-dimensional embeddings |

> 📝 The `all-MiniLM-L6-v2` embedding model (~90MB) is downloaded automatically the first time you run the app. After that it is cached locally. **Your journal text never leaves your machine during the embedding step** — this model runs 100% offline.

---

## Step 3 — Set Up PostgreSQL with pgvector

### 3.1 Install PostgreSQL

If PostgreSQL is not already installed on your system:

**macOS (using Homebrew):**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Windows:**
Download the installer from [https://www.postgresql.org/download/windows/](https://www.postgresql.org/download/windows/) and follow the setup wizard. During installation, note down the password you set for the `postgres` user — you will need it later.

### 3.2 Install the pgvector Extension

pgvector adds vector similarity search capabilities to PostgreSQL. This is the engine behind MindMirror's semantic search.

**macOS (Homebrew):**
```bash
brew install pgvector
```

**Ubuntu / Debian:**
```bash
sudo apt install postgresql-15-pgvector
```

> For other operating systems or manual compilation from source, refer to the official pgvector repository: [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)

### 3.3 Create a Database and Enable pgvector

Open the PostgreSQL interactive shell:

```bash
# On Linux / macOS
sudo -u postgres psql

# On Windows — open 'SQL Shell (psql)' from the Start Menu
# Or run: psql -U postgres
```

Inside the psql shell, run the following commands one by one:

```sql
-- Create a new database for MindMirror
CREATE DATABASE mindmirror_db;

-- Connect to it
\c mindmirror_db

-- Enable the pgvector extension (only needs to be done once per database)
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3.4 Create the Table Schema

MindMirror stores your journal chunks in a table. Create it with the exact schema below:

```sql
CREATE TABLE personal_diary (
    id         SERIAL PRIMARY KEY,
    created_at DATE,
    content    TEXT,
    embeddings VECTOR(384)
);
```

**Column breakdown:**

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PRIMARY KEY | Auto-incrementing unique ID for each stored chunk |
| `created_at` | DATE | Date of the original diary entry this chunk came from |
| `content` | TEXT | The actual text of the semantic chunk |
| `embeddings` | VECTOR(384) | The 384-dimensional embedding vector (matches `all-MiniLM-L6-v2` output) |

### 3.5 Indexing for Large Tables

As your journal grows over months and years, vector searches can slow down. By default, pgvector scans every row — fine for small tables, but inefficient at scale.

**When your table reaches around 10,000 rows**, create an IVFFlat index to dramatically speed up similarity searches:

```sql
CREATE INDEX ON personal_diary
USING ivfflat (embeddings vector_cosine_ops)
WITH (lists = 100);
```

**When your table grows beyond 10,000 rows**, scale the `lists` parameter proportionally using this formula:

```
lists = total number of rows / 1000
```

For example, if your table has 50,000 rows:

```sql
CREATE INDEX ON personal_diary
USING ivfflat (embeddings vector_cosine_ops)
WITH (lists = 50);
```

> 💡 The `lists` value controls how many clusters the IVFFlat index uses for approximate nearest-neighbour search. A higher value increases accuracy but takes slightly longer to build. The formula `lists = rows / 1000` is a well-established practical guideline.

To exit the psql shell at any time, type `\q` and press Enter.

---

## Step 4 — Get Your Google Gemini API Key

MindMirror uses Google Gemini as its language model to generate natural language answers from the retrieved journal chunks. Follow these steps to get your free API key:

1. Open your browser and navigate to **[https://aistudio.google.com](https://aistudio.google.com)**

2. **Sign in** with your Google account.

3. On the left-hand side panel, click the **"Get API key"** button.

4. Click **"Create API Key"**.

5. When prompted to choose a project, click **"New project"** — this creates a fresh Google Cloud project associated with your key.

6. Your new API key will appear on screen. It will look something like:
   ```
   AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

7. **Copy this key immediately** and save it somewhere safe — you will paste it into your `.env` file in the next step.

> ⚠️ **Security:** Treat your API key like a password. Never paste it directly into `app.py`, never commit it to GitHub, and never share it publicly. Store it only in your local `mind_mirror_config.env` file.

> 💰 **Cost:** Google Gemini offers a generous free tier that is more than sufficient for personal journaling use. You are unlikely to hit any limits with typical daily usage.

---

## Step 5 — Configure the .env File

In your project folder, create a file named exactly `mind_mirror_config.env`. Open it in any text editor and fill it in as follows:

```env
# mind_mirror_config.env
# !! Never commit this file to Git !!

# --- PostgreSQL Configuration ---
DB_HOST=localhost
DB_NAME=mindmirror_db
DB_USER=postgres
DB_PASSWORD=your_postgres_password_here

# --- Google Gemini ---
GEMINI_API_KEY=your_gemini_api_key_here

# --- Journal Directory ---
# Full absolute path to the folder where CloudScript Journal saves your diary files
# Each file in this folder must be named in YYYY-MM-DD format (e.g. 2024-03-15)
DIARY_DIR=/home/yourname/Documents/your_diary_folder
```

**How to find these values:**

- `DB_HOST` — Use `localhost` if PostgreSQL is running on the same machine (which it is for most setups).
- `DB_NAME` — The name of the database you created in Step 3 (`mindmirror_db`).
- `DB_USER` — Your PostgreSQL username. On most fresh installations this is `postgres`.
- `DB_PASSWORD` — The password for that PostgreSQL user.
- `GEMINI_API_KEY` — The key you copied in Step 4.
- `DIARY_DIR` — The full absolute path to the folder where CloudScript Journal saves your `.txt` diary entries. Each file in this folder should be named `YYYY-MM-DD` (e.g. `2024-03-15`). On Linux/macOS, navigate to the folder in your terminal and run `pwd` to get the full path.

**Add this file to `.gitignore` immediately:**

```bash
echo "mind_mirror_config.env" >> .gitignore
```

---

## Step 6 — Running the App

Make sure your virtual environment is activated and PostgreSQL is running, then simply run:

```bash
python3 app.py
```

### What Happens on First Run

On the very first run, MindMirror will:

1. Scan all diary files in your configured `DIARY_DIR`.
2. Download the `all-MiniLM-L6-v2` embedding model (~90MB). This happens only once and is cached for future runs.
3. Split every diary entry into semantic chunks using the SemanticChunker (threshold: 97% similarity — meaning chunks are only split when their content meaningfully diverges).
4. Generate a 384-dimensional vector embedding for each chunk.
5. Insert all chunks, dates, and embeddings into your `personal_diary` table in PostgreSQL.

This first run may take a few minutes depending on how many journal entries you have and your machine's speed. Subsequent runs are much faster.

### What Happens on Every Subsequent Run

MindMirror checks the most recent `created_at` date in the database. It then only processes diary files newer than that date — making the update step incremental and quick.

### Asking Your Question

Once the database sync is complete, you will see a prompt in your terminal. Type your question and press Enter:

```
What would you like to ask your journal?
> What are the key learnings from all my job interviews?

Searching your journal...

Based on your journal entries, here are the key learnings from your interviews:
...
```

---

## 💬 Example Queries

Here are some questions you can ask MindMirror once your journal is indexed:

```
What are the recurring themes in my life this year?
What did I learn from my failures?
How have my goals changed over the past 6 months?
What are the best decisions I have made?
When do I feel the most productive?
What patterns do I notice in my relationships?
How have I been feeling about my career lately?
What are things I keep saying I will do but haven't?
What moments made me the happiest this year?
```

The more you journal with CloudScript Journal, the richer and more useful the answers from MindMirror become.

---

## 🗺 Future Roadmap

MindMirror is designed to grow beyond diary files. Planned additions include:

- **Spreadsheet support** — Query your financial logs, habit trackers, or any tabular records you keep.
- **PDF support** — Ask questions across your university research papers, saved articles, or e-books.
- **Resume & career documents** — Semantic search through your own professional history.
- **Unified query layer** — A single interface to search across all your personal document types at once.
- **Optional web UI** — A simple browser-based chat interface instead of terminal input.

The long-term vision: a personal AI that knows everything you have ever documented about your own life — and runs entirely on your machine.

---

## 🔒 Privacy

MindMirror is built with privacy as a first principle:

- All vector embeddings are generated **locally** using a HuggingFace model. Your journal text never leaves your machine for the embedding step.
- Your diary content is stored only in your **local PostgreSQL database** on your own system.
- The **only external network call** is to the Google Gemini API, which receives only the top 5 retrieved chunks (not your entire journal) along with your question to generate an answer.
- No usage data, telemetry, or personal information is collected by this project.

---

## 🤝 Related Project

MindMirror is built to work alongside **[CloudScript Journal](https://github.com/rohan-408/cloudscript_journal)** — a simple, terminal-based journaling tool that saves your entries as plain-text files named by date, stored on your local system. If you have not set that up yet, start there first before running MindMirror.

---

*Built with ❤️ for people who journal seriously and want to remember everything they learn.*
