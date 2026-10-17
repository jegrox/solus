## Installation
1. Clone the repository into your local box.
2. Install the following dependencies:
   ```
   pip install anthropic deepeval voyageai rank-bm25 chromadb pymupdf4llm
   ```
## Set up authentication
Create a .env file in your local directory and add the following keys. Replace with your personal or organization keys

```
ANTHROPIC_API_KEY="{anthropic_key}"
VOYAGE_API_KEY="{voyageai_key}"
OPENAI_API_KEY="{openai_key}"
```

Currently, this project makes use of Claude for running the main agent loop, deepeval is run with its default ChatGPT agent and voyageAPI are used to create the embeddings for our vector index.
As the project grows, this dependencies will become looser and you should be able to select which APIs to use.

## Create indexes
Before been able to run the evaluation or the main agent loop, we need to create the index locally. If using the default github doc dataset (maybe for testing) skip until the "Run indexing" section

### Fill in the data and the evaluation set
Replace the documents in the data/ folder with the document set you wish to use. The indexer will iterate through the documents in a recursive way, so no specific folder structure is required.

Also replace the eval-set.txt with your own evaluation set. The evaluator takes a list of question/answer pairs in the following format:

```
"{question}", "{answer}"
```

Example:
```
"What does amending a commit do in GitHub Desktop, and why should you avoid amending a commit that has already been pushed?", "Amending a commit modifies the most recent commit on your current branch by replacing it with a new commit. It is useful when you need to edit the commit message or forgot to include changes. If the commit has already been pushed, you must force push to overwrite the commit history in the remote repository, which may confuse collaborators who have already based work on the commit."
```

### Run indexing
```
python main.py build-index
```
You can run the build-index several times and it will only build new indexes. If you want to rebuild the whole index you can use the --force flag as follows:

```
python main.py build-index --force
```

## Running the evaluation
```
python main.py eval
```

## Run the agent
```
python main.py
```
