# Chat With Multiple Pdfs

## Introduction
------------
A simple application to upload multiple pdfs at once and ask questions across them. It will return you the most closest answer along with pinpointing the source of information too. 

## High-level overview
------------

1. `PyPDF2` for pdf reading/parsing.

2. `Langchain` for most of the LLM related tasks like splitting, embedding and semantic search.

3. `FAISS` for vectorstore. 

4. `Streamlit` for putting it all together.


## Setup and Installation

1. Start by cloning the repository to your local machine:
```shell
git clone https://github.com/suyogdahal/chat-with-multiple-pdfs.git
cd chat-with-multiple-pdfs
```

2. Poetry is being used as the dependency manager in this project. If you don't have it installed, install it from the official Poetry documentation.

3. Once you have Poetry installed, simply run the following command to install the application dependencies:

```shell
poetry install
```

4. Activate the Poetry shell to handle dependencies in a virtual environment:
```shell
poetry shell
```

5. Use the command below to run the app:
```shell
streamlit run app.py
```

Once the application is running, navigate to the localhost URL (usually http://localhost:8501) displayed in your terminal.
Now, you are all set to upload your PDFs and chat with them!