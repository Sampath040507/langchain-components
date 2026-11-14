# langchain-components
Hands-on examples and practical implementations of LangChain components, covering chat models, embeddings, prompt templates, message history, chains, output parsers, and runnables. Includes step-by-step explorations of how these components work together to build LLM-powered applications.

This repository explores how models, embeddings, prompts, message history, chains, output parsers, and runnables work through small, easy-to-follow programs.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Programs and Examples](#programs-and-examples)
- [Environment Variables and API Keys](#environment-variables-and-api-keys)
- [Further Learning](#further-learning)

## Overview

This repository is a collection of examples created while learning LangChain. Each program focuses on a specific concept and shows how it can be used in practice.

The examples range from finding relevant documents with embeddings to building prompts, maintaining chat history, and connecting components into workflows.

## Repository Structure

```text
langchain-components/
│
├── 1_document_similarity.py
├── 2_standalonequery_dynamic.py
├── 3_listofmessages_static.py
├── 4_chat_history_example.txt
├── 4_listofmessages_dynamic.py
├── 5_langchain_chains_runnablespart1.ipynb
├── 5_langchain_chains_runnablespart2.ipynb
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd langchain-components
```

Replace `<repository-url>` with the URL of your GitHub repository.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install langchain langchain-core langchain-openai python-dotenv scikit-learn streamlit jupyter
```

### 4. Configure your API key

Some examples use OpenAI models or embeddings. Follow the instructions in [Environment Variables and API Keys](#environment-variables-and-api-keys) before running those programs.

## Programs and Examples

### 1. Document Similarity - `1_document_similarity.py`

**Objective:** Find the document that is most relevant to a user's query.

This program converts a collection of documents and a query into embeddings, then uses cosine similarity to compare them. It identifies and displays the document with the highest similarity score.

**Concepts covered:**
- OpenAI embeddings
- Document and query embeddings
- Cosine similarity
- Semantic similarity search

**Run:**
```bash
python 1_document_similarity.py
```

### 2. Dynamic Prompt Templates - `2_standalonequery_dynamic.py`

**Objective:** Create an interactive research-paper summarization tool.

The program uses Streamlit to collect a paper title, explanation style, and explanation length. These selections are inserted into a `PromptTemplate` and sent to a chat model to generate a summary. The code also includes an alternative approach using a prompt-model chain.

**Concepts covered:**
- `ChatOpenAI`
- `PromptTemplate`
- Dynamic input variables
- Template validation
- Model invocation and prompt-model chaining
- Streamlit interface

**Run:**
```bash
streamlit run 2_standalonequery_dynamic.py
```

### 3. Chat with Message History - `3_listofmessages_static.py`

**Objective:** Build a simple command-line chatbot that keeps the conversation history.

Each user message and model response is added to a message list. The complete list is sent to the model on each turn, allowing it to use earlier messages as context.

**Concepts covered:**
- `SystemMessage`
- `HumanMessage`
- `AIMessage`
- Maintaining conversation history
- Model invocation

**Run:**
```bash
python 3_listofmessages_static.py
```

Enter `exit` to end the conversation.

### 4. Dynamic Chat Prompts with Message History - `4_listofmessages_dynamic.py`

**Objective:** Build a chat prompt that includes previous conversation context along with the current user input.

The program uses `ChatPromptTemplate` to define the prompt structure and `MessagesPlaceholder` to specify where chat history should be included. It reads the saved history from `4_chat_history_example.txt` and supplies it to the prompt when creating the final prompt.

**Concepts covered:**
- `ChatPromptTemplate`
- `MessagesPlaceholder`
- Dynamic prompt inputs
- Loading saved conversation history

**Run:**
```bash
python 4_listofmessages_dynamic.py
```

Keep `4_chat_history_example.txt` in the repository's root directory so the program can read it.

### 5. Understanding Chains - `5_langchain_chains_runnablespart1.ipynb`

**Objective:** Understand how a prompt template and a language model can be combined into a single workflow.

This notebook starts with simple, custom prompt-template and language-model classes. It then introduces a simplified `LLMChain`-style class that handles prompt formatting and model execution through one `run()` call.

**Concepts covered:**
- Prompt templates and input variables
- Language-model execution
- Combining components into a chain
- Sequential execution through a single method

Open the notebook in Jupyter Notebook or JupyterLab and run the cells in order.

### 6. Understanding Runnables - `5_langchain_chains_runnablespart2.ipynb`

**Objective:** Understand the Runnable idea and how components can be connected so that one component's output becomes the next component's input.

The notebook builds simplified runnable components, including a prompt template, a mock language model, an output parser, and a connector. It then uses these components to create a sequential workflow and connect multiple chains.

**Concepts covered:**
- Runnable interface and `invoke()`
- Prompt, model, and parser components
- Sequential component execution
- Connecting multiple chains
- Passing outputs between components

The notebook uses custom educational classes to demonstrate these ideas; it is not an implementation of LangChain's internal Runnable classes.

Open the notebook in Jupyter Notebook or JupyterLab and run the cells in order.

## Environment Variables and API Keys

The repository does not include an `.env` file because API keys are private. If you want to run the examples that use OpenAI, create your own `.env` file in the repository's root directory.

Add your key like this:

```env
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your own API key. The programs that call `load_dotenv()` will load the variable from this file.

**Keep your API key safe:**
- Do not upload your `.env` file to GitHub.
- Add `.env` to your `.gitignore` file.
- Never paste your API key into source code or commit history.
- Do not share your key publicly.

## Further Learning

These programs cover only a few selected LangChain components. There is much more to explore, including:

- **Models:** Different model types and integrations.
- **Prompts:** Different prompt types and templates.
- **Structured Output and Output Parsers:** Getting model responses in specific formats and parsing them.
- **Chains:** Different chain types and ways to combine components.
- **Runnables:** Different Runnable types and composition patterns.
- **Memory:** Managing and using conversation context.
- **Indexes:** Working with documents and making information easier to retrieve.
- **Agents:** Building applications that can choose and use tools.

This repository is a growing collection of hands-on examples. More concepts and implementations can be added as the learning journey continues.

