# CSV Agent

## Overview
The **CSV Agent** is an AI-powered assistant that helps analyze and interact with CSV data. Using Large Language Models (LLMs), it can answer questions, extract insights, and perform operations on CSV files uploaded by users.

## Features
- Parses and reads CSV files dynamically.
- Uses LLMs to analyze data and provide insights.
- Supports various queries on structured data.
- Handles large datasets efficiently.



## Usage

### Uploading a CSV File
- The application provides a file uploader for CSV files.
- Once uploaded, the agent will parse the file and prepare it for analysis.


```

### Example Queries
- "How many unique values are in column X?"
- "What is the maximum value in column Y?"
- "Find all rows where column Z is greater than 100."

## Configuration
Modify `config.py` to customize the agent's behavior, such as setting:
```python
LLM_MODEL = "deepseek"
```

## Dependencies
- Python 3.8+
- Pandas
- LangChain
- OpenAI API (or other LLM providers)




