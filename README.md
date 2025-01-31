# PDF Documents Chatbot

## Overview

This application allows users to chat with PDF documents that have been uploaded to a Pinecone index. Users can ask questions about the content of the documents, and the application will provide relevant answers based on the indexed data.

## Technologies Used

- **Python 3.9.7**: The core programming language used for developing the application.
- **Langchain**: Utilized for natural language processing and understanding.
- **Pinecone**: Serves as the vector database to index and search document content efficiently.
- **Chainlit**: Used to create the frontend interface of the application.
- **OpenAI**: Leveraged as the Large Language Model (LLM) to enhance the natural language understanding and response generation capabilities of the application.

## How It Works

1. **Upload PDF Documents**: Users upload PDF documents to the application.
2. **Indexing**: The content of the uploaded documents is processed and indexed using Pinecone.
3. **Chat Interface**: Users can interact with the application through a chat interface created with Chainlit.
4. **Ask Questions**: Users can ask questions about the content of the documents.
5. **Receive Answers**: The application processes the questions using Langchain and OpenAI, and retrieves relevant information from the Pinecone index to provide accurate answers.

This application is designed to make it easy to extract and interact with information from PDF documents, providing a seamless and efficient user experience.

## Dependencies

Create a virtual environment first:
python3 -m venv .venv && source .venv/bin/activate
Then, install the required packages:
To run this project, you need to install the following dependencies:
pip3 install pinecone-client
pip3 install langchain
pip3 install langchain-core
pip3 install langchain-community
pip3 install langchain-openai
pip3 install langchain-pinecone
pip3 install langchain-text-splitters
pip3 install pypdf
pip3 install chainlit
pip3 install PyPDF2

## How to run it

Run with the following command within the main folder (cd main):
chainlit run chatbot-docs-app.py -w --port 8080
chainlit run chatbot-upload-pdf-app.py -w --port 8080
