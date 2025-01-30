from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

import os
import openai
import chainlit as cl

# Set your API keys for OpenAI and Pinecone
openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize OpenAI Embeddings using LangChain
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Specify which embedding model

# Connect to the Pinecone index using LangChain's Pinecone wrapper
pinecone_index_name = "index1536"
vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)

# Initialize GPT-4 with OpenAI
llm = ChatOpenAI( model="gpt-4", openai_api_key=openai.api_key, temperature=0.7 )



# Define Prompt Template
prompt_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.  
Example of your response should be as follows:      
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt

def create_retrieval_qa_chain(llm, prompt, db):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.

    This function initializes a RetrievalQA object with a specific chain type and configurations,
    and returns this QA chain. The retriever is set up to return the top 3 results (k=1).

    Args:
        llm (any): The language model to be used in the RetrievalQA.
        prompt (str): The prompt to be used in the chain type.
        db (any): The database to be used as the retriever.

    Returns:
        RetrievalQA: The initialized QA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    print("qa_chain:", qa_chain)
    return qa_chain

def create_retrieval_qa_bot(
    model_name="text-embedding-ada-002",
    index_name=pinecone_index_name,
):
    try:
        embeddings = OpenAIEmbeddings(model=model_name)  # type: ignore
    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name {model_name}: {str(e)}"
        )
   
    db = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

    qa_prompt = (
        set_custom_prompt()
    )  # Assuming this function exists and works as expected

    try:
        qa = create_retrieval_qa_chain(
            llm=llm, prompt=qa_prompt, db=db
        )  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa


# Using chainlit to init the chat bot
@cl.on_chat_start
async def initialize_bot():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    # Create a new instance of the retrieval QA bot
    qa_bot = create_retrieval_qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to the AMD chatbot!"
    )
    await welcome_message.update()

    # Send a welcome message
    await cl.Message("Ask me anything related to documentation for AMD processors, accelerators, graphics, and other products.").send()

    # Store the bot instance in the user's session
    cl.user_session.set("qa_bot", qa_bot)

@cl.on_message
async def process_chat_message(message: cl.Message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    qa_chain = cl.user_session.get("qa_bot")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True
    response = await qa_chain.acall(message.content, callbacks=[callback_handler])
    bot_answer = response["result"]
    
    source_documents = response["source_documents"]

    if source_documents:
        bot_answer += f"\nSources:" + str(source_documents)
    else:
        bot_answer += "\nNo sources found"

    await cl.Message(content=bot_answer).send()

""" Using standard output

# Define the retrieval mechanism
retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Retrieve top-1 relevant documents

# Create LLM Chain
llm_chain = prompt_template | llm | StrOutputParser()

# Retrieve documents
query = "
What are the gen processors include hardened security features that help protect against malicious users, devices, and code?
"
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])
    
# Run LLM chain with the retrieved context
answer = llm_chain.invoke({"context": context, "question": query})

# Output the Answer and Sources
print("Answer:", answer)
"""