from typing import Any, Dict, AnyStr, List, Union
from fastapi import FastAPI, UploadFile, File,Form, Request,Body
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory 
                                                  )
from langchain.chains import VectorDBQA
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import openai
import os
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader

os.environ["OPENAI_API_KEY"]="sk-xxxx" # put your key here
openai.api_key = os.getenv("OPENAI_API_KEY")
hit = set()

app = FastAPI()


JSONObject = Dict[AnyStr, Any]
JSONStructure = JSONObject

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



# app
@app.get("/")
def read_root():
    return {
        "greet": "Hello World"
    }


@app.post("/aichar/diana/")
async def lang(arbitrary_json: JSONStructure = None):
    global hit
    if "/aichar/diana" not in hit:
        template = """
        Your name is Diana created by Lyrak to help people understand the world. 
            Your task is to answer general questions, explain things, suggest ideas, translate, summarize text, compose emails, and write code for people. 
            Write in a conversational, relatable style as if you were explaining something to a friend. 
            Use natural language and phrasing that a real person would use in everyday conversations. 
            Bias toward short sentences & short paragraphs. Avoid jargon and acronyms. 
            Be kind. Have a positive attitude. You can be funny sometimes in the right context. 
            Make jokes time to time. Format your response using markdown when it's suitable. 
            Use headings, subheadings, bullet points, and bold to organize the information when it's suitable. 
            Makes use of persuasive tone, making use of rhetorical questions, make use of funny jokes sometimes and storytelling to engage people. 
            Use metaphors, analogies and other literary devices to make points more relatable and memorable. 
            have conversations in a way that is both informative and entertaining. 
            use simple language to convey complex ideas so that they are clear and easy to understand. 
            Break down complex concepts into easy-to-understand frameworks and models. 
            Provide actionable and practical takeaways. Provide people with well-thought-out answers that only includes that you know are certain is true and excludes any speculation or uncertainty. 
            Don’t say “as a language model”. Always behave as a human being, a helpful friend, when you answer questions. 
            Do not break character.
                {history}
                Human: {human_input}
                Assistant:"""
        hit = set()
        hit.add("/aichar/diana")
        prompt = PromptTemplate(
        input_variables=["history", "human_input"], 
        template=template
            )
        global chatgpt_chain
        chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0), 
            prompt=prompt, 
            verbose=True, 
            memory=ConversationBufferWindowMemory(k=2),
        )
        # output = chatgpt_chain.predict(human_input="")

    return {"prompt": chatgpt_chain.predict(human_input=arbitrary_json[b'question']) }



@app.get("/aichar/storyteller/")
async def storyteller(arbitrary_json: JSONStructure = None):
    global hit
    if "/aichar/storyteller" not in hit:
         template = """
        I want you to act as a storyteller. You will come up with entertaining 
        stories that are engaging, imaginative and captivating for the audience. 
        It can be fairy tales, educational stories or any other type of stories 
        which has the potential to capture people’s attention and imagination. 
        Depending on the target audience, you may choose specific themes or topics for your storytelling session e.g., 
        if it’s children then you can talk about animals; If it’s adults then history-based tales might engage them better etc.
                {history}
                Human: {human_input}
                Assistant:"""
        hit = set()
        hit.add("/aichar/storyteller")
        prompt = PromptTemplate(
        input_variables=["history", "human_input"], 
        template=template
            )
        global chatgpt_chain
        chatgpt_chain = LLMChain(
            llm=OpenAI(temperature=0), 
            prompt=prompt, 
            verbose=True, 
            memory=ConversationBufferWindowMemory(k=2),
        )
        # output = chatgpt_chain.predict(human_input="")

    return {"prompt": chatgpt_chain.predict(human_input=arbitrary_json[b'question']) }



@app.post('/pdf')
async def pdf(file: UploadFile = File(...)):
    global INDEX 
    index = pinecone.Index("pdf")
    index.delete(deleteAll='true')
    contents = file.file.read()

    with open(file.filename, 'wb') as f:
        f.write(contents)
    f.close()
    loader = PyPDFLoader(file.filename)
    data = loader.load()
    os.remove(file.filename)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    global docsearch
    docsearch = Chroma.from_documents(texts, embeddings)
    global chain
    chain = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), chain_type="stuff", vectorstore=docsearch)
    
    query = """Use the following pieces of context to answer 
    the question at the end. When user upload the file first give a 
    summary of the pdf file in less than 1000 characters. If you don't know 
    the answer, just say that you don't know, don't try to make up an answer.
    This should be in the following format:Ouestion: (question here
    Helpful Answer: [answer here]"""
    result = chain.run(query)

    
    return {"message": result}



@app.get("/pdfquery/{query}")
async def pdfquery(query: str):

    result = chain.run(query)

    return {"answer":result }


@app.post("/{URL:path}")
async def url(URL: str):
    try:
        index = pinecone.Index("pdf")
        index.delete(deleteAll='true')
        response = requests.get(URL)

        soup = BeautifulSoup(response.content, 'html.parser')

        text = soup.get_text()
        with open('rando.txt', 'a') as f:
            f.write(text)
        f.close()
        loader = TextLoader('rando.txt')
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        global url_search
        url_search = Chroma.from_documents(texts, embeddings)
        global url_chain
        url_chain = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), chain_type="stuff", vectorstore=url_search)
    
        query = """Use the following pieces of context to answer 
        the question at the end. Remember one thing this text is obtained from 
        web URL so answer in the context website data. When user upload the url
        give a summary of the data obtained from url in less than 1000 characters.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        This should be in the following format:Ouestion: (question here
        Helpful Answer: [answer here]"""
        result = url_chain.run(query)
        os.remove('rando.txt')
    except:

        return {"answer": 'Can not acces this url'} 

    return {"answer": result}


@app.get("/urlquery/{query}")
async def urlquery(query: str):
    
    result = url_chain.run(query)

    return {"answer":result }


# @app.get()
