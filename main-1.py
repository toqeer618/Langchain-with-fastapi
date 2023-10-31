

from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory 
                                                  )
from langchain.chains import VectorDBQA
from langchain import OpenAI, LLMChain, PromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
import openai
import os
from langchain.vectorstores import Chroma

from langchain.document_loaders import TextLoader
from langchain.agents import initialize_agent
from langchain.agents import load_tools

os.environ["OPENAI_API_KEY"]="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"
openai.api_key = os.getenv("OPENAI_API_KEY")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



llm = OpenAI(
    openai_api_key=OPENAI_API_KEY,  # platform.openai.com
    temperature=0,
    model_name="text-davinci-003"
)
     
tools = load_tools(
    ['llm-math'],
    llm=llm
)

global zero_shot_agent 
zero_shot_agent = initialize_agent(
agent="zero-shot-react-description",
tools=tools,
llm=llm,
verbose=True,
max_iterations=2
)

template = """You are math agent after querying from the math agent and docs search with no response we came to you so answer the querY
            Human: {human_input}
            Assistant:"""


prompt = PromptTemplate(
        input_variables=["human_input"], 
        template=template
            )
global chatgpt_chain
chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0.8), 
    prompt=prompt, 
    verbose=True, 
)




def load_document(fileName):
    loader = TextLoader(fileName)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    global text_search
    text_search = Chroma.from_documents(texts, embeddings)
    global math_chain
    math_chain = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), chain_type="stuff", vectorstore=text_search)

    return "File uploaded and vectorized"




def textquery(query: str):

    result = math_chain.run(query)
    if "I don't know" not in result:
      return {"answer":result }
    if "I don't know" in result:
        try:
            result = zero_shot_agent(query)
            print(result)
        except:
            result = chatgpt_chain.predict(human_input = query)
            return {"answer":result }

    if "Agent stopped due to iteration limit" in result['output']:
        result = chatgpt_chain.predict(human_input = query)


    return result

if __name__ == "__main__":
    print("File is loading.....")
    msg = load_document('text.txt')
    print(msg)
    clear = lambda: os.system('clear')
    cond = 'C'
    while cond=='C':
        clear()
        query = input("Write your query here and press enter and wait for response: ")
        resutlt = textquery(query)
        print(resutlt)

        cond = input("\n\n Ente 'c' or 'C' to continue chatting any other key to exit")

        

