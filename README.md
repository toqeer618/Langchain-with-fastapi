## 1 - Install all the required files with pip3/pip 
`pip3 install -r requirements.txt`

## 2- This FAST API has multiple functionalities using langchain

###  2.1- Diana chatbot 
    `it is personalized chatbot based on prompt and response kept the history upto last 2 messages `
### 2.2-  Story Teller chatbot
    ` A story telling bot `

You can create many more bots by just changing the prompt passing to bot

### 3- chat with pdf 
    ` 1. Document loaded as pdf from any directory and loaded to in the chunks
    of 1000`
    ` 2. vectorized using openAI ada-002 and saved the vectors to chromadb`
    ` 3. Then question is asked over the document and answerd from the vectorized db.`
### 4- Chat with URL 
    ` Pass the URL and ask the question and got the answer that is in the document ` 
## 5- Running the app on local host 
`uvicorn main:app --reload` 
