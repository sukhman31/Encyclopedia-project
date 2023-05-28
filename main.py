import streamlit as st
import os
from langchain.vectorstores import DeepLake
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import requests
from bs4 import BeautifulSoup
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
import requests
import io
from PIL import Image

with st.form('input'):
    question = st.text_input('Enter your question')

    if st.form_submit_button('Submit'):
        llm = OpenAI(temperature=0)
        template="""In only two words, please specify what the question is about. 
        The question is {question}.
        """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        topic = llm_chain.predict(question = question)

        url_req = 'https://www.britannica.com/search?query='+topic
        request_results = requests.get(url_req)
        web_page = BeautifulSoup(request_results.text, "html.parser")
        div_text= web_page.find("class",{"class":"RESULT-1"})
        for elm in web_page.select(".RESULT-1"):
            link = elm.select('a')
            subdiv = link[0]['href']
        
        data_from = 'https://www.britannica.com' + subdiv
        request_data = requests.get(data_from)
        data_page = BeautifulSoup(request_data.text, "html.parser")
        paragraphs = []
        for para in data_page.select('p'):
            paragraphs.append(str(para))

        CLEANR = re.compile('<.*?>') 

        def cleanhtml(raw_html):
            cleantext = re.sub(CLEANR, '', raw_html)
            return cleantext 
        
        clean_paragraphs = []
        for para in paragraphs:
            clean_paragraphs.append(cleanhtml(para))
        
        text_splitter = CharacterTextSplitter(separator = '.',chunk_size=200, chunk_overlap=0)
        texts = text_splitter.create_documents(clean_paragraphs)

        embeddings = OpenAIEmbeddings()
        db = DeepLake.from_documents(texts, embeddings)

        ans = db.similarity_search(question,k=2)

        context = ""
        for i in range(2):
            context += ans[i].page_content
            context += "\n"

        template="""Your only source of knowledge is the following context. Please use only the following context to provide a suitable answer. You have to underline the proper nouns in the answer and return the
            answer in markdown. The answer can only be in markdown.  For the underlined words, add hyperlink in the format : 'https://www.britannica.com/search?query=word'
            Context: {context}
            Question: {question}
            """
        
        prompt = PromptTemplate(template=template, input_variables=["context","question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)


        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Authorization": f"Bearer {os.environ['BEARER_TOKEN']}"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content
        image_bytes = query({
            "inputs": topic,
        })

        image = Image.open(io.BytesIO(image_bytes))

        st.markdown(llm_chain.predict(context = context, question = question))
        st.image(image)