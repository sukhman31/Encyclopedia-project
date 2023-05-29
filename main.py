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
import openai
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

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

        ans = db.similarity_search(question,k=5)

        context = ""
        for i in range(2):
            context += ans[i].page_content
            context += "\n"

        template="""Your only source of knowledge is the following context. Please use only the following context to provide a suitable answer. You have to underline the proper nouns, important words, names of places, art references in the answer and return the
            answer in markdown. The answer can only be in markdown.  For the underlined words, it is necessary to add hyperlink in the format : 'https://www.britannica.com/search?query=word'. You have to identify
            atlease three such words or phrases and you have to prove the hyperlink, otherwise the output won't be good.
            Context: {context}
            Question: {question}
            """
        
        prompt = PromptTemplate(template=template, input_variables=["context","question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        response = openai.Embedding.create(
            input=question,
            model="text-embedding-ada-002"
            )
        embedding = response['data'][0]['embedding']
        images = data_page.find_all("img")
        maxi = 0
        right_image = None
        for img in images:
            try:
                text = img['alt']
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                text_embedding = response['data'][0]['embedding']
                if cosine_similarity(embedding,text_embedding) > maxi:
                    right_image = img
                    maxi = cosine_similarity(embedding,text_embedding)
            except:
                continue

        st.markdown(llm_chain.predict(context = context, question = question))
        if right_image != None:
            st.markdown(str(right_image), unsafe_allow_html=True)