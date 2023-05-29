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
import numpy as np
from numpy.linalg import norm
from supabase import create_client, Client
import openai




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

        data = supabase.table("embeddings").select("*").eq("url", data_from).execute()

        if (len(data.data)==0):
            st.write('Not found')
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

            context = []

            response = openai.Embedding.create(
                input=question,
                model="text-embedding-ada-002"
                )
            question_embedding = response['data'][0]['embedding']

            for doc in texts:
                response = openai.Embedding.create(
                    input=doc.page_content,
                    model="text-embedding-ada-002"
                )
                embedding = response['data'][0]['embedding']
                data, count = supabase.table('embeddings').insert({"url": data_from, "embedding": embedding, "Content":doc.page_content}).execute()

                if(cosine_similarity(embedding,question_embedding) > 0.85):
                    context.append(doc.page_content) 
                    
        else:
            st.write('Found')
            response = openai.Embedding.create(
                input=question,
                model="text-embedding-ada-002"
                )
            question_embedding = response['data'][0]['embedding']

            context = []

            for l in data.data:
                change = l['embedding']
                change_split = change.split(',')
                change_split[-1] = change_split[-1][:-1]
                change_split[0] = change_split[0][1:]
                
                change_split_int = []
                for i in change_split:
                    change_split_int.append(float(i))

                if (cosine_similarity(change_split_int,question_embedding) > 0.85):
                    context.append(l['Content'])

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