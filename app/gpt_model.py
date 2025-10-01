import json
import os
import re
import requests
from openai import OpenAI
import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

# Загружаем переменные окружения
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


class GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        if not openai_api_key:
            raise ValueError("API ключ OpenAI не найден! Проверьте .env")
        self.client = OpenAI(api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.model = model
        self.search_index = None
        self.log = ''

    def load_search_indexes(self, url):
        match_ = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
        if not match_:
            raise ValueError("Неверный Google Docs URL")
        doc_id = match_.group(1)
        response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
        response.raise_for_status()
        text = response.text
        return self.create_embedding(text)

    def num_tokens_from_string(self, string):
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(string))

    def create_embedding(self, data):
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1024, chunk_overlap=0)
        chunks = [Document(page_content=c, metadata={}) for c in splitter.split_text(data)]
        self.search_index = Chroma.from_documents(chunks, self.embeddings)
        self.log += f"Данные загружены в векторную базу\n"
        return self.search_index

    def answer_index(self, system, topic, temp=1):
        if not self.search_index:
            self.log += "Векторная база не создана!\n"
            return ''
        docs = self.search_index.similarity_search(topic, k=5)
        message_content = "\n".join([f"Отрывок {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        messages = [
            {"role": "system", "content": system + message_content},
            {"role": "user", "content": topic}
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp
        )
        return completion.choices[0].message.content