import json
import os
import time
from typing import List

import html2text
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from selenium_recursive_loader import SeleniumRecursiveLoader, default_page_ready_check


class PageExtractor:
    @staticmethod
    def extract(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        for tag in ["header", "footer", "script", "style"]:
            for elem in soup.find_all(tag):
                elem.decompose()

        return html2text.html2text(str(soup)).strip()


class DocumentStorage:
    def __init__(self, json_filepath: str):
        self.json_filepath = json_filepath

    def save_documents(self, documents: List[Document]):
        with open(self.json_filepath, "w", encoding="utf-8") as f:
            json.dump([doc.dict() for doc in documents], f, ensure_ascii=False, indent=4)

    def load_documents(self) -> List[Document]:
        with open(self.json_filepath, "r", encoding="utf-8") as f:
            documents_dicts = json.load(f)
            return [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in documents_dicts]

    def documents_exist(self) -> bool:
        return os.path.exists(self.json_filepath)


class DocumentFetcher:
    def __init__(self, url: str, base_url: str, exclude_urls: List[str], page_extractor):
        self.url = url
        self.base_url = base_url
        self.exclude_urls = exclude_urls
        self.page_extractor = page_extractor

    def fetch_documents(self) -> List[Document]:
        initial_links = [self.url]
        loader = SeleniumRecursiveLoader(
            urls=initial_links,
            base_url=self.base_url,
            page_extractor=self.page_extractor,
            exclude_urls=self.exclude_urls,
            headless=False,
            page_ready_check=default_page_ready_check,
        )
        return loader.load()


class DocumentProcessor:
    PERSIST_DIRECTORY = "./chroma_db"

    def __init__(self, text_splitter, document_storage: DocumentStorage, document_fetcher: DocumentFetcher):
        self.text_splitter = text_splitter
        self.document_storage = document_storage
        self.document_fetcher = document_fetcher

    def fetch_and_process_documents(self) -> List[Document]:
        if self.document_storage.documents_exist():
            print(f"Loading documents from {self.document_storage.json_filepath}")
            return self.document_storage.load_documents()
        else:
            print(f"Fetching and processing documents from {self.document_fetcher.url}")
            docs = self.document_fetcher.fetch_documents()
            self.document_storage.save_documents(docs)
            return docs

    def process_documents_to_vectorstore(
        self,
        docs: List[Document],
        persist_directory: str,
        max_retries: int = 10,
        retry_delay: int = 60,
    ):
        for splitted_document in docs:
            for attempt in range(max_retries):
                try:
                    vectorstore = Chroma.from_documents(
                        documents=[splitted_document],
                        embedding=OpenAIEmbeddings(
                            retry_max_seconds=120,
                            retry_min_seconds=retry_delay,
                            max_retries=max_retries,
                            show_progress_bar=True,
                        ),
                        persist_directory=persist_directory,
                    )
                    print(f"Document {splitted_document.metadata['source']} successfully processed.")
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Giving up.")


# Usage example:
if __name__ == "__main__":
    url = "https://vika-it.rtuitlab.dev/"
    json_filepath = "documents.json"
    base_url = "https://vika-it.rtuitlab.dev"
    exclude_urls = [
        "https://priem.mirea.ru/lk/",
        "https://priem.mirea.ru/school/",
        "https://priem.mirea.ru/open-doors",
        "https://priem.mirea.ru/dod-online/",
        "https://priem.mirea.ru/event",
        "https://priem.mirea.ru/about/specscholarship/",
        "https://priem.mirea.ru/guide-direction",
        "https://priem.mirea.ru/olympiad-page",
        "https://priem.mirea.ru/olymp-landing/",
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=2000, chunk_overlap=200
    )

    document_storage = DocumentStorage(json_filepath)
    page_extractor = PageExtractor.extract
    document_fetcher = DocumentFetcher(url, base_url, exclude_urls, page_extractor)
    processor = DocumentProcessor(text_splitter, document_storage, document_fetcher)

    documents = processor.fetch_and_process_documents()
    documents = processor.text_splitter.split_documents(documents)
    processor.process_documents_to_vectorstore(documents, DocumentProcessor.PERSIST_DIRECTORY)
