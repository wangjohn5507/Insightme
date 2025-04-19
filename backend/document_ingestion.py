from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import bs4
import os

from dotenv import load_dotenv
load_dotenv()

from consts import INDEX_NAME
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def load_documents():
    loader = DirectoryLoader(
        "data",
        glob="**/*.pdf",  # Match all PDFs in directory and subdirectories
        loader_cls=PyPDFLoader,
    )
    return loader.load()

def load_web_documents():
    url1 = "https://wangjohn5507.github.io/post/project1/"  
    url2 = "https://wangjohn5507.github.io/post/project2/"
    url3 = "https://wangjohn5507.github.io/post/project3/"
    url4 = "https://wangjohn5507.github.io/post/project4/"
    url5 = "https://wangjohn5507.github.io/post/project5/"
    url6 = "https://wangjohn5507.github.io/post/project6/"
    bs4_strainer = bs4.SoupStrainer(class_=('prose prose-slate lg:prose-xl dark:prose-invert'))
    loader = WebBaseLoader(
        web_path=(url1, url2, url3, url4, url5, url6),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def store_documents(documents):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        print(f"Existing indexes: {existing_indexes}")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="my_namespace")
        vector_store.delete(delete_all=True)
        vector_store.add_documents(documents)
    except Exception as e:
        print(f"Error storing documents: {e}")

if __name__ == "__main__":
    print("Loading documents...")
    documents = []
    documents.extend(load_documents())
    documents.extend(load_web_documents())
    print(f"{len(documents)} Documents loaded.")
    print(f"Total characters: {sum([len(documents[doc].page_content) for doc in range(len(documents))])}")
    split_documents = split_documents(documents)
    print(f"Split blog post into {len(split_documents)} sub-documents.")
    store_documents(split_documents)
    print("Documents stored in Pinecone.")