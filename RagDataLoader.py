from langchain_community.document_loaders import PyMuPDFLoader, BSHTMLLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from enum import Enum
import os

class TextSplitterSelector(Enum):
    Chroma_RecursiveCharacterTextSplitter = 1
    Chroma_CharacterTextSplitter = 2

class TokenizerSelector(Enum):
    voyage_large_2_instruct = 'voyage-large-2-instruct'
    bert_base_uncased = 'google-bert/bert-base-uncased' #ok
    bert_large_uncased = 'google-bert/bert-large-uncased' #ok
    roberta_base = 'FacebookAI/roberta-base' #ok
    llama_chinese_81M = 'p208p2002/llama-chinese-81M' #ok
    Llama_2_7b_chat_hf = 'meta-llama/Llama-2-7b-chat-hf'
    all_MiniLM_L6_v2 = 'sentence-transformers/all-MiniLM-L6-v2' #ok
    

class RAG_DataLoader:
    def __init__(
        self,
        tokenize_model : TokenizerSelector = TokenizerSelector.all_MiniLM_L6_v2,
        chunk_size : int = 100,
        chunk_overlap : int = 5,
        textsplitter : TextSplitterSelector = TextSplitterSelector.Chroma_RecursiveCharacterTextSplitter,
        db_path : str = 'db',
        device : str = 'cpu',
    ):
        self.embedding = self.__load_tokenizer(tokenize_model=tokenize_model,device=device)
        self.db_path = db_path
        self._chroma_db = self.__load_chroma_db(db_path)
        self.text_splitter = self.__load_textsplitter(textsplitter,chunk_size,chunk_overlap)
        self.all_splits = []
    
    def add(self,file_path):
        splits = None
        try:
            if file_path.endswith('.pdf'):
                splits = self._pdf_loader(file_path=file_path)
            elif file_path.endswith('.csv'):
                splits = self._csv_loader(file_path=file_path)
            elif file_path.endswith('.json'):
                splits = self._json_loader(file_path=file_path)
            elif file_path.endswith('.html'):
                splits = self._html_loader(file_path=file_path)
            elif file_path.endswith('.markdown'):
                splits = self._markdown_loader(file_path=file_path)
            if splits is None:
                raise ValueError(f'Unavailable file format: {file_path}')
            self.all_splits.extend(splits)
                
        except Exception as e:
            raise f"Error processing file {file_path}: {e}"   

    def save(self):
        if not self.all_splits:
            raise 'No Document is added'
        self._chroma_db.add_documents(documents=self.all_splits)
        self._chroma_db.persist()
        self.all_splits.clear()

    def get_chromadb(self):
        return self._chroma_db

    def __load_tokenizer(self, tokenize_model, device):
        return HuggingFaceEmbeddings( model_name=tokenize_model.value, model_kwargs={'device': device} )
    
    def __load_chroma_db(self,db_path):
        return Chroma(persist_directory=db_path,embedding_function=self.embedding)
    
    def __load_textsplitter(self,text_splitter,chunk_size,chunk_overlap):
        if text_splitter.name == 'Chroma_RecursiveCharacterTextSplitter':
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if text_splitter.name == 'Chroma_CharacterTextSplitter':
            return CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        raise f'Invalid text splitter: {text_splitter}'

    def _pdf_loader(self,file_path):
        data = PyMuPDFLoader(file_path=file_path).load()
        return self.text_splitter.split_documents(data)
        
    def _csv_loader(self, file_path, encoding='utf-8'):
        data = CSVLoader(file_path=file_path,encoding=encoding).load()
        return self.text_splitter.split_documents(data)
    
    def _json_loader(self,file_path):
        data = JSONLoader(file_path=file_path).load()
        return self.text_splitter.split_documents(data)
    
    def _html_loader(self,file_path):
        data = BSHTMLLoader(file_path=file_path).load()
        return self.text_splitter.split_documents(data)
                
    def _markdown_loader(self,file_path):
        data = UnstructuredMarkdownLoader(file_path=file_path, mode='elements').load()
        headers_to_split_on = [ ('#','Header 1'), ('##','Header 2') ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_splits = markdown_splitter.split_text(data)
        return self.text_splitter.split_documents(md_splits)    