from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.prompt_selector import ConditionalPromptSelector
from enum import Enum

class PromptSelector(Enum): #TemplateSelector
    defualt_prompt = PromptTemplate( input_variables=['question'], template='<s>[INST] {question} [/INST]' )
    defualt_rag_template = PromptTemplate( input_variables=['query'], template='<s>[INST] {question} [/INST]' )

class MyLLM:
    def __init__(
        self,
        model_path : str,
    ):
        self.llm = self.__load_llm(model_path)
        self.prompt_selector = self.__init_prompt_selector()
        self.prompt = None
        self.llm_retriever = None
        
    def chat(self, sentence):
        self.llm(sentence)
    
    def chat_prompt_qa(self,question):
        if self.prompt is None:
            raise 'Please set a prompt'
        llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        llm_chain.invoke({"question": question})
    
    def chat_rag(self,query):
        if self.llm_retriever is None:
            print('Please set a vector database')
            return
        self.llm_retriever.invoke(query)
    
    def set_prompt(self, prompt):
        pass

    def add_prompt(self, keyword, template):
        pass
    
    def set_retriever(self,vectordb):
        retriever = vectordb.as_retriever()
        self.llm_retriever = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=retriever, 
            verbose=True
        )

    def get_model_info(self):
        pass
    
    def __load_llm(self, model_path):
        return LlamaCpp(
            model_path=model_path,
            n_gpu_layers=100,
            n_batch=512,
            n_ctx=2048,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
        )

    def __init_prompt_selector(self):
        self.prompt_selector_conditionals = [] 
        return ConditionalPromptSelector( default_prompt=PromptSelector.defualt_prompt.value, conditionals=self.prompt_selector_conditionals )
    
    def __update_prompt_selector(self,keyword_target):
        def __check_function(keyword_input):
            return keyword_input == keyword_target
        self.prompt_selector_conditionals.append()
    
class TestLLM:
    def __init__(
        self,
        model_path : str,
    ):
        self.llm = self.__load_llm(model_path)
        self.llm_chain = LLMChain(prompt=PromptSelector.defualt_prompt.value, llm=self.llm)
        
    def chat(self, sentence):
        self.llm(sentence)
    
    def chat_prompt_qa(self,question): #template
        self.llm_chain.invoke({"question": question})
    
    def chat_rag(self,query,vectordb):
        retriever = vectordb.as_retriever()
        self.llm_retriever = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=retriever, 
            verbose=True
        )
        self.llm_retriever.invoke(query)
    
    def chat_promptrag(self,query,vectordb):
        retriever = vectordb.as_retriever()
        self.llm_retriever = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type = 'stuff',
            retriever=retriever,
            chain_type_kwargs={"prompt": PromptSelector.defualt_rag_template.value},
            verbose=True
        )
        self.llm_retriever.invoke(query)


    def __load_llm(self, model_path):
        return LlamaCpp(
            model_path=model_path,
            n_gpu_layers=100,
            n_batch=512,
            n_ctx=2048,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
        )