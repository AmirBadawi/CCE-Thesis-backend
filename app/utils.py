# General
import os
import re
import asyncio
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse
import json
import uuid
import requests 
import re
import textwrap
from datetime import datetime
from urllib.parse import urljoin, quote


# Langchain
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.agents import initialize_agent
from langchain.chains import LLMChain
from langchain_community.chat_models import azure_openai
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import azure_openai
from langchain_community.document_loaders import PyPDFLoader


# Our Modules
from CustomRetriever import CustomRetriever as CustomR


# Azure
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, ComplexField, CorsOptions, SearchableField, VectorSearch, SimilarityAlgorithm
from azure.ai.formrecognizer import DocumentAnalysisClient

# def get_vectorstore():
#     try:
#         embeddings = azure_openai(
#             openai_api_key=os.getenv("OPENAI_API_KEY"),
#             openai_api_base=os.getenv("OPENAI_BASE"),
#             openai_api_version=os.getenv("OPENAI_API_VERSION", "2023-03-15-preview"),
#             openai_api_type=os.getenv("OPENAI_TYPE", "azure"),
#             chunk_size=1,
#             request_timeout=10,
#             max_retries=4,
#             model=os.getenv("OPENAI_ADA_EMBEDDING_MODEL"),
#         )
#         vectorstore = AzureSearch(
#             azure_search_endpoint=os.getenv("AZURE_SEARCH_BASE"),
#             azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
#             index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
#             embedding_function=embeddings.embed_query,
#         )
#         return vectorstore
#     except Exception:
#         raise

def get_vectorstore():
    try:
        embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(deployment=os.getenv("OPENAI_ADA_EMBEDDING_MODEL"),openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_BASE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION", "2023-03-15-preview"),
            openai_api_type=os.getenv("OPENAI_TYPE", "azure"),
            chunk_size=1,
            request_timeout=10,
            max_retries=4,
            model=os.getenv("OPENAI_ADA_EMBEDDING_MODEL"))
        index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME")
        vector_store: AzureSearch = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_BASE"),
            azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
            index_name=index_name,
            embedding_function=embeddings.embed_query,
        )
        return vector_store
    except Exception:
        raise

def get_embeddings(text):
    try:
        embeddings = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_BASE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            openai_api_type=os.getenv("OPENAI_TYPE"),
            chunk_size=1,
            request_timeout=10,
            max_retries=4,
            model=os.getenv("OPENAI_ADA_EMBEDDING_MODEL"),
        )

        embeddings_vector = embeddings.embed_query(text=text)
        return embeddings_vector
    except Exception:
        raise


def get_custom_retriever(query, filename):
    return CustomR(vectorstore=get_vectorstore(), query= query, filename=filename)


def get_chat_llm(temp = 0):
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("OPENAI_BASE"),
        openai_api_version=os.getenv("OPENAI_API_VERSION", "2023-03-15-preview"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_type=os.getenv("OPENAI_TYPE", "azure"),
        deployment_name=os.getenv("OPENAI_MODEL"),
        model=os.getenv("OPENAI_MODEL"),
        verbose=False,
        temperature=temp,
    )


def get_chat_turbo_llm(temp = 0):
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("OPENAI_BASE"),
        openai_api_version=os.getenv("OPENAI_API_VERSION", "2023-03-15-preview"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_type=os.getenv("OPENAI_TYPE", "azure"),
        deployment_name=os.getenv("OPENAI_TURBO_MODEL"),
        model=os.getenv("OPENAI_TURBO_MODEL"),
        verbose=False,
        temperature=temp,
    )


def read_prompt_from_file(file):
    data = ""
    try:
        with open(file, "r", encoding="utf-8") as prompt:
            data = prompt.read()
    except Exception as e:
        print("Exception reading prompt: {0}".format(e))

    return data


def get_prompt(filename):
    prompt_template = "app/prompts/" + filename
    prompt_template = read_prompt_from_file(prompt_template)
    if not prompt_template:
        print("No prompt extracted")
        return "No prompt found"
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["query"])
    return PROMPT


def get_prompt_with_memory(filename):
    prompt_template = "app/prompts/" + filename
    prompt_template = read_prompt_from_file(prompt_template)
    if not prompt_template:
        print("No prompt extracted")
        return "No prompt found"
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["memory", "query"])
    return PROMPT


def get_qabase_prompt():
    prompt_template = os.getenv("QABASE_PROMPT_PATH", "app/prompts/QAbase.txt")

    prompt_template = read_prompt_from_file(prompt_template)
    if not prompt_template:
        print("No prompt extracted")
        return "No prompt found"

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return PROMPT


def get_chain_type_kwargs():
    PROMPT = get_prompt_with_memory("base.txt")

    # return PROMPT
    return {"prompt": PROMPT}


async def transform_query_async(query):
    prompt = get_prompt("transform.txt")
    try:
        llm = get_chat_llm()
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False,
        )
        # print("Running the ll_chain...")
        response = await asyncio.wait_for(
            llm_chain.apredict(query=query), timeout=int(os.getenv("TIMEOUT", 18))
        )
        print("Transformed query ==> " + response)
    except Exception:
        raise
    return response


async def content_filter_async(query, gpt4model=False):
    prompt = get_prompt("content_filter.txt")
    try:
        if gpt4model:
            llm = get_chat_llm()
            llm_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=False,
            )
        else:
            llm = get_chat_turbo_llm()
            llm_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=False,
            )
        # print("Running the ll_chain...")
        response = await asyncio.wait_for(
            llm_chain.apredict(query=query), timeout=int(os.getenv("TIMEOUT", 18))
        )
        print("Content filter result ==> " + response)
        # return response
        if "yes" in str(response).lower():
            raise Exception("content filter")
    except Exception:
        raise


def filter_response(input_str):
    print("Before filtering ==> " + input_str)
    if "Human:" in input_str:
        parts = input_str.split("Human:")
        # Take the first part before "Human:" if it exists
        if len(parts) > 1:
            return parts[0].strip()
    if "Answer:" in input_str:
        parts = input_str.split("Answer:")
        # Take the first part after "Answer:" if it exists
        if len(parts) > 1:
            return parts[1].strip()
    if "AI:" in input_str:
        parts = input_str.split("AI:")
        # Take the first part after "AI:" if it exists
        if len(parts) > 1:
            return parts[1].strip()
    if "Assistant:" in input_str:
        parts = input_str.split("AI:")
        # Take the first part after "AI:" if it exists
        if len(parts) > 1:
            return parts[1].strip()
    return input_str.strip()


def detect_language(text):
    # Remove non-alphanumeric characters and whitespace
    cleaned_text = re.sub(r"[^a-zA-Z؀-ۿ ]", "", text)

    # Define regular expressions for language-specific character ranges
    arabic_pattern = re.compile("[؀-ۿ]")
    french_pattern = re.compile("[àâçéèêëîïôûùüÿœæ]")

    # Count the occurrences of characters from each language in the cleaned text
    arabic_count = len(re.findall(arabic_pattern, cleaned_text))
    french_count = len(re.findall(french_pattern, cleaned_text))

    # Determine the language based on character counts
    if arabic_count > french_count:
        return "Arabic"
    elif french_count > arabic_count:
        return "French"
    else:
        return "English"
    


# KME
    
# Process the given path and adds it to the index
async def add_file_to_index(file_path):
    
    # load_blob_file(file_path)
    print("in add file")
    
    # response = download_file(file_path)
    # print("response: ", response)

    # Check if file exists before processing in local dierectory
    if is_path_exists(file_path):
        
        # TODO: support file based on extension #done
        
        file_exists = file_exists_in_index(file_path)
        print("file_exists", file_exists)
    
        if file_exists:
        
            return Response(content="file already exists: "+file_path, status_code=200)
        
        else:
            raw_text = process_file_by_extension(file_path) #returns a list of json for each pdf page
            # print(raw_text)
            # checks if the extension is supported
            if raw_text is None:
                return {"File Extension not supported"}

            # data, meta, title, description = get_file_details(file_path, raw_text)
            # text_chunks = get_format_text_chunks(file_path, data, meta, title, description)
            
            # if using form recognizer
            data = raw_text["content"]

            # for testing using langchian pdf loader
            # data = raw_text[4].page_content


            text_chunks = get_format_text_chunks(file_path, data)
            documents_to_upload = create_documents_chunks(text_chunks)
            # print(documents_to_upload)
            index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
            print("name", index_name)
            index_exists = await get_azure_index(index_name)
            print("index exists", index_exists)
            
            if index_exists:
                add_document_azure(documents_to_upload)
                # return Response(content="added file: "+file_path, status_code=200)
            else:
                create_azure_search_index(index_name)
                add_document_azure(documents_to_upload)
                # return Response(content="added file: "+file_path, status_code=200)
            json_file_path = change_file_extension(file_path, "json")
            delete_file(json_file_path)
            return Response(content="added file: "+file_path, status_code=200)
    else:
        raise HTTPException(status_code=400, detail=f"The requested file does not exist in the blob storage")


def download_file(filename: str):
    print("down file", filename)
    file_path = os.path.join("C:\\Users\\FCC\\VS Code Projects\\Intelligencia-AI-Demo-Backend\\file", filename)
    return FileResponse(file_path, media_type="application/pdf", filename=filename)


# Loads file from blob storage and downloads it to the local storage
def load_blob_file(file_path):
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv('CONNECTION_STRING'))
    #use the client to connect to the container
    container_client = blob_service_client.get_container_client(os.getenv('CONTAINER_NAME'))

    #get filename and folder path
    filename = os.path.basename(file_path)
    folder_path = os.path.dirname(file_path)
    #create folder path
    if folder_path != "":
        if os.path.exists(folder_path):
            print("The path exists.")
        else:
            print("The path does not exist.")
            os.makedirs(folder_path, exist_ok=True)
            print("path created")
    else:
        folder_path = "/public"
        os.makedirs(folder_path, exist_ok=True)
        print("path created")
    with open(file=file_path, mode="wb") as download_file:
        #download file or with it directly
        download_file.write(container_client.download_blob(file_path).readall())



# Checks if given path exists
def is_path_exists(file_path):
    return os.path.exists(file_path)


def delete_file(file_path):
    if is_path_exists(file_path):
        os.remove(file_path)
        # print(f"File '{file_path}' deleted successfully.")



def read_json_file(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    # print(data)
    return data

def process_pdf(file_path):
    # print("Processing PDF file:", file_path)
    # Load file name as json file for the following:
    json_file_path = change_file_extension(file_path, "json")
    
    # Check if file already loaded locally
    if is_path_exists(json_file_path):
        # print("Reading existing json")
        raw_text = read_json_file(json_file_path)
    # load existing file and return
    
    # if file not loaded
    else:
        # print("Form Recognizer")
        # use form recognizer to load into a json file with same naming scheme (PR/file.json)
        raw_text = analyze_general_documents(file_path)
        
    return raw_text


def get_pdf_documents(pdf):
    loader = PyPDFLoader(pdf)
    pages = loader.load()
    return pages


def process_unknown(file_path):
    print("Unknown file type:", file_path)

def process_file_by_extension(file_path):
    file_extension = file_path.split(".")[-1].lower()

    # Define a dictionary mapping file extensions to corresponding functions
    extension_function_map = {
        # "pdf": get_pdf_documents # For testing
        "pdf": process_pdf
    }
    

    # Get the appropriate function based on the file extension, or use process_unknown if not found
    process_function = extension_function_map.get(file_extension, process_unknown)

    # Call the selected function
    raw_text = process_function(file_path)
    return raw_text


def analyze_general_documents(file_path):
    # print("formmmmmm")
    endpoint = os.getenv('AZURE_FORM_RECOGNIZER_ENDPOINT')
    api_key = os.getenv('AZURE_FORM_RECOGNIZER_KEY')

    with open(file_path, "rb") as file:
        file_content = file.read()
        # create your `DocumentAnalysisClient` instance and `AzureKeyCredential` variable
        document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

        poller = document_analysis_client.begin_analyze_document("prebuilt-read", file_content)
        result = poller.result()
        save_file(file_path, json.dumps(result.to_dict(), ensure_ascii=False), "json")
        
    return result.to_dict()


def recursive_chunks(text):
    tiktoken_cache_dir = "./app/tiktoken"
    # tiktoken_cache_dir = "/Users/mac/Projects/Intelligencia/Intelligencia-AI-Demo-Backend/app/tiktoken"
    # tiktoken_cache_dir = "C:/Users/FCC/VS Code Projects/Intelligencia-AI-Demo-Backend/app/tiktoken"
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
    # validate
    assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))
    # chunking
    chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
    overlap_size = int(os.getenv('CHUNK_OVERLAP', 200))

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",". "," ",""],
        chunk_size=chunk_size, 
        chunk_overlap=0
        )
    chunks = text_splitter.split_text(text)

    final_chunks = [chunks[0]]  # Initialize with the first chunk

    for i in range(1, len(chunks)):
        # Reverse the previous chunk
        reversed_chunk = chunks[i - 1][::-1]
        # Create a smaller splitter for reversed overlap
        overlap_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n","\n"," ."," ",""],
            chunk_size=overlap_size, 
            chunk_overlap=0
            )
        # Split the reversed chunk to get the overlap
        reversed_overlap = overlap_splitter.split_text(reversed_chunk)[0]
        # Reverse the overlap back to original order
        original_order_overlap = reversed_overlap[::-1]
        # Append the original order overlap to the start of the current chunk
        final_chunks.append(original_order_overlap +"\n"+ chunks[i])

    return final_chunks



def get_date(source):
    # Define a list of regex patterns to match different date formats
    date_patterns = [

        re.compile(r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)-(\d{2})-(\d{4})\b)', re.IGNORECASE), #April/01/2017

        re.compile(r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)/(\d{2})/(\d{4})\b)', re.IGNORECASE), #April/01/2017

        
        re.compile(r'\b(\d{1,2})-(?:January|February|March|April|May|June|July|August|September|October|November|December)-(\d{2})\b', re.IGNORECASE), # 02-April-17
        re.compile(r'\b(\d{1,2})-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2})\b', re.IGNORECASE), # 02-Apr-17

        re.compile(r'\b(\d{1,2})-(?:January|February|March|April|May|June|July|August|September|October|November|December)-(\d{4})\b', re.IGNORECASE), # 02-April-2017       
        re.compile(r'\b(\d{1,2})-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{4})\b', re.IGNORECASE), # 02-Apr-2017



        re.compile(r'\b(\d{1,2})/(?:January|February|March|April|May|June|July|August|September|October|November|December)/(\d{2})\b', re.IGNORECASE), # 02/April/17
        re.compile(r'\b(\d{1,2})/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/(\d{2})\b', re.IGNORECASE), # 02/Apr/17

        re.compile(r'\b(\d{1,2})/(?:January|February|March|April|May|June|July|August|September|October|November|December)/(\d{4})\b', re.IGNORECASE), # 02/April/2017       
        re.compile(r'\b(\d{1,2})/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/(\d{4})\b', re.IGNORECASE), # 02/Apr/2017




        re.compile(r'\b(\d{1,2})\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s(\d{2})\b', re.IGNORECASE), # 02 April 17
        re.compile(r'\b(\d{1,2})\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{2})\b', re.IGNORECASE), # 02 Apr 17

        re.compile(r'\b(\d{2})\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s(\d{4})\b', re.IGNORECASE), # 02 April 2017       
        re.compile(r'\b(\d{2})\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{4})\b', re.IGNORECASE), # 02 Apr 2017




        re.compile(r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\d{4}\b)', re.IGNORECASE), #April2017
        re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{4}\b)', re.IGNORECASE), #Apr2017
        

        re.compile(r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\d{2}\b)', re.IGNORECASE), #April17
        re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2}\b)', re.IGNORECASE), #Apr17



        re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s(\d{4})\b',  re.IGNORECASE), #April 2017
        re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{4})\b)', re.IGNORECASE), # Apr 2017
        
        re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s(\d{2})\b', re.IGNORECASE), #April 17
        re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{2})\b)', re.IGNORECASE), #Apr 17


        
        re.compile(r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)/(\d{4})\b)', re.IGNORECASE), #April/2017
        re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/(\d{4})\b)', re.IGNORECASE), #Apr/2017


        re.compile(r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)/(\d{2})\b)', re.IGNORECASE), #April/17
        re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/(\d{2})\b)', re.IGNORECASE), #Apr/17
        



        re.compile(r'(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)-(\d{4})\b)', re.IGNORECASE), #April-2017
        re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{4})\b)', re.IGNORECASE), #Apr-2017
        
        re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)-(\d{2})\b', re.IGNORECASE), #April-2017
        re.compile(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2})\b)', re.IGNORECASE), #Apr-17


    

        re.compile(r'\b(?:0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}\b'), # dd/mm/yyyy
        re.compile(r'\b(?:0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{2}\b'), # dd/mm/yy
        re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}\b'), #mm/dd/yyyy
        re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{2}\b'), #mm/dd/yy

        
        re.compile(r'\b(?:[1-9]|[12][0-9]|3[01])/([1-9]|1[0-2])/\d{4}\b'), # dd/mm/yyyy
        re.compile(r'\b(?:[1-9]|[12][0-9]|3[01])/([1-9]|1[0-2])/\d{2}\b'), # dd/mm/yy
        re.compile(r'\b([1-9]|1[0-2])/([1-9]|[12][0-9]|3[01])/\d{4}\b'), #mm/dd/yyyy
        re.compile(r'\b([1-9]|1[0-2])/([1-9]|[12][0-9]|3[01])/\d{2}\b'), #mm/dd/yy
        

        re.compile(r'\b(?:0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-\d{4}\b'), # dd-mm-yyyy
        re.compile(r'\b(?:0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-\d{2}\b'), # dd-mm-yy
        re.compile(r'\b(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-\d{4}\b'), #mm-dd-yyyy
        re.compile(r'\b(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-\d{2}\b'), #mm-dd-yy

        
        re.compile(r'\b(?:[1-9]|[12][0-9]|3[01])-([1-9]|1[0-2])-\d{4}\b'), # dd-mm-yyyy
        re.compile(r'\b(?:[1-9]|[12][0-9]|3[01])-([1-9]|1[0-2])-\d{2}\b'), # dd-mm-yy
        re.compile(r'\b([1-9]|1[0-2])-([1-9]|[12][0-9]|3[01])-\d{4}\b'), #mm-dd-yyyy
        re.compile(r'\b([1-9]|1[0-2])-([1-9]|[12][0-9]|3[01])-\d{2}\b'), #mm-dd-yy
        

        re.compile(r'\b(\d{4})-(\d{4})\b'), # 2021-2022
        re.compile(r'\b(\d{4})\b') # 2022

        # Add more patterns as needed
    ]


    # List of date formats to try
    date_formats = ['%B%Y', '%b%Y',  '%B%y', '%b%y', 
                    '%B %Y', '%b %Y', '%B %y', '%b %y',
                    '%B/%Y', '%b/%Y', '%B/%y', '%b/%y', 
                    '%B-%Y', '%b-%Y', '%B-%y', '%b-%y', 
                    
                    '%B-%d-%Y',
                    '%B/%d/%Y',

                    '%d/%B/%Y', '%d/%b/%Y', '%d/%B/%y', '%d/%b/%y',  
                    '%d-%B-%Y', '%d-%b-%Y', '%d-%B-%y', '%d-%b-%y',   
                    '%d %B %Y', '%d %b %Y', '%d %B %y', '%d %b %y',  
                    '%d/%m/%Y', '%d/%m/%y','%m/%d/%Y', '%m/%d/%y', 
                    '%m-%d-%Y', '%m-%d-%y', '%d-%m-%Y','%d-%m-%y']

    match_year_year = None
    match_year = None
    # Iterate through the patterns and search for a match
    for pattern in date_patterns:
        match = pattern.search(source)
        if match:
            date_string = match.group(0)
            # print("Found date:", date_string)

                
            pattern_year_year = r'\b(\d{4})-(\d{4})\b'  # Example pattern: YYYY-YYYY

            match_year_year = re.match(pattern_year_year, date_string)
            if match_year_year != None:
                # print(match_year_year[0])
                pass

            pattern_year = r'\b(\d{4})\b'  # Example pattern: YYYY

            match_year = re.match(pattern_year, date_string)
            # print(match_year)
            if match_year != None:
                # print("gg",match_year[0])
                pass
            
            break  # Exit the loop if a match is found

        
    if match_year_year != None:
        second_year = match_year_year[0].split("-")[1]
        # print("seond_year",second_year)
        date_object = datetime(int(second_year), 9, 15)
        return date_object
    
    if match_year != None:
        # print("nott none")
        year = match_year[0]
        # print("year", year)
        date_object = datetime(int(year), 1, 1)
        return date_object
    
    if match_year_year is None:
        result = ""
        if not match:
            # print("No date found in the string.")
            pass

            # Convert the date string to a datetime object
        else:
            result = convert_to_datetime(date_string, date_formats)


        # print the result
        if result:
            # print("Converted datetime object:", result)
            # print(type(result))
            date = result
        else:
            # print("Could not determine the date format.")
            # date = datetime.min
            date = datetime(2000, 1, 1)
        return date



# Function to try converting the date string with different formats
def convert_to_datetime(date_string, date_formats):
    for format_str in date_formats:
        try:
            date_object = datetime.strptime(date_string, format_str)
            return date_object  # Return the datetime object if conversion is successful
        except ValueError:
            pass  # If ValueError occurs, try the next format

    return None  # Return None if none of the formats match



# Format text chunks by adding fields to each chunk as required
# def get_format_text_chunks(file_path, data, meta, title, description):
def get_format_text_chunks(file_path, data):
    chunked_content=[]
    # chunked=chunk_and_wrap(data, title, description)
    chunked = recursive_chunks(data)
    #print(chunked)
    ctr = 0
    for chunk in chunked:
        ctr = ctr + 1
        #print(chunk)
        #print("-0------------\n")
        
        filename = os.path.basename(file_path)
        title, extension = os.path.splitext(filename)
        chunk_id = ctr

        base_url = urljoin(os.getenv("ACCOUNT_URL"), os.getenv("CONTAINER_NAME"))
        url = urljoin(base_url + '/',  quote(filename))
        url = url.replace("<","").replace(">","")
  
        # base_url = urljoin(os.getenv("ACCOUNT_URL"), os.getenv("CONTAINER_NAME"))
        # url = urljoin(base_url, file_path)
        
        url = url.replace("<","").replace(">","")

        # content_date = get_date(filename)

        chunk_data = {"content":chunk, "title": title, "chunk_id": ctr, "filename": filename }
        # chunk_data["description"]= (chunk["chunk_part"]  + " " + description).replace('\u200c', '')
        chunked_content.append(chunk_data)
        
    return chunked_content

# Creates documents from the created chunks to be added to the index
def create_documents_chunks(text_chunks):

    documents_to_upload = []
    embeddings = []

    for i, chunk in enumerate(text_chunks):
        embedding = get_embeddings(chunk["content"])
        # embeddings.append(embedding)
        document = {
            "@search.action": "mergeOrUpload",
            "id": str(uuid.uuid4()),
            # "metadata": doc["metadata"],
            "content": chunk["content"],
            "content_vector": embedding,
            "title": chunk["title"],
            "filename": chunk["filename"], 
            # "content_date": chunk["content_date"],
            "chunk_id": chunk["chunk_id"]
        }
        documents_to_upload.append(document)
    return documents_to_upload




# Saves file with the given extension
def save_file(file_path, text, extenstion):
    new_file_path = change_file_extension(file_path, extenstion)
    with open(new_file_path, "w", encoding="utf-8") as file:
        file.write(text)
    # Set read permissions for the file
    os.chmod(file_path, 0o644)
    return new_file_path

def change_file_extension(file_path, new_extension):
    # Get the base file name without extension
    folder_path = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create the new file name with the desired extension
    new_file_name = os.path.join(folder_path, f"{base_name}.{new_extension}")
    return new_file_name


# Adds document to azure search index
def add_document_azure(documents_to_index):
    
    endpoint = os.getenv('AZURE_SEARCH_BASE')
    api_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
    credential = AzureKeyCredential(os.getenv('AZURE_SEARCH_ADMIN_KEY'))
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')

    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    indexing = search_client.upload_documents(documents=documents_to_index)
    return indexing
# Creates azure search index with custom schema
def create_azure_search_index(index_name):
    
    print("create fct")
    # print("end: ", os.getenv("AZURE_SEARCH_BASE"))
    # print("key: ", os.getenv("AZURE_SEARCH_ADMIN_KEY"))
    # print("version: ", os.getenv("AZURE_SEARCH_API_VERSION"))
    # print("acc: ", os.getenv("AZURE_SEARCH_ACCOUNT_NAME"))
    
    endpoint = os.getenv('AZURE_SEARCH_BASE')
    api_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
    api_version = os.getenv('AZURE_SEARCH_API_VERSION')
    credential = AzureKeyCredential(os.getenv('AZURE_SEARCH_ADMIN_KEY'))
    account = os.getenv("AZURE_SEARCH_ACCOUNT")
    client = SearchIndexClient(endpoint, AzureKeyCredential(api_key))
    
    # Create the index
    
    index_url = f"https://{account}.search.windows.net/indexes?api-version={api_version}"
    print("index", index_url)
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    
    index_definition = {
           "name": index_name,
            "defaultScoringProfile": None,
            "fields": [
                {
                "name": "id",
                "type": "Edm.String",
                "searchable": False,
                "filterable": True,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": True,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": None,
                
                "dimensions": None,
                "vectorSearchProfile": None,
                "synonymMaps": []
                },
                {
                "name": "content",
                "type": "Edm.String",
                "searchable": True,
                "filterable": False,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": "ar.microsoft",
                
                "dimensions": None,
                "vectorSearchProfile": None,
                "synonymMaps": []
                },
                {
                "name": "content_vector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "filterable": False,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": None,
                
                "dimensions": 1536,
                "vectorSearchProfile": "default-profile",
                "synonymMaps": []
                },
                {
                "name": "filename",
                "type": "Edm.String",
                "searchable": False,
                "filterable": True,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": None,
                
                "dimensions": None,
                "vectorSearchProfile": None,
                "synonymMaps": []
                },
                {
                "name": "title",
                "type": "Edm.String",
                "searchable": True,
                "filterable": True,
                "retrievable": True,
                "sortable": False,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": "ar.microsoft",
                
                "dimensions": None,
                "vectorSearchProfile": None,
                "synonymMaps": []
                },

                # {
                # "name": "content_date",
                # "type": "Edm.DateTimeOffset",
                # "searchable": False,
                # "filterable": True,
                # "retrievable": True,
                # "sortable": False,
                # "facetable": False,
                # "key": False,
                # "indexAnalyzer": None,
                # "searchAnalyzer": None,
                # "analyzer": None,
                
                # "dimensions": None,
                # "vectorSearchProfile": None,
                # "synonymMaps": []
                # },
                {
                "name": "chunk_id",
                "type": "Edm.Int32",
                "searchable": False,
                "filterable": True,
                "retrievable": True,
                "sortable": True,
                "facetable": False,
                "key": False,
                "indexAnalyzer": None,
                "searchAnalyzer": None,
                "analyzer": None,
                
                "dimensions": None,
                "vectorSearchProfile": None,
                "synonymMaps": []
                },
                
            ],
            "scoringProfiles":[],
            "corsOptions": None,
            "suggesters": [],
            "analyzers": [],
            
            "tokenizers": [],
            "tokenFilters": [],
            "charFilters": [],
            "encryptionKey": None,
            "similarity": {
                "@odata.type": "#Microsoft.Azure.Search.BM25Similarity",
                "k1": None,
                "b": None
            },
            "semantic": {
                "defaultConfiguration": None,
                "configurations": [
                {
                    "name": "semantic_config",
                    "prioritizedFields": {
                    "titleField": {
                        "fieldName": "title"
                    },
                    "prioritizedContentFields": [
                        {
                        "fieldName": "content"
                        }
                    ],
                    "prioritizedKeywordsFields": []
                    }
                }
                ]
            },
            "vectorSearch": {
                "algorithms": [
                {
                    "name": "default",
                    "kind": "hnsw",
                    "hnswParameters": {
                    "metric": "cosine",
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500
                    },
                    "exhaustiveKnnParameters": None
                }
                ],
                "profiles": [
                {
                    "name": "default-profile",
                    "algorithm": "default"
                }
                ]
            }

    }
   
    response = requests.post(index_url, headers=headers, json=index_definition)
    print("response", response.text)
    if response.status_code == 201:
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Failed to create the index. Status code: {response.status_code}, Response: {response}")
    return response


# Check if given index is found in azure search indexes
async def get_azure_index(index_name):
    endpoint = os.getenv('AZURE_SEARCH_BASE')
    credential = AzureKeyCredential(os.getenv('AZURE_SEARCH_ADMIN_KEY'))
    print("n get:", endpoint, credential, index_name)
    search_index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    try:
    # Attempt to get the index (this will raise a ResourceNotFoundError if it doesn't exist)
        search_index = await search_index_client.get_index(index_name)
        return True
    except Exception:
        return False


# Delete azure search index
def delete_azure_index(index_name):
    if index_name is None or index_name == "":
        raise Exception(f"Name Error: Index name must be a valid name!")
    else:
        endpoint = os.getenv('AZURE_SEARCH_BASE')
        credential = AzureKeyCredential(os.getenv('AZURE_SEARCH_ADMIN_KEY'))

        search_index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        search_index_client.delete_index(index_name)


def get_file_document_ids(index_client, filename, batch, skip=0):
    result = []

    response = index_client.search(search_text="*", filter=f"filename eq '{filename}'", select="id", top=batch,skip=skip)
    print("response", response)
    ctr=0
    for document in response:
        print(document)
        ctr=ctr+1
        result.append(document['id'])
        print("id",document['id'])
    return result,ctr


def file_exists_in_index(filename):
    endpoint = os.getenv('AZURE_SEARCH_BASE')
    api_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
    credential = AzureKeyCredential(str(api_key))
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
    search_service_name = os.getenv("AZURE_SEARCH_ACCOUNT_NAME")

    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)


    # Define the search query to check if the field with specific value exists
    search_query = f" filename eq '{filename}'"

    

    # Execute the search query
    response = search_client.search(search_text="*", filter=f"filename eq '{filename}'")
    # print("response file existss", response)
    # Check if any documents match the query
    
    for result in response:
        # print("in for", result)
        if result.get("filename"):
            # print(result.get("document"))
            return True  # The field with the specified value exists

    return False  # No matching documents found


def delete_file_in_index(filename):
    print("delete")
    endpoint = os.getenv('AZURE_SEARCH_BASE')
    api_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
    credential = AzureKeyCredential(str(api_key))
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
    search_service_name = os.getenv("AZURE_SEARCH_ACCOUNT_NAME")

    print("end", index_name)

    batch=int(os.getenv("BATCH"))

    client = SearchClient(endpoint, index_name, credential)

    old_ctr=0
    ids=[]
    while(True):
        
        results,ctr = get_file_document_ids(client, filename, batch, skip=old_ctr)
        print(results)
        ids.append(results)
        old_ctr=old_ctr+ctr

        if ctr<1000:
            break

    if ids[0]:
        for batch_docs in ids:
            dict_of_ids = [{"id": value} for value in batch_docs]
            print(len(dict_of_ids))
            client.delete_documents(documents=dict_of_ids)
        
        return {"deleted": filename}
    else:
        return {"deleted": f"No such file '{filename}'"}



def get_all_document_ids(index_client,skip=0):
    result = []


    batch=os.getenv("BATCH")

    response = index_client.search(search_text="*", select="id", top=batch,skip=skip)
    ctr=0
    for document in response:

        ctr=ctr+1
        result.append(document['id'])

    return result,ctr

def clear_index():
    endpoint = os.getenv('AZURE_SEARCH_BASE')
    api_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
    credential = AzureKeyCredential(str(api_key))
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
    search_service_name = os.getenv("AZURE_SEARCH_ACCOUNT_NAME")

    client = SearchClient(endpoint, index_name, AzureKeyCredential(api_key))

    old_ctr=0
    ids=[]
    while(True):
        
        results,ctr = get_all_document_ids(client,skip=old_ctr)

        ids.append(results)
        old_ctr=old_ctr+ctr

        if ctr<1000:
            break

    for batch_ids in ids:
        dict_of_ids = [{"id": value} for value in batch_ids]
        print(dict_of_ids)
        client.delete_documents(documents=dict_of_ids)
