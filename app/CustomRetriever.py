import json
from langchain.schema import BaseRetriever # The base class for creating custom retrievers in LangChain.
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.schema import Document # The LangChain document class that holds metadata.
from pydantic import BaseModel # From Pydantic, used for data validation.
from dotenv import load_dotenv
import asyncio
import requests
import utils as u
import os

class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: AzureSearch
    query: str

    class Config:
        arbitrary_types_allowed = True

    def combine_metadata(self, doc) -> str:
        metadata = doc.metadata
        print("Source: " + metadata["source"])
        return (
            "Source: " + metadata["source"]
        )

    def get_relevant_documents(self, query):
        print("Entered get_relevant_documents in CustomR")
        docs = []
        for doc in self.vectorstore.similarity_search(query):
            # print("doc is ==> "+ doc)
            # content = self.combine_metadata(doc)
            docs.append(Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            ))
        print(docs)
        return docs

    async def aget_relevant_documents(self, query):
        query = self.query
        docs = []
        # Define the API endpoint and parameters
        api_url = os.getenv('AZURE_SEARCH_BASE')+"/indexes/"+os.getenv('AZURE_SEARCH_INDEX_NAME')+"/docs/search?api-version="+os.getenv('AZURE_SEARCH_API_VERSION')
        headers = {
            "Content-Type": "application/json",

            "api-key": os.getenv('AZURE_SEARCH_ADMIN_KEY')
        }

        # Transform query; extract keywords
        try:
            query = await u.transform_query_async(query)
        except:
            raise

        vec = u.get_embeddings(query)

        # Define the number of search results to retrieve
        top = os.getenv("Top", 10)

        #Define the request payload
        payload = {
            "search": query,
            "queryType": "semantic",
            # "searchFields": "title,description,content",
            "semanticConfiguration": "semantic_config",
            "top": top,
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": vec,
                    "fields": "content_vector",
                    "k": top
                }
            ]   
        }

        # Make the POST request
        response = requests.post(api_url, json=payload, headers=headers)

        # Check the response status code
        if response.status_code == 200:
            # Successful response
            data = response.json()["value"]
            # print(data)
            # Sort context
            data_len = len(data)
            if data_len >= 1:
                # sorted_data = self.inverse_bell_sort(data, highestFirst = False, scoreBase = "@search.rerankerScore")
                sorted_data = self.linear_sort(data, highestFirst = True, scoreBase = "@search.rerankerScore")
            else:
                print("Nothing been returned from the Index")
            # Prepare context for LLM consumption
            for doc in sorted_data:
                print(doc["title"]+"    "+str(doc["@search.score"])+"    "+str(doc["@search.rerankerScore"]))
                docs.append(Document(
                    page_content="{\"title\" : \""+doc["title"]+"\" , \n\"content\" : \""+doc["content"]+"\" , \n\"source\" : \""+doc["title"]+"\"}",
                    # metadata={"source": doc["filepath"]}
                ))
            # print(docs)
            return docs
        else:
            # Error response
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

    # Sort the context on an inverted bell curve with orientation handling
    def inverse_bell_sort(self, data, highestFirst, scoreBase):
        # Sorting the array in ascending score value
        sorted_data = sorted(data, key=lambda x: x[scoreBase])
        # Splitting array in two, odd index items (in reverse) and even index items
        odds = sorted_data[::2] [::-1]
        evens = sorted_data[1::2]
        # Combine arrays
        sorted_data = odds + evens
        # Handle array orientation
        data_size = len(sorted_data)
        if data_size % 2 == 0 and highestFirst:
            sorted_data.reverse()
        elif data_size %2 != 0 and not highestFirst:
            sorted_data.reverse()
        # Return sorted and oriented array
        return sorted_data
    

    # Sort the context linearly with orientation handling
    def linear_sort(self, data, highestFirst, scoreBase):
        # Sorting the array in ascending score value
        sorted_data = sorted(data, key=lambda x: x[scoreBase])
        # Handle array orientation
        if highestFirst:
            sorted_data.reverse()
        # Return sorted and oriented array
        return sorted_data