import uvicorn
import os
import asyncio
import chat as ch
import chat_streaming as ch_s
import memory as mem
import cosmos_utils as cu
import utils as u
import exceptions as e
import database as db
import datetime
import time
import openai
from models import ChatRequest, FilePath, Index
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import logging
from fastapi.middleware.cors import CORSMiddleware



from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()
vector_store = None
sql = False

@app.on_event("startup")
async def startup_event():
    on_startup()


@app.get('/')
def root():
    return {'Status': "Working"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # file_path = os.path.join("/file/", file.filename)
    file_path = file.filename
    filepath = FilePath(path=file_path)
    with open(filepath.path, "wb") as f:
        f.write(file.file.read())
    response = await add(filepath)

    # return {"filename": file.filename}
    return file_path, response

# pip install python-multipart

# Creates an index with custom schema
@app.post('/create')
def create(index: Index | None=None):
    print("create")
    try:
        print("try")
        if index is not None and index.index_name is not None:
            index_name = index.index_name
        else:
            index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
        print("index name:", index_name)
        response = u.create_azure_search_index(index_name)
        print(response.text)
        if response.status_code == 201:
            return {"Succefully Created": index_name}
        else:
            raise Exception("An error occurred while creating the index, please check index name and if it already exists")
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/build")
def build(index: Index | None=None):
    try:
        if index is not None and index.index_name is not None:
            index_name = index.index_name
        else:
            index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
        u.get_vectorstore(index_name)
        return {"Succefully Created": index_name}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/recreate")
def recreate(index: Index | None=None):
    try:
        index_exists = u.get_azure_index(index.index_name)
        if index_exists:
            if index is None:
                raise HTTPException(status_code=400, detail="Enter a valid index to delete")
            else:
                try:
                    u.delete_azure_index(index.index_name)
                    create(index)
                    return {"Success": "Recreated successfully"}
                except Exception as ex:
                    logging.error(f"An error occurred: {ex}")
                    raise HTTPException(status_code=500, detail="An error occurred while recreating the index, please check index name")
        else:
            raise HTTPException(status_code=404, detail=f"The requested index does not exist to recreate it, please build it first")
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/add")
async def add(path: FilePath):
    try:
        response = await u.add_file_to_index(path.path)
        # print(response.status_code)
        # print(response.body.decode('utf-8')) #as str
        return response

    except Exception as ex:
        logging.error(f"An error occurred: {ex}")
        # return {"Error": str(ex)}
        raise HTTPException(status_code=500, detail=str(ex))
    finally:
        u.delete_file(path.path)

@app.post("/delete")
def delete(path: FilePath):
    try:
        response = u.delete_file_in_index(path.path)
        return response

    except Exception as ex:
        logging.error(f"An error occurred: {ex}")
        # return {"Error": str(ex)}
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.Text or not request.ConversationID:
        return {"bad request": "no query"}
    if  request.Text.strip().lower() == "reset":
        delete_chat_history_cosmos(request.ConversationID)
        return {"response": "Chat history was deleted successfully"}
    query = request.Text
    query = query.replace("intelligencia", "Intelligencia")
    # print("passed in query ==> "+query)
    print(request)
    conv_id = request.ConversationID
    response =""
    try:
        # Content filter on user input
        # await u.content_filter_async(query,gpt4model=False)
        # Load the memory
        memory = mem.get_memory_cosmos(conv_id)
        # print("Running the custom agent...")
        response = await asyncio.wait_for(ch.custom_agent(query, memory, conv_id, request.filename, request.turbo, request.index_access), timeout=int(os.getenv('TIMEOUT', 18)))
        print(response)
        # content filter
        # print("response before filtering ==> "+response)
        response = u.filter_response(response)
        # Content filter on response
        await u.content_filter_async(response,gpt4model=False)
        # print("Saving the chat...")
        memory.save_context({"Human": query}, {"AI": response})
        mem.save_memory_cosmos(memory, conv_id)
        # Conversation logging
        if sql:
            db.log_to_sql(conv_id=conv_id, query=query, request=request, response=response)
    except asyncio.TimeoutError:
        # print("Exceeded the timeout")
        response = await e.generate_timeout_response()
    except openai.RateLimitError:
            print("Exception caught: "+ str(ex))
            response = "You have exceeded the call rate limit of your current OpenAI tier. Please try again in a few seconds."
    except Exception as ex:
        print("Exception caught: "+ str(ex))
        if "content filter" in str(ex).lower():
            # response = "Intelligencia AI Virtual Assistant's content filter has been triggered! Please double check your query and make sure it conforms to our safe-usage policies."
            try:
                response = await e.generate_content_filter_error_async(query)
            except:
                return {"response": "Intelligencia AI Virtual Assistant's content filter has been triggered! Please double check your query and make sure it conforms to our safe-usage policies."}
        else:
            try:
                response = await e.generate_userfriendly_error_response()
            except:
                return {"response": "We ran into a problem! Please try again later."}
            # return {"response": "We ran into a problem! Please try again later."}
            # return {"response": str(ex)}

    return {"response": response.strip()}
    # return response

@app.post("/stream_chat")
async def stream_chat(request: ChatRequest):
    try:
        print(request)
        if not request.Text or not request.ConversationID:
            return {"bad request": "no query"}
        if request.Text.strip().lower() == "reset":
            delete_chat_history_cosmos(request.ConversationID)
            return {"response": "Chat history was deleted successfully"}
        
        query = request.Text.replace("intelligencia", "Intelligencia")
        conv_id = request.ConversationID

        generator = ch_s.custom_agent(query, conv_id, request, sql)
        
        return StreamingResponse(generator, media_type="text/event-stream")

    except asyncio.TimeoutError:
        # print("Exceeded the timeout")
        return await e.generate_timeout_response()
    except openai.RateLimitError:
            # print("Exception caught: "+ str(ex))
            return "You have exceeded the call rate limit of your current OpenAI tier. Please try again in a few seconds."
    except Exception as ex:
        print("Exception caught: "+ str(ex))
        if "content filter" in str(ex).lower():
            # response = "Intelligencia AI Virtual Assistant's content filter has been triggered! Please double check your query and make sure it conforms to our safe-usage policies."
            try:
                return await e.generate_content_filter_error_async(query)
            except:
                return "Intelligencia AI Virtual Assistant's content filter has been triggered! Please double check your query and make sure it conforms to our safe-usage policies."
        else:
            try:
                return await e.generate_userfriendly_error_response()
            except:
                return "We ran into a problem! Please try again later."
            # return {"response": "We ran into a problem! Please try again later."}
            # return {"response": str(ex)}

@app.delete("/history/cosmos/{conv_id}")
def delete_chat_history_cosmos(conv_id):
    response = cu.delete_cosmos_item_by_id(conv_id)
    if response == True:
        return {"status": "Success"}

    else:
        return {"status": "Error"}

@app.delete("/history/reset")
def delete_chat_history_cosmos():
    empty = cu.check_cosmos_empty()
    if empty:
        response = cu.delete_all_cosmos_items()
        if response == True:
            return {"status": "Success"}

        else:
            return {"status": "Error"}
    else:
        return {"status": "Success"}


def on_startup():
    if sql:
        db.create_table_if_not_exists()


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)