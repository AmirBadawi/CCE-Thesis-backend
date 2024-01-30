import datetime

from pydantic import BaseModel

# Requests Models ==>


# Model to handle request body of /chat API Endpoint
class ChatRequest(BaseModel):
    Text: str
    ConversationID: str
    UserID: str
    BotID: str
    Time: datetime.datetime
    Name: str
    email: str


class PDFPath(BaseModel):
    path: str


class FilePath(BaseModel):
    path: str

    
class Index(BaseModel):
    index_name: str = None
