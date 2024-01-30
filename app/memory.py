import os
import cosmos_utils as cu
from langchain.memory import ConversationBufferMemory
from langchain.memory import MongoDBChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory


def get_memory_cosmos(conv_id, channel='web', window_memory = 16):
    previous_conv = cu.get_record_from_cosmos(conv_id, channel)
    window_memory = -1 * window_memory
    if previous_conv != {}:
        messages_dict = previous_conv['conversation']
        history = messages_from_dict(messages_dict)
        message_history = ChatMessageHistory()
        message_history.messages = history
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, chat_memory=message_history, k=16)
        memory.chat_memory.messages = memory.chat_memory.messages[window_memory:]
        # print(memory)
    else:
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=16)
        memory.chat_memory.messages = memory.chat_memory.messages[window_memory:]
        # print(memory)
    return memory

def save_memory_cosmos(memory, conv_id, channel="web"):
    cosmos_object = {}
    cosmos_object["id"] = conv_id
    cosmos_object["channel"] = channel
    cosmos_object["conversation"] = messages_to_dict(memory.chat_memory.messages)
    cu.upsert_record_into_cosmos(cosmos_object)

def get_memory_mongodb(conv_id):
    message_history = MongoDBChatMessageHistory(
        connection_string=os.getenv('MONGODB_CONNECTION'), session_id=conv_id
    )
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, chat_memory=message_history, k=os.getenv('K', 10))
    return memory

def save_memory_mongodb(user_message, ai_message):
    message_history = MongoDBChatMessageHistory(
        connection_string=os.getenv('MONGODB_CONNECTION'), session_id=os.getenv('conv_id')
    )
    message_history.add_user_message(user_message)
    message_history.add_ai_message(ai_message)
    print(message_history.messages)


def memory_to_text(memory):
    history_messages = memory.chat_memory.messages
    history = ''
    if history_messages:
        for item in history_messages:
            if isinstance(item, HumanMessage):
                history += '\nHuman: ' + item.content
            elif isinstance(item, AIMessage):
                history += '\nAssistant: ' + item.content
        history += '\n'
    # print('History is:', history)
    return history