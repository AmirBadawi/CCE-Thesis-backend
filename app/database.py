import asyncio
from concurrent.futures import ThreadPoolExecutor
import pyodbc
import os
from dotenv import load_dotenv
from urllib import parse
import datetime
import time


def azure_sql_connection(user, password, hostname, database_name, driver):
    # Building the connection string directly using string formatting
    conn_str = (
        f"Driver={{{driver}}};"
        f"Server=tcp:{hostname},1433;"
        f"Database={database_name};"
        f"Uid={user};"
        f"Pwd={{{password}}};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )
    conn = pyodbc.connect(conn_str)
    return conn


def table_exists(cursor, table_name):
    cursor.execute("""
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_name = ?
    """, [table_name])
    if cursor.fetchone()[0] == 1:
        return True
    return False


def create_table(cursor):
    cursor.execute("""
    CREATE TABLE conversation (
        _id INT PRIMARY KEY IDENTITY(1,1),
        conversation_id NVARCHAR(MAX),
        user_message NVARCHAR(MAX),
        ai_message NVARCHAR(MAX),
        user_id NVARCHAR(MAX),
        user_email NVARCHAR(MAX),
        user_name NVARCHAR(MAX),
        bot_id NVARCHAR(MAX),
        inserted_timestamp DATETIME,
        chat_timestamp DATETIME
    )
    """)


def insert_rows(cursor, data):
    insert_query = """
    INSERT INTO conversation (
        conversation_id, user_message, ai_message, user_id, 
        user_email, user_name, bot_id, inserted_timestamp, chat_timestamp
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    try:
        cursor.execute(insert_query, (
            data['conversation_id'],
            data['user_message'],
            data['ai_message'],
            data['user_id'],
            data['user_email'],
            data['user_name'],
            data['bot_id'],
            data['inserted_timestamp'],
            data['chat_timestamp']
        ))
        cursor.commit()
    except pyodbc.Error as e:
        if e.args[0] == '42S02':  # Table doesn't exist
            create_table(cursor)
            cursor.commit()
            print("Table created successfully.")
            insert_rows(cursor, data)  # Retry inserting data
        else:
            print(f"Error occurred: {e}")    


_executor = ThreadPoolExecutor()

async def log_conversation(**data):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _sync_log_conversation, data)

def _sync_log_conversation(data):
    load_dotenv()
    conn = azure_sql_connection(os.getenv('DATABASE_USER'), os.getenv('DATABASE_PASSWORD'), os.getenv('DATABASE_SERVER'), os.getenv('DATABASE_NAME'), os.getenv('DRIVER'))
    cursor = conn.cursor()

    try:
        insert_rows(cursor, data)
        conn.commit()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cursor.close()
        conn.close()


def create_table_if_not_exists():
    load_dotenv()
    conn = azure_sql_connection(
        os.getenv('DATABASE_USER'),
        os.getenv('DATABASE_PASSWORD'),
        os.getenv('DATABASE_SERVER'),
        os.getenv('DATABASE_NAME'),
        os.getenv('DRIVER')
    )
    cursor = conn.cursor()

    if not table_exists(cursor, 'conversation'):
        create_table(cursor)
        conn.commit()
        print("Table created successfully.")
    else:
        print("Table already exists.")

    cursor.close()
    conn.close()


# KME
def update_row(cursor, data):
    update_query = """
    UPDATE watermarktable SET in_index = ?
    WHERE file_relative_url = ?
    """
    
    try:
        cursor.execute(update_query, (
            data['in_index'],
            data['file_relative_url']
        ))
        cursor.commit()
    except pyodbc.Error as e:
        # print(f"Error occurred: {e}")
        raise    


def select_row(cursor, data):
    select_query = """
    SELECT watermarktable.in_index FROM watermarktable
    WHERE watermarktable.file_relative_url = ?
    """
    
    try:
        cursor.execute(select_query, (
            data['file_relative_url']
        ))
        for row in cursor.fetchall():
            result = row.in_index
            return result
        
    except pyodbc.Error as e:
        # print(f"Error occurred: {e}")    
        raise

def update_file_in_index_status(data):
    load_dotenv()
    conn = azure_sql_connection(os.getenv('DATABASE_USER'), os.getenv('DATABASE_PASSWORD'), os.getenv('DATABASE_SERVER'), os.getenv('DATABASE_NAME'), os.getenv('DRIVER'))
    cursor = conn.cursor()

    try:
        update_row(cursor, data)
        conn.commit()
    except Exception as e:
        # print(f"Error occurred: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def check_file_in_index_status(data):
    load_dotenv()
    conn = azure_sql_connection(os.getenv('DATABASE_USER'), os.getenv('DATABASE_PASSWORD'), os.getenv('DATABASE_SERVER'), os.getenv('DATABASE_NAME'), os.getenv('DRIVER'))
    cursor = conn.cursor()

    try:
        return select_row(cursor, data)
    except Exception as e:
        # print(f"Error occurred: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def log_to_sql(conv_id, query, request, response):
    myobj = {
                "ConversationID": conv_id,
                "Text": query,
                "UserID": request.UserID,
                "email": request.email,
                "Name": request.Name,
                "BotID": request.BotID,
                "chat_timestamp": request.Time
            }
    # print("Started the API call in the background")
    asyncio.create_task(log_conversation(
        conversation_id=myobj["ConversationID"],
        user_message=myobj["Text"],
        ai_message=response,
        user_id=myobj["UserID"],
        user_email=myobj["email"],
        user_name=myobj["Name"],
        bot_id=myobj["BotID"],
        inserted_timestamp = datetime.datetime.fromtimestamp(time.time()),
        chat_timestamp=myobj["chat_timestamp"]
    ))