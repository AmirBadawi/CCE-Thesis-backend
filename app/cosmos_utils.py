import os
import logging
from azure.cosmos import CosmosClient


def upsert_record_into_cosmos(object):

    endpoint = os.getenv('COSMOS_ENDPOINT')
    key = os.getenv('COSMOS_KEY')
    database_name = os.getenv('COSMOS_DB')
    container_name = os.getenv('COSMOS_COL')

    cosmos_client = CosmosClient(endpoint, key)
    database = cosmos_client.get_database_client(database_name)
    container = database.get_container_client(container_name)

    try:
        container.upsert_item(
            object
        )
    except Exception as e:
        logging.warning('Could not upsert, conversation state may be lost: {0}'.format(e))
        return False
    return True


def delete_cosmos_item_by_id(conv_id):
    # Initialize the Cosmos client
    client = CosmosClient(os.getenv('COSMOS_ENDPOINT'), os.getenv('COSMOS_KEY'))

    # Get a reference to the database and container
    database = client.get_database_client(os.getenv('COSMOS_DB'))
    container = database.get_container_client(os.getenv('COSMOS_COL'))

    # Delete the item by its ID
    try:
        container.delete_item(conv_id, partition_key=conv_id)
        print(f"Item with ID '{conv_id}' has been deleted from the container.")
        return True
    except Exception as e:
        print(f"Error deleting item with ID '{conv_id}': {e}")
        return False
    

def get_record_from_cosmos(user_id, channel, default={}):

    endpoint = os.getenv('COSMOS_ENDPOINT')
    key = os.getenv('COSMOS_KEY')
    database_name = os.getenv('COSMOS_DB')
    container_name = os.getenv('COSMOS_COL')

    cosmos_client = CosmosClient(endpoint, key)
    database = cosmos_client.get_database_client(database_name)
    container = database.get_container_client(container_name)

    query_cosmos = "SELECT * FROM c WHERE c.id = @id"
    parameters = [
        {"name": "@id", "value": user_id},
        {"name": "@channel", "value": channel}
    ]
    try:
        results = container.query_items(query=query_cosmos, parameters=parameters, enable_cross_partition_query=True)
    except Exception as e:
        logging.warning('Could not search container: {0}'.format(e))
        return default

    if not results:
        logging.info('Not found in Cosmos')
        return default

    for result in results:
        logging.info("Found in Cosmos")
        return result

    return default

def delete_all_cosmos_items():
    # Initialize the Cosmos client
    client = CosmosClient(os.getenv('COSMOS_ENDPOINT'), os.getenv('COSMOS_KEY'))

    # Get a reference to the database and container
    database = client.get_database_client(os.getenv('COSMOS_DB'))
    container = database.get_container_client(os.getenv('COSMOS_COL'))

    # Query all items in the container
    items = list(container.read_all_items())

    # Delete each item
    for item in items:
        try:
            container.delete_item(item, partition_key=item.get('id'))
            print(f"Item with ID '{item.get('id')}' has been deleted from the container.")
        except Exception as e:
            print(f"Error deleting item with ID '{item.get('id')}': {e}")

    print("All items have been deleted from the container.")
    return True