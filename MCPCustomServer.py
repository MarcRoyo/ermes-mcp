import random
from fastmcp import FastMCP, Context
from pymilvus import MilvusClient, DataType, model
import logging
import base64
from datetime import datetime
from deltalake import DeltaTable, write_deltalake
import pyarrow.dataset as ds
import pandas as pd
import json
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

hdr = logging.FileHandler(filename='output.log', encoding='utf-8')
hdr.setFormatter(logging.Formatter('[%(asctime)s] thread=%(thread)s:%(levelname)s: %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(hdr)
logger.setLevel(logging.DEBUG)

client = MilvusClient("http://milvus-standalone:19530")

mcp = FastMCP(name="Dice Roller")

embedding_fn = model.DefaultEmbeddingFunction()

def base64_decode(string):
    """
    Adds back in the required padding before decoding.
    """
    padding = 4 - (len(string) % 4)
    string = string + ("=" * padding)
    return base64.urlsafe_b64decode(string).decode('utf-8')

@mcp.tool
async def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]

@mcp.tool
async def add_web_scrapping_data(capture_time: datetime, data_type: str, data: dict[str, str]) -> int:
    """Add web scraping data."""
    table_path = "./delta_data/web_scraping_data"
    
    capture_time_array = []
    data_type_array = []
    data_key_array = []
    data_value_array = []

    for key, value in data.items():
        capture_time_array.append(capture_time)
        data_type_array.append(data_type)
        data_key_array.append(key)
        data_value_array.append(value)

    df = pd.DataFrame({
        "capture_time": capture_time_array,
        "data_type": data_type_array,
        "data_key": data_key_array,
        "data_value": data_value_array
    })
    write_deltalake(table_path, df, mode="append")
    return len(data)

@mcp.tool
async def remove_web_scrapping_data() -> int:
    """Add web scraping data."""
    table_path = "./delta_data/web_scraping_data"

    dt = DeltaTable(table_path)
    dt.delete()
    return 0

@mcp.tool(output_schema={
    "type": "object", 
    "properties": {
        "data": {
            "type": "array", 
            "items": {
                "type": "object",
                "properties": {
                    "capture_time": {"type": "string", "format": "date-time"},
                    "data_type": {"type": "string"},
                    "data_key": {"type": "string"},
                    "data_value": {"type": "string"}
                }
            }
        }
    }
})
async def query_web_scrapping_data(start_time: datetime, end_time: datetime, data_type: str) -> list[dict]:
    """Query web scraping data."""
    try: 
        table_path = "./delta_data/web_scraping_data"
        dt = DeltaTable(table_path)
        dataset = dt.to_pyarrow_dataset()
        condition = (ds.field("capture_time") >= pd.Timestamp(start_time)) & (ds.field("capture_time") <= pd.Timestamp(end_time)) & (ds.field("data_type") == data_type)
        #condition = (ds.field("data_type") == data_type)

        # Filter data based on time range and data type
        filtered_data = dataset.to_table(
            filter=condition,
            columns=["capture_time", "data_type", "data_key", "data_value"]
        ).to_pandas()

        print(f"Filtered data:\n{filtered_data}")
        
        data_list = []
        for index, row in filtered_data.iterrows():
            print(f"Row {index}: capture_time={row['capture_time']}, data_type={row['data_type']}, data_key={row['data_key']}, data_value={row['data_value']}")
            data_list.append({
                "capture_time": row['capture_time'].isoformat(),
                "data_type": row['data_type'],
                "data_key": row['data_key'],
                "data_value": row['data_value']
            })

       
    except Exception as e:
        logger.error(f"Error querying web scraping data: {e}")
        data_list = []

    return ToolResult(
        content=[TextContent(type="text", text="Human-readable summary")],
        structured_content={"data": data_list},
        meta={"execution_time_ms": 145}
    )

@mcp.tool
async def save_chat_conversation_item(capture_time: datetime, session_id: str, user_id: str, who_talks: str, message: str) -> int:
    """Add chat conversation data."""
    table_path = "./delta_data/chat_conversation_data"

    df = pd.DataFrame({
        "capture_time": [capture_time],
        "session_id": [session_id],
        "user_id": [user_id],
        "who_talks": [who_talks],
        "message": [message]
    })
    logger.info(f"Saving chat conversation item: {df.to_dict(orient='records')}")
    write_deltalake(table_path, df, mode="append")
    logger.info("Chat conversation item saved successfully.")
    #logger.info(f"Chat conversation item - capture_time: {capture_time}, session_id: {session_id}, user_id: {user_id}, who_talks: {who_talks}, message: {message}")
    return len(message)

@mcp.resource("resource://{question}/subject")
async def get_subject(question: str, ctx: Context) -> dict:
    """Get details for a specific name."""
    minimum_distance = {"hora": 0.75, "index": 0.75, "musica": 0.75, "para-musica": 0.75, "temps": 0.75, "train": 0.75} # threshold for similarity. if more distance, then marked as "out of scope"
    distances = {}
    decoded_question = base64_decode(question)
    logger.info("Decoded question: %s", decoded_question)
    query_vectors = embedding_fn.encode_queries([decoded_question])

    result = client.search(
        collection_name="intents",  # target collection
        data=query_vectors,  # query vectors
        limit=10,  # number of returned entities
        output_fields=["text", "subject"],  # specifies fields to be returned
        search_params={"metric_type": "IP"} # to determine the similarity it will be used this metric
    )

    #logger.info("Milvus search result repr: %r", result)

    # Defensive extraction: Milvus client can return different structures (list of dicts, list of lists,
    # or objects). Try several access patterns and fall back to empty strings when unavailable.
    text_val = "out of scope"
    subject_val = "out_of_scope"
    try:
        if not result:
            raise ValueError("empty result")

        for result_item in result:
            logger.info("Processing result item: %s", result_item)
            if len(result_item) > 0:
                for element in result_item:
                    if element.entity:
                        current_subject = element.entity.subject
                        if distances.get(current_subject) is None:
                            distances[current_subject] = {
                                "distance": element.distance,
                                "id": element.id,
                                "text": element.entity.text
                            }
                        

                        #logger.info("entity fields: %s", element.entity.subject)
                        #logger.info("entity text: %s", element.entity.text)
                    #logger.info("Element: %s", element)
                    #logger.info("distance: %s", element.distance)
                    #logger.info("distance: %s", element.id)
        for key, value in distances.items():
            logger.info("Distance item %s: %s", key, value)
            if value["distance"] > minimum_distance.get(key):
                subject_val = key
                text_val = value["text"]
                break


            #candidate = result_item[0] if len(result_item) > 0 else {}
            #text_val = getattr(candidate, "text", str(candidate))
            #subject_val = getattr(candidate, "subject", "")


        

        # If first is a dict-like
        # if isinstance(first, dict):
        #     logger.info("First result is a dict: %s", first)
        #     text_val = first.get("text", "")
        #     subject_val = first.get("subject", "")

        # If first is a list/tuple (e.g., list of hits), try to access its first element
        # elif isinstance(first, (list, tuple)):
        #     logger.info("First result is a list/tuple: %s", first)
        #     candidate = first[0] if len(first) > 0 else {}
        #     if isinstance(candidate, dict):
        #         logger.info("Candidate is a dict: %s", candidate)
        #         text_val = candidate.get("text", "")
        #         subject_val = candidate.get("subject", "")
        #     else:
        #         text_val = getattr(candidate, "text", str(candidate))
        #         subject_val = getattr(candidate, "subject", "")
        # else:
        #     logger.info("First result is an object: %s", first)
        #     # Try attribute access as a last resort
        #     text_val = getattr(first, "text", None)
        #     subject_val = getattr(first, "subject", None)
        #     if text_val is None:
        #         text_val = str(first)

    except Exception as e:
        logger.exception("Failed to extract text/subject from Milvus result: %s", e)

    return {
        "text": text_val or "",
        "subject": subject_val or "",
        "accessed_at": ctx.request_id,
    }

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8182)