import asyncio
import base64
import json
import logging
import random
import sys
import threading

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds

from deltalake import DeltaTable, write_deltalake

from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from mcp.types import TextContent

from pymilvus import DataType, MilvusClient, model
from pymilvus.exceptions import MilvusException


# =============================================================================
# Configuration
# =============================================================================

MILVUS_URI = "http://milvus-standalone:19530"

INTENTS_COLLECTION = "intents"
INTENTS_DATA_DIR = Path(__file__).resolve().parent / "intent_data"

DEFAULT_SUBJECT = "unknown"

CHAT_CONVERSATION_PATH = "./delta_data/chat_conversation_data"
WEB_SCRAPING_PATH = "./delta_data/web_scraping_data"


INTENT_THRESHOLDS = {
    "hora": 0.75,
    "index": 0.75,
    "musica": 0.75,
    "para-musica": 0.75,
    "temps": 0.75,
    "train": 0.75,
}


UTC_TIMESTAMP = pa.timestamp(
    "us",
    tz="UTC",
)


CHAT_CONVERSATION_SCHEMA = pa.schema(
    [
        pa.field(
            "capture_time",
            UTC_TIMESTAMP,
        ),
        pa.field(
            "session_id",
            pa.string(),
        ),
        pa.field(
            "user_id",
            pa.string(),
        ),
        pa.field(
            "who_talks",
            pa.string(),
        ),
        pa.field(
            "message",
            pa.string(),
        ),
    ]
)


WEB_SCRAPING_SCHEMA = pa.schema(
    [
        pa.field(
            "capture_time",
            UTC_TIMESTAMP,
        ),
        pa.field(
            "data_type",
            pa.string(),
        ),
        pa.field(
            "data_key",
            pa.string(),
        ),
        pa.field(
            "data_value",
            pa.string(),
        ),
    ]
)


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(
    "ermes.mcp"
)

logger.setLevel(
    logging.DEBUG
)

logger.propagate = False


if not logger.handlers:
    handler = logging.StreamHandler(
        sys.stdout
    )

    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] "
            "thread=%(thread)s:"
            "%(levelname)s: "
            "%(message)s"
        )
    )

    logger.addHandler(
        handler
    )


# =============================================================================
# MCP
# =============================================================================

mcp = FastMCP(
    name="Dice Roller"
)


# =============================================================================
# Lazy external dependencies
# =============================================================================

_milvus_client = None
_embedding_fn = None

_milvus_client_lock = threading.Lock()
_intents_collection_lock = threading.RLock()
_embedding_fn_lock = threading.Lock()


def get_milvus_client() -> MilvusClient:
    """
    Lazily create the Milvus client.

    MCP is allowed to start even when Milvus is unavailable.
    """

    global _milvus_client

    if _milvus_client is not None:
        return _milvus_client

    with _milvus_client_lock:
        if _milvus_client is not None:
            return _milvus_client

        logger.info(
            "Connecting to Milvus at %s",
            MILVUS_URI,
        )

        _milvus_client = MilvusClient(
            uri=MILVUS_URI,
        )

        logger.info(
            "Milvus client initialized"
        )

        return _milvus_client


def reset_milvus_client():
    """
    Discard the current Milvus connection after an error.

    A later request will attempt to reconnect automatically.
    """

    global _milvus_client

    with _milvus_client_lock:
        current_client = _milvus_client
        _milvus_client = None

    if current_client is not None:
        try:
            current_client.close()

        except Exception:
            logger.debug(
                "Error closing Milvus client",
                exc_info=True,
            )


def get_embedding_function():
    """
    Lazily initialize the embedding model.

    A problem loading the model must not prevent MCP from starting.
    """

    global _embedding_fn

    if _embedding_fn is not None:
        return _embedding_fn

    with _embedding_fn_lock:
        if _embedding_fn is not None:
            return _embedding_fn

        logger.info(
            "Initializing embedding function"
        )

        _embedding_fn = (
            model.DefaultEmbeddingFunction()
        )

        logger.info(
            "Embedding function initialized"
        )

        return _embedding_fn


# =============================================================================
# Generic helpers
# =============================================================================

def ensure_utc_datetime(
    value: datetime,
) -> datetime:
    """
    Return a timezone-aware UTC datetime.

    FastMCP normally provides timezone-aware values, but this keeps the
    Delta schemas stable even if a naive datetime reaches the MCP tool.
    """

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    )


def base64_decode(
    string: str,
) -> str:
    """
    Restore missing URL-safe Base64 padding and decode the string.
    """

    padding = (
        -len(string)
    ) % 4

    padded = (
        string
        + ("=" * padding)
    )

    return base64.urlsafe_b64decode(
        padded
    ).decode(
        "utf-8"
    )


def build_subject_response(
    ctx: Context,
    *,
    subject: str,
    text: str = "",
    routing_status: str = "ok",
    reason: str | None = None,
) -> dict:
    """
    Build the standard intent-classification response.

    routing_status:
        ok
            The classifier worked normally.

        degraded
            Classification could not be performed reliably and the caller
            should use a safe fallback.
    """

    response = {
        "text": text,
        "subject": subject,
        "routing_status": routing_status,
        "accessed_at": ctx.request_id,
    }

    if reason is not None:
        response[
            "reason"
        ] = reason

    return response


def extract_entity_field(
    entity,
    field_name: str,
):
    """
    Read a field from the different entity representations PyMilvus may
    return.
    """

    if entity is None:
        return None

    if isinstance(
        entity,
        dict,
    ):
        return entity.get(
            field_name
        )

    return getattr(
        entity,
        field_name,
        None,
    )


# =============================================================================
# Test tool
# =============================================================================

@mcp.tool
async def roll_dice(
    n_dice: int,
) -> list[int]:
    """
    Roll n_dice six-sided dice.
    """

    return [
        random.randint(
            1,
            6,
        )
        for _ in range(
            n_dice
        )
    ]


# =============================================================================
# Web scraping persistence
# =============================================================================

def _write_web_scraping_data(
    capture_time: datetime,
    data_type: str,
    data: dict[str, str],
) -> int:
    """
    Blocking Delta write executed outside the asyncio event loop.

    PyArrow is used directly; pandas is deliberately not used because
    pandas 2.3.2 crashes in this Python 3.14 image while handling datetime
    arrays.
    """

    Path(
        WEB_SCRAPING_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture_time = (
        ensure_utc_datetime(
            capture_time
        )
    )

    rows = [
        {
            "capture_time":
                capture_time,

            "data_type":
                data_type,

            "data_key":
                str(key),

            "data_value":
                str(value),
        }
        for key, value
        in data.items()
    ]

    if not rows:
        return 0

    table = pa.Table.from_pylist(
        rows,
        schema=WEB_SCRAPING_SCHEMA,
    )

    write_deltalake(
        WEB_SCRAPING_PATH,
        table,
        mode="append",
    )

    return len(
        rows
    )


@mcp.tool
async def add_web_scrapping_data(
    capture_time: datetime,
    data_type: str,
    data: dict[str, str],
) -> int:
    """
    Add web scraping data.

    Persistence is best-effort: a storage failure is logged and returned as
    -1 rather than killing the caller.
    """

    try:
        result = await asyncio.to_thread(
            _write_web_scraping_data,
            capture_time,
            data_type,
            data,
        )

        logger.info(
            "Saved %d web scraping items",
            result,
        )

        return result

    except Exception:
        logger.exception(
            "Failed to save web scraping data"
        )

        return -1


def _remove_web_scraping_data():
    """
    Blocking Delta delete.
    """

    dt = DeltaTable(
        WEB_SCRAPING_PATH
    )

    dt.delete()


@mcp.tool
async def remove_web_scrapping_data() -> int:
    """
    Remove all web scraping data.
    """

    try:
        await asyncio.to_thread(
            _remove_web_scraping_data
        )

        logger.info(
            "Web scraping data removed"
        )

        return 0

    except Exception:
        logger.exception(
            "Failed to remove web scraping data"
        )

        return -1


def _query_web_scraping_data(
    start_time: datetime,
    end_time: datetime,
    data_type: str,
) -> list[dict]:
    """
    Blocking Delta query using PyArrow only.
    """

    start_time = (
        ensure_utc_datetime(
            start_time
        )
    )

    end_time = (
        ensure_utc_datetime(
            end_time
        )
    )

    dt = DeltaTable(
        WEB_SCRAPING_PATH
    )

    dataset = (
        dt.to_pyarrow_dataset()
    )

    start_scalar = pa.scalar(
        start_time,
        type=UTC_TIMESTAMP,
    )

    end_scalar = pa.scalar(
        end_time,
        type=UTC_TIMESTAMP,
    )

    condition = (
        (
            ds.field(
                "capture_time"
            )
            >= start_scalar
        )
        &
        (
            ds.field(
                "capture_time"
            )
            <= end_scalar
        )
        &
        (
            ds.field(
                "data_type"
            )
            == data_type
        )
    )

    table = dataset.to_table(
        filter=condition,
        columns=[
            "capture_time",
            "data_type",
            "data_key",
            "data_value",
        ],
    )

    data_list = []

    for row in table.to_pylist():
        capture_time_value = (
            row[
                "capture_time"
            ]
        )

        if isinstance(
            capture_time_value,
            datetime,
        ):
            capture_time_text = (
                capture_time_value
                .isoformat()
            )

        else:
            capture_time_text = str(
                capture_time_value
            )

        data_list.append(
            {
                "capture_time":
                    capture_time_text,

                "data_type":
                    row[
                        "data_type"
                    ],

                "data_key":
                    row[
                        "data_key"
                    ],

                "data_value":
                    row[
                        "data_value"
                    ],
            }
        )

    return data_list


@mcp.tool(
    output_schema={
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "capture_time": {
                            "type": "string",
                            "format": "date-time",
                        },
                        "data_type": {
                            "type": "string",
                        },
                        "data_key": {
                            "type": "string",
                        },
                        "data_value": {
                            "type": "string",
                        },
                    },
                },
            }
        },
    }
)
async def query_web_scrapping_data(
    start_time: datetime,
    end_time: datetime,
    data_type: str,
) -> list[dict]:
    """
    Query web scraping data.

    An unavailable/missing Delta table returns an empty list and records the
    failure in the logs rather than propagating it to routing.
    """

    try:
        data_list = await asyncio.to_thread(
            _query_web_scraping_data,
            start_time,
            end_time,
            data_type,
        )

        logger.debug(
            "Web scraping query returned %d rows",
            len(
                data_list
            ),
        )

    except Exception:
        logger.exception(
            "Error querying web scraping data"
        )

        data_list = []

    return ToolResult(
        content=[
            TextContent(
                type="text",
                text="Human-readable summary",
            )
        ],
        structured_content={
            "data":
                data_list
        },
        meta={
            "execution_time_ms":
                0
        },
    )


# =============================================================================
# Conversation persistence
# =============================================================================

def _write_chat_conversation_item(
    capture_time: datetime,
    session_id: str,
    user_id: str,
    who_talks: str,
    message: str,
) -> int:
    """
    Blocking Delta write using PyArrow directly.

    This intentionally avoids pandas. The Python 3.14 MCP image currently
    contains pandas 2.3.2, which was observed to segfault while constructing
    a timezone-aware datetime DataFrame.
    """

    Path(
        CHAT_CONVERSATION_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture_time = (
        ensure_utc_datetime(
            capture_time
        )
    )

    table = pa.Table.from_pylist(
        [
            {
                "capture_time":
                    capture_time,

                "session_id":
                    session_id,

                "user_id":
                    user_id,

                "who_talks":
                    who_talks,

                "message":
                    message,
            }
        ],
        schema=(
            CHAT_CONVERSATION_SCHEMA
        ),
    )

    write_deltalake(
        CHAT_CONVERSATION_PATH,
        table,
        mode="append",
    )

    return len(
        message
    )


@mcp.tool
async def save_chat_conversation_item(
    capture_time: datetime,
    session_id: str,
    user_id: str,
    who_talks: str,
    message: str,
) -> int:
    """
    Persist a conversation message.

    Persistence is intentionally best-effort.

    A persistence failure MUST NOT make message routing fail.

    Returns:
        >= 0
            number of characters persisted

        -1
            persistence failed
    """

    logger.info(
        "Saving conversation item: "
        "session_id=%s user_id=%s "
        "who_talks=%s message_length=%d",
        session_id,
        user_id,
        who_talks,
        len(
            message
        ),
    )

    try:
        result = await asyncio.to_thread(
            _write_chat_conversation_item,
            capture_time,
            session_id,
            user_id,
            who_talks,
            message,
        )

    except Exception:
        logger.exception(
            "Conversation persistence failed. "
            "Routing will continue."
        )

        return -1

    logger.info(
        "Conversation item saved successfully"
    )

    return result


# =============================================================================
# Intent classification
# =============================================================================

def build_intents_schema():
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=512)

    indexes = MilvusClient.prepare_index_params()
    indexes.add_index(field_name="id", index_type="AUTOINDEX")
    indexes.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="IP")
    return schema, indexes


@mcp.tool
def create_intents_schema() -> dict:
    """Build and return the intended Milvus schema and indexes for the intents collection."""
    schema, indexes = build_intents_schema()
    return {
        "collection": INTENTS_COLLECTION,
        "schema": schema.to_dict(),
        "indexes": [
            {"field_name": index.field_name, **index.get_index_configs()}
            for index in indexes
        ],
    }


def ensure_intents_collection() -> bool:
    with _intents_collection_lock:
        client = get_milvus_client()
        if client.has_collection(INTENTS_COLLECTION, timeout=10):
            return False

        schema, indexes = build_intents_schema()
        client.create_collection(
            collection_name=INTENTS_COLLECTION,
            schema=schema,
            index_params=indexes,
            timeout=60,
        )
        return True


@mcp.tool
async def create_intents_collection() -> dict:
    """Create the Milvus intents collection if it is missing. Leave an existing collection intact."""
    try:
        created = await asyncio.to_thread(ensure_intents_collection)
    except MilvusException as exc:
        logger.exception("Failed to ensure the Milvus intents collection exists")
        raise ToolError(f"Could not create the Milvus intents collection: {exc.message}") from exc
    return {"collection": INTENTS_COLLECTION, "created": created}


def remove_intents_collection() -> bool:
    with _intents_collection_lock:
        client = get_milvus_client()
        if not client.has_collection(INTENTS_COLLECTION, timeout=10):
            return False

        client.drop_collection(collection_name=INTENTS_COLLECTION, timeout=60)
        return True


@mcp.tool(annotations={"destructiveHint": True, "idempotentHint": True})
async def drop_intents_collection() -> dict:
    """Permanently delete the Milvus intents collection and all stored utterances."""
    try:
        dropped = await asyncio.to_thread(remove_intents_collection)
    except MilvusException as exc:
        logger.exception("Failed to drop the Milvus intents collection")
        raise ToolError(f"Could not drop the Milvus intents collection: {exc.message}") from exc
    return {"collection": INTENTS_COLLECTION, "dropped": dropped}


def read_intents() -> dict:
    client = get_milvus_client()
    if not client.has_collection(INTENTS_COLLECTION, timeout=10):
        raise ToolError(f"The Milvus collection '{INTENTS_COLLECTION}' does not exist.")

    iterator = client.query_iterator(
        collection_name=INTENTS_COLLECTION,
        batch_size=500,
        output_fields=["id", "subject", "text"],
        consistency_level="Strong",
        timeout=10,
    )
    try:
        rows = []
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()

    rows.sort(key=lambda row: (row["subject"], row["id"]))
    return {"intents": rows}


@mcp.tool
async def list_intents() -> dict:
    """List the stored intent subjects and their example utterances from Milvus."""
    try:
        return await asyncio.to_thread(read_intents)
    except MilvusException as exc:
        logger.exception("Failed to list stored Milvus intents")
        raise ToolError(f"Could not list Milvus intents: {exc.message}") from exc


def read_intent_examples() -> dict[str, list[str]]:
    paths = sorted(INTENTS_DATA_DIR.glob("*.json"))
    if not paths:
        raise ToolError(f"No intent JSON files found in {INTENTS_DATA_DIR}.")

    examples = {}
    for path in paths:
        try:
            with path.open(encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, ValueError) as exc:
            raise ToolError(f"Could not read {path.name}: {exc}") from exc

        utterances = payload.get("utterances") if isinstance(payload, dict) else None
        if not isinstance(utterances, list) or any(
            not isinstance(text, str) or not text.strip() for text in utterances
        ):
            raise ToolError(f"{path.name} must contain an 'utterances' array of non-empty strings.")
        texts = list(dict.fromkeys(text.strip() for text in utterances))
        if any(len(value.encode("utf-8")) > 512 for value in [path.stem, *texts]):
            raise ToolError(f"A subject or utterance in {path.name} exceeds 512 UTF-8 bytes.")
        examples[path.stem] = texts

    if not any(examples.values()):
        raise ToolError("The intent JSON files contain no utterances.")
    return examples


def load_intents_from_files() -> dict:
    examples = read_intent_examples()
    with _intents_collection_lock:
        created = ensure_intents_collection()
        client = get_milvus_client()
        existing = [] if created else read_intents()["intents"]
        known = {(row["subject"], row["text"].strip()) for row in existing}
        next_id = max((row["id"] for row in existing), default=-1) + 1
        rows = []
        inserted_by_subject = {}
        for subject, texts in examples.items():
            pending = [text for text in texts if (subject, text) not in known]
            inserted_by_subject[subject] = len(pending)
            if not pending:
                continue
            vectors = get_embedding_function().encode_documents(pending)
            for text, vector in zip(pending, vectors, strict=True):
                rows.append({
                    "id": next_id,
                    "vector": vector.tolist(),
                    "text": text,
                    "subject": subject,
                })
                next_id += 1

        for offset in range(0, len(rows), 500):
            client.insert(INTENTS_COLLECTION, data=rows[offset:offset + 500], timeout=60)

        return {
            "collection": INTENTS_COLLECTION,
            "created": created,
            "inserted": len(rows),
            "skipped": sum(map(len, examples.values())) - len(rows),
            "inserted_by_subject": inserted_by_subject,
        }


@mcp.tool
async def load_intents() -> dict:
    """Embed configured intent JSON utterances and add missing examples to Milvus."""
    try:
        return await asyncio.to_thread(load_intents_from_files)
    except MilvusException as exc:
        logger.exception("Failed to load intent embeddings")
        raise ToolError(f"Could not load intent embeddings: {exc.message}") from exc


@mcp.resource(
    "resource://{question}/subject"
)
async def get_subject(
    question: str,
    ctx: Context,
) -> dict:
    """
    Semantic intent classification.

    Behaviour:

    successful classification
        -> actual subject, routing_status=ok

    classifier works but no known intent exceeds its threshold
        -> out_of_scope, routing_status=ok

    missing/empty intent collection
        -> unknown, routing_status=degraded

    Milvus unavailable
        -> unknown, routing_status=degraded

    embedding/internal failure
        -> unknown, routing_status=degraded
    """

    # -------------------------------------------------------------------------
    # Decode input
    # -------------------------------------------------------------------------

    try:
        decoded_question = (
            base64_decode(
                question
            )
        )

    except Exception:
        logger.exception(
            "Intent classification degraded: "
            "invalid Base64 question"
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason=(
                "invalid_question_encoding"
            ),
        )

    logger.info(
        "Intent classification request: %s",
        decoded_question,
    )


    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    try:
        embedding_fn = (
            await asyncio.to_thread(
                get_embedding_function
            )
        )

        query_vectors = (
            await asyncio.to_thread(
                embedding_fn.encode_queries,
                [
                    decoded_question
                ],
            )
        )

    except Exception:
        logger.exception(
            "Intent classification degraded: "
            "embedding generation failed"
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason="embedding_failed",
        )


    # -------------------------------------------------------------------------
    # Milvus connection
    # -------------------------------------------------------------------------

    try:
        client = await asyncio.to_thread(
            get_milvus_client
        )

    except Exception:
        logger.exception(
            "Intent classification degraded: "
            "Milvus unavailable"
        )

        await asyncio.to_thread(
            reset_milvus_client
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason="milvus_unavailable",
        )


    # -------------------------------------------------------------------------
    # Collection check
    # -------------------------------------------------------------------------

    try:
        collection_exists = (
            await asyncio.to_thread(
                client.has_collection,
                collection_name=(
                    INTENTS_COLLECTION
                ),
            )
        )

    except MilvusException:
        logger.exception(
            "Intent classification degraded: "
            "failed checking Milvus collection"
        )

        await asyncio.to_thread(
            reset_milvus_client
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason="milvus_unavailable",
        )

    except Exception:
        logger.exception(
            "Intent classification degraded: "
            "unexpected collection check failure"
        )

        await asyncio.to_thread(
            reset_milvus_client
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason=(
                "milvus_collection_check_failed"
            ),
        )


    if not collection_exists:
        logger.warning(
            "Intent classification degraded: "
            "collection '%s' does not exist",
            INTENTS_COLLECTION,
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason=(
                "intents_collection_missing"
            ),
        )


    # -------------------------------------------------------------------------
    # Semantic search
    # -------------------------------------------------------------------------

    try:
        result = await asyncio.to_thread(
            client.search,
            collection_name=(
                INTENTS_COLLECTION
            ),
            data=query_vectors,
            limit=10,
            output_fields=[
                "text",
                "subject",
            ],
            search_params={
                "metric_type":
                    "IP"
            },
        )

    except MilvusException:
        logger.exception(
            "Intent classification degraded: "
            "Milvus search failed"
        )

        await asyncio.to_thread(
            reset_milvus_client
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason="milvus_search_failed",
        )

    except Exception:
        logger.exception(
            "Intent classification degraded: "
            "unexpected semantic search failure"
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason=(
                "classifier_internal_error"
            ),
        )


    # -------------------------------------------------------------------------
    # Empty search
    # -------------------------------------------------------------------------

    if not result:
        logger.warning(
            "Intent classification degraded: "
            "collection '%s' contains no searchable intents",
            INTENTS_COLLECTION,
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason="no_intents_available",
        )


    # -------------------------------------------------------------------------
    # Parse candidates
    # -------------------------------------------------------------------------

    best_subject = None
    best_text = None
    best_distance = None

    found_any_hit = False

    try:
        for result_item in result:

            if not result_item:
                continue

            for element in result_item:

                found_any_hit = True

                entity = getattr(
                    element,
                    "entity",
                    None,
                )

                subject = (
                    extract_entity_field(
                        entity,
                        "subject",
                    )
                )

                text = (
                    extract_entity_field(
                        entity,
                        "text",
                    )
                )

                distance = getattr(
                    element,
                    "distance",
                    None,
                )

                logger.debug(
                    "Intent candidate: "
                    "subject=%s "
                    "distance=%s "
                    "text=%s",
                    subject,
                    distance,
                    text,
                )


                if subject is None:
                    logger.warning(
                        "Ignoring Milvus hit "
                        "without subject"
                    )

                    continue


                threshold = (
                    INTENT_THRESHOLDS.get(
                        subject
                    )
                )


                if threshold is None:
                    logger.warning(
                        "Ignoring unsupported "
                        "intent returned by Milvus: %s",
                        subject,
                    )

                    continue


                if distance is None:
                    logger.warning(
                        "Ignoring intent '%s' "
                        "without similarity score",
                        subject,
                    )

                    continue


                if distance < threshold:
                    continue


                if (
                    best_distance is None
                    or distance > best_distance
                ):
                    best_subject = (
                        subject
                    )

                    best_text = (
                        text
                    )

                    best_distance = (
                        distance
                    )

    except Exception:
        logger.exception(
            "Intent classification degraded: "
            "invalid Milvus search result"
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason="invalid_milvus_result",
        )


    # -------------------------------------------------------------------------
    # Collection exists but contains no hits
    # -------------------------------------------------------------------------

    if not found_any_hit:
        logger.warning(
            "Intent classification degraded: "
            "collection '%s' has no intent entries",
            INTENTS_COLLECTION,
        )

        return build_subject_response(
            ctx,
            subject=DEFAULT_SUBJECT,
            routing_status="degraded",
            reason="no_intents_available",
        )


    # -------------------------------------------------------------------------
    # Valid classifier, no intent above threshold
    # -------------------------------------------------------------------------

    if best_subject is None:

        logger.info(
            "Intent classification result: "
            "out_of_scope"
        )

        return build_subject_response(
            ctx,
            subject="out_of_scope",
            text="out of scope",
            routing_status="ok",
            reason=(
                "no_match_above_threshold"
            ),
        )


    # -------------------------------------------------------------------------
    # Successful classification
    # -------------------------------------------------------------------------

    logger.info(
        "Intent classified: "
        "subject=%s similarity=%s",
        best_subject,
        best_distance,
    )

    return build_subject_response(
        ctx,
        subject=best_subject,
        text=best_text or "",
        routing_status="ok",
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    logger.info(
        "Starting ERMES MCP server"
    )

    try:
        logger.info(
            "Warming up embedding model"
        )

        get_embedding_function()

        logger.info(
            "Embedding model ready"
        )

    except Exception:
        logger.exception(
            "Embedding warm-up failed. "
            "MCP will start in degraded mode "
            "and retry when intent classification "
            "is requested."
        )

    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8182,
    )
