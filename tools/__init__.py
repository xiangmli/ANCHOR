from tools.base import Tool
from tools.summarize import TextSummarizer
from tools.wikipedia import Wikipedia
from tools.info_database import InfoDatabase
from tools.interaction import InteractionRetriever

TOOL_MAP: dict[str, type] = {
    'summarize': TextSummarizer,
    'wikipedia': Wikipedia,
    'info': InfoDatabase,
    'interaction': InteractionRetriever,
}
