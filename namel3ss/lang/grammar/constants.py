"""Regular expression constants for grammar parsing."""

import re

# Header patterns for various declarations
DATASET_HEADER_RE = re.compile(r'^dataset\s+"([^"]+)"\s+from\s+(\w+)\s+([A-Za-z0-9_"\.]+)\s*:\s*$')
FRAME_HEADER_RE = re.compile(r'^frame\s+"([^"]+)"(?:\s+from\s+(\w+)\s+([A-Za-z0-9_"\.]+))?\s*:\s*$')
PAGE_HEADER_RE = re.compile(r'^page\s+"([^"]+)"\s+at\s+"([^"]+)"\s*:\s*$')
LLM_HEADER_RE = re.compile(r'^llm\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
TOOL_HEADER_RE = re.compile(r'^tool\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
PROMPT_HEADER_RE = re.compile(r'^prompt\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
CHAIN_HEADER_RE = re.compile(r'^define\s+chain\s+"([^"]+)"(?:\s+effect\s+([\w\-]+))?\s*:\s*$')
INDEX_HEADER_RE = re.compile(r'^index\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
RAG_PIPELINE_HEADER_RE = re.compile(r'^rag_pipeline\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$')
AGENT_HEADER_RE = re.compile(r'^agent\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$')
GRAPH_HEADER_RE = re.compile(r'^graph\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$')
POLICY_HEADER_RE = re.compile(r'^policy\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$')
APP_HEADER_RE = re.compile(
    r'^app\s+"([^"]+)"(?:\s+connects\s+to\s+[A-Za-z_][A-Za-z0-9_]*\s+"([^"]+)")?\s*\.?$'
)
MODULE_DECL_RE = re.compile(r'^module\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*$')
LANGUAGE_VERSION_RE = re.compile(r'^language_version\s+"([0-9]+\.[0-9]+\.[0-9]+)"\s*\.?$')
IMPORT_TARGET_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?\s*$')

__all__ = [
    'DATASET_HEADER_RE',
    'FRAME_HEADER_RE',
    'PAGE_HEADER_RE',
    'LLM_HEADER_RE',
    'TOOL_HEADER_RE',
    'PROMPT_HEADER_RE',
    'CHAIN_HEADER_RE',
    'INDEX_HEADER_RE',
    'RAG_PIPELINE_HEADER_RE',
    'AGENT_HEADER_RE',
    'GRAPH_HEADER_RE',
    'POLICY_HEADER_RE',
    'APP_HEADER_RE',
    'MODULE_DECL_RE',
    'LANGUAGE_VERSION_RE',
    'IMPORT_TARGET_RE',
]
