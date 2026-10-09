"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )
    internal_db_url: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": (
                    "Connection string for the internal database that the "
                    "data_analyst researcher queries. Use a read-only "
                    "database user for safety."
                )
            }
        }
    )
    
    # ---- Reliability & safety (stage 3)
    tool_timeout_s: float = Field(
        default=90.0,
        metadata={"x_oap_ui_config": {"type": "number", "default": 90, "min": 1, "max": 600,
                                      "description": "Wall-clock timeout for a single tool call."}},
    )
    tool_max_retries: int = Field(
        default=2,
        metadata={"x_oap_ui_config": {"type": "number", "default": 2, "min": 0, "max": 5,
                                      "description": "Retries (exponential backoff + jitter) for transient failures of idempotent tools."}},
    )
    breaker_failure_threshold: int = Field(
        default=3,
        metadata={"x_oap_ui_config": {"type": "number", "default": 3, "min": 1, "max": 50,
                                      "description": "Consecutive failed calls (after retries) before a tool's circuit opens."}},
    )
    breaker_cooldown_s: float = Field(
        default=15.0,
        metadata={"x_oap_ui_config": {"type": "number", "default": 15, "min": 1, "max": 600,
                                      "description": "Seconds an open circuit waits before allowing one probe call."}},
    )
    fallback_model: Optional[str] = Field(
        default=None,
        metadata={"x_oap_ui_config": {"type": "text",
                                      "description": "Optional backup model (e.g. 'deepseek:deepseek-chat') used when the research model keeps failing with timeouts, rate limits or 5xx errors."}},
    )
    kb_retrieval_mode: str = Field(
        default="hybrid",
        metadata={"x_oap_ui_config": {"type": "select", "default": "hybrid",
                                      "options": [{"label": "Hybrid (vector + BM25, RRF)", "value": "hybrid"},
                                                  {"label": "Vector only", "value": "vector"}],
                                      "description": "Knowledge-base retrieval strategy."}},
    )
    kb_top_k: int = Field(
        default=3,
        metadata={"x_oap_ui_config": {"type": "number", "default": 3, "min": 1, "max": 10,
                                      "description": "Chunks returned per knowledge-base search."}},
    )
    kb_candidates: int = Field(
        default=20,
        metadata={"x_oap_ui_config": {"type": "number", "default": 20, "min": 5, "max": 100,
                                      "description": "Candidates taken from each retriever before fusion."}},
    )
    kb_rerank: bool = Field(
        default=False,
        metadata={"x_oap_ui_config": {"type": "boolean", "default": False,
                                      "description": "Cross-encoder rerank of the fused candidates (+~10 pts hit@1 but ~4 s/query on CPU)."}},
    )
    schema_cache_ttl_s: float = Field(
        default=300.0,
        metadata={"x_oap_ui_config": {"type": "number", "default": 300, "min": 0, "max": 86400,
                                      "description": "Seconds to cache schema-introspection query results (0 disables). Business-data queries are never cached."}},
    )
    enable_request_triage: bool = Field(
        default=True,
        metadata={"x_oap_ui_config": {"type": "boolean", "default": True,
                                      "description": "Refuse requests to modify/delete/export data or override the rules before any research starts (no LLM call)."}},
    )
    approval_mode: str = Field(
        default="interrupt",
        metadata={"x_oap_ui_config": {"type": "select", "default": "interrupt",
                                      "options": [{"label": "Pause for human approval", "value": "interrupt"},
                                                  {"label": "Auto-approve (evals/batch)", "value": "auto_approve"},
                                                  {"label": "Deny sensitive queries", "value": "deny"}],
                                      "description": "What to do when a tool call needs human approval (row-level order data)."}},
    )
    approval_row_limit: int = Field(
        default=10,
        metadata={"x_oap_ui_config": {"type": "number", "default": 10, "min": 1, "max": 200,
                                      "description": "Row-level queries on the sales table returning more rows than this need approval."}},
    )
    approval_log_path: str = Field(
        default="logs/approvals.jsonl",
        metadata={"x_oap_ui_config": {"type": "text", "description": "Append-only JSONL audit log of approval decisions."}},
    )

    mem0_api_key: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "API key for the Mem0 Platform (cross-session long-term memory)."
            }
        }
    )
    mem0_user_id: str = Field(
        default="default_user",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": (
                    "User id that memories are scoped to. This is a single-user "
                    "demo project so a fixed value is fine; a real multi-user "
                    "deployment would derive this per authenticated end user."
                )
            }
        }
    )
    postgres_connection_string: str = Field(
        default="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "PostgreSQL(pgvector)连接串，格式为 postgresql+psycopg://user:pass@host:port/db"
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        # per-user identity must come from the authenticated request, never from a process-wide
        # environment variable - otherwise one MEM0_USER_ID would silently share memory across users
        if configurable.get("mem0_user_id"):
            values["mem0_user_id"] = configurable["mem0_user_id"]
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True