import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from .configuration import Configuration, SearchAPI
from .prompts import summarize_webpage_prompt
from .state import ResearchComplete, Summary


######
# Tavily Search Tool
######
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # 1、异步执行搜索
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    # 2、搜索结果去重
    unique_results = {}
    for response in search_results:
        for result in response["results"]:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']} # 根据query可以信息溯源
    # 3、设置summary model
    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = configurable.max_content_length
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    # 4、生成summary
    async def noop():
        return None
    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]
    # 5、并行执行summary任务
    summaries = await asyncio.gather(*summarization_tasks)
    # 6、合并summary与result
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
    }
    # 7、输出格式消息
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output
    


async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True,
    config: RunnableConfig = None
):
    """
        异步执行多个搜索查询
        Args:
            search_queries: 搜索查询列表
            max_results: 最大搜索结果数
            include_raw_content: 是否包含原始内容
        return:
            tavily api搜索结果字符串
    """
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]
    # 并行搜索
    search_results = await asyncio.gather(*search_tasks)
    return search_results


# TODO 考虑去除Open Agent Platform 生产部署
def get_tavily_api_key(config: RunnableConfig):
    """
        从RunnableConfig中获取tavily api key
    """
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", False)
    if should_get_from_config.lower() == "true":
        # Open Agent Platform 生产部署
        api_keys = config.get("configurable", {}).get("api_keys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        # 本地开发
        return os.getenv("TAVILY_API_KEY")


def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """
        从env或者Config中获取指定模型的api key
    """
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", False)
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        # Open Agent Platform 生产部署
        api_keys = config.get("configurable", {}).get("api_keys", {})
        if not api_keys:
            return None
        if model_name.startswith("ark"):
            return api_keys.get("ARK_API_KEY")
        elif model_name.startswith("deepseek"):
            return api_keys.get("DEEPSEEK_API_KEY")
        return None
    else:
        # 本地开发
        if model_name.startswith("ark"):
            return os.getenv("ARK_API_KEY")
        elif model_name.startswith("deepseek"):
            return os.getenv("DEEPSEEK_API_KEY")
        return None


def get_today_str() -> str:
    """
        获取当前日期字符串'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """
        异步生成网页摘要
        Args:
            model: 摘要模型
            webpage_content: 网页内容
        return:
            网页摘要字符串（如果失败则返回原本的内容）
    """
    try:
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content,
            date=get_today_str()
        )
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0
        )
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>\n\n"
        )
        return formatted_summary
    except asyncio.TimeoutError:
        # 超时报错
        logging.warning("Summarization timed out after 60 seconds, returning original content")
        return webpage_content
    except Exception as e:
        logging.warning(f"Summarization failed with error: {str(e)}, returning original content")
        return webpage_content

######
# Reflection Tool
######
@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection: {reflection}"


######
# MCP
######
async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """
        获取MCP访问令牌
        Args:
            supabase_token: Supabase访问令牌
            base_mcp_url: MCP基础URL
        return:
            MCP访问令牌字典（如果失败则返回None）
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    return None


async def get_tokens(config: RunnableConfig):
    """
        获取MCP访问令牌
        Args:
            config: RunnableConfig对象
        return:
            MCP访问令牌字典（如果失败则返回None）
    """
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
    user_id = config.get("configurable", {}).get("owner")
    if not user_id:
        return None
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    # 判断是否过期
    expires_in = tokens.value.get("expires_in") # 过期时长
    created_at = tokens.created_at  # 创建时间
    current_time = datetime.now(timezone.utc) # 当前时间
    expiration_time = created_at + timedelta(seconds=expires_in) # 过期时间点
    if current_time > expiration_time:
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value


async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """
        设置MCP访问令牌
        Args:
            config: RunnableConfig对象
            tokens: MCP访问令牌字典
    """
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    await store.aput((user_id, "tokens"), "data", tokens)


async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """
        智能获取MCP访问令牌
        Args:
            config: RunnableConfig对象
        return:
            MCP访问令牌字典（如果失败则返回None）
    """
    current_token = await get_tokens(config)
    if current_token:
        return current_token
    # 如果mcp token没获取到，重新生成
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None
    await set_tokens(config, mcp_tokens)
    return mcp_tokens


def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """
        包装MCP认证工具，添加错误处理和用户友好的错误消息
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """递归搜索异常链中的MCP错误"""
            if isinstance(exc, McpError):
                return exc
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        try:
            return await original_coroutine(**kwargs)
        except BaseException as original_error:
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                raise original_error
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            if error_code == -32003:
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                raise ToolException(error_message) from original_error
            raise original_error
    tool.coroutine = authentication_wrapper
    return tool


async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str]
) -> list[BaseTool]:
    """
        加载MCP工具
        Args:
            config: RunnableConfig对象
            existing_tool_names: 已存在的工具名称集合
        return:
            加载的MCP工具列表
    """
    configurable = Configuration.from_runnable_config(config)
    # 1、如果需要，处理身份认证
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    # 2、验证 MCP 配置
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    if not config_valid:
        return []
    # 3、设置链接
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # 4、加载工具
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        return []
    # 5、过滤并创建工具
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    return configured_tools
    

######
# Search Tool
######
async def get_search_tool(search_api: SearchAPI):
    """
        获取搜索工具
    """
    if search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        return []  
    # fallback策略
    return []


async def get_all_tools(config: RunnableConfig) -> list[BaseTool]:
    """
        获取所有工具
    """
    # 核心 research tool
    tools = [tool(ResearchComplete), think_tool]
    # search tool
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    # 防止冲突过滤
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    # mcp tool
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools


def get_config_value(value):
    """
        获取配置值，处理枚举类型和None值
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """
        从消息列表中提取所有工具调用消息的内容
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


######
# Model Provider Native Websearch Utils
######
# TODO 补充具有搜索功能的模型

def anthropic_websearch_called(response):
    return False

def openai_websearch_called(response):
    return False

######
# Token Limit Exceeded Utils
######
def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """
        判断异常是否因为token超出限制导致
    """
    error_str = str(exception).lower()
    # 1、确定模型供应商
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    
    # 2、根据供应商检查token超出限制
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    return False

MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}

def get_model_token_limit(model_string):
    """
        获取模型的token限制
    """
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """
        从消息列表中移除所有AI消息之前的消息：消息历史截断工具，用于处理 token 限制超出错误。
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]
    return messages