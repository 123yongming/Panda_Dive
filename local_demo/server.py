"""Local Panda_Dive demo server with streaming SSE support for real-time progress."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

active_jobs: dict[str, asyncio.Queue] = {}

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_root_env():
    if load_dotenv is None:
        return
    load_dotenv(ROOT / ".env", override=False)


_load_root_env()


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _ensure_runtime_compat():
    os.environ.setdefault("GET_API_KEYS_FROM_CONFIG", "false")


class SSEConnection:
    """Helper class to send Server-Sent Events (SSE) through HTTP response."""

    def __init__(self, handler: BaseHTTPRequestHandler):
        """Initialize SSE connection with HTTP request handler."""
        self.handler = handler

    def send_event(self, event_type: str, data: dict):
        """Send an SSE event with specified type and data payload."""
        event_data = json.dumps(data, ensure_ascii=False)
        self.handler.wfile.write(f"event: {event_type}\n".encode())
        self.handler.wfile.write(f"data: {event_data}\n\n".encode())
        self.handler.wfile.flush()


async def run_research_streaming(job_id: str, topic: str, settings: dict[str, Any]):
    """Run research with streaming events for a given topic.

    This function streams real-time progress updates via SSE while
    executing the Panda_Dive research pipeline.
    """
    from langchain_core.messages import HumanMessage

    from Panda_Dive import Configuration, deep_researcher

    _ensure_runtime_compat()

    event_queue = active_jobs.get(job_id)
    if not event_queue:
        return

    async def send_event(event_type: str, payload: dict):
        await event_queue.put({"type": event_type, "payload": payload})

    try:
        await send_event(
            "research_started",
            {"topic": topic, "timestamp": time.time(), "message": "开始研究..."},
        )

        search_api = settings.get("search_api", "duckduckgo")
        if search_api not in {"duckduckgo", "tavily", "none"}:
            search_api = "duckduckgo"

        config = Configuration(
            search_api=search_api,
            max_researcher_iterations=_clamp_int(
                settings.get("max_researcher_iterations"), 1, 8, 4
            ),
            max_concurrent_research_units=_clamp_int(
                settings.get("max_concurrent_research_units"), 1, 8, 3
            ),
            allow_clarification=_safe_bool(settings.get("allow_clarification"), False),
            query_variants=_clamp_int(settings.get("query_variants"), 1, 6, 3),
            relevance_threshold=_clamp_float(
                settings.get("relevance_threshold"), 0.0, 1.0, 0.7
            ),
            rerank_top_k=_clamp_int(settings.get("rerank_top_k"), 1, 20, 10),
        )

        graph_config = {"configurable": config.model_dump()}

        async for event in deep_researcher.astream_events(
            {"messages": [HumanMessage(content=topic)]},
            config=graph_config,
            version="v2",
        ):
            event_type = event.get("event")
            event_data = event.get("data", {})
            event_name = event.get("name", "")

            if event_type == "on_chain_start":
                node_name = event_name or "unknown"
                await send_event(
                    "node_start",
                    {
                        "node": node_name,
                        "timestamp": time.time(),
                        "message": f"开始执行: {node_name}",
                    },
                )

            elif event_type == "on_chain_end":
                node_name = event_name or "unknown"
                await send_event(
                    "node_end",
                    {
                        "node": node_name,
                        "timestamp": time.time(),
                        "message": f"完成执行: {node_name}",
                    },
                )

            elif event_type == "on_tool_start":
                tool_name = event_name or "unknown"
                tool_input = event_data.get("input", {})
                await send_event(
                    "tool_start",
                    {
                        "tool": tool_name,
                        "timestamp": time.time(),
                        "input": str(tool_input)[:200] if tool_input else "",
                        "message": f"开始工具调用: {tool_name}",
                    },
                )

            elif event_type == "on_tool_end":
                tool_name = event_name or "unknown"
                await send_event(
                    "tool_end",
                    {
                        "tool": tool_name,
                        "timestamp": time.time(),
                        "message": f"完成工具调用: {tool_name}",
                    },
                )

            elif event_type == "on_llm_start":
                await send_event(
                    "llm_start",
                    {"timestamp": time.time(), "message": "开始 LLM 调用..."},
                )

            elif event_type == "on_llm_end":
                await send_event(
                    "llm_end", {"timestamp": time.time(), "message": "完成 LLM 调用"}
                )

        # Get final result
        result = await deep_researcher.ainvoke(
            {"messages": [HumanMessage(content=topic)]},
            config=graph_config,
        )

        final_report = result.get("final_report", "")

        await send_event(
            "research_completed",
            {
                "final_report": final_report,
                "status": "succeeded" if final_report else "failed",
                "timestamp": time.time(),
            },
        )

    except Exception as exc:
        import traceback

        logging.exception("Streaming research request failed")
        await send_event(
            "research_error",
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": time.time(),
            },
        )


class DemoHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Panda_Dive demo server."""

    server_version = "PandaDiveDemo/1.0"

    def _send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _serve_static(self, requested_path: str) -> None:
        relative = (
            "index.html" if requested_path in {"/", ""} else requested_path.lstrip("/")
        )
        target = (STATIC_DIR / relative).resolve()

        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content_type, _ = mimetypes.guess_type(str(target))
        if not content_type:
            content_type = "application/octet-stream"

        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        """Handle HTTP OPTIONS requests for CORS preflight."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Handle HTTP GET requests for health check, SSE stream, and static files."""
        if self.path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if self.path.startswith("/api/research/stream/"):
            job_id = self.path.split("/")[-1]
            self._handle_sse_stream(job_id)
            return
        self._serve_static(self.path)

    def _handle_sse_stream(self, job_id: str) -> None:
        event_queue = asyncio.Queue()
        active_jobs[job_id] = event_queue

        self._send_sse_headers()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            def send_events():
                while True:
                    try:
                        future = asyncio.ensure_future(event_queue.get())
                        event_data = loop.run_until_complete(
                            asyncio.wait_for(future, timeout=0.5)
                        )

                        if event_data is None:
                            break

                        event_type = event_data.get("type", "message")
                        payload = event_data.get("payload", {})

                        yield f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logging.exception("Error in SSE stream")
                        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                        break

            for sse_data in send_events():
                self.wfile.write(sse_data.encode("utf-8"))
                self.wfile.flush()

        except Exception:
            logging.exception("SSE connection error")
        finally:
            if job_id in active_jobs:
                del active_jobs[job_id]
            try:
                loop.close()
            except Exception:
                pass

    def do_POST(self):
        """Handle HTTP POST requests for research submission."""
        if self.path == "/api/research":
            self._handle_research_request()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _handle_research_request(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json({"error": "Empty request body"}, HTTPStatus.BAD_REQUEST)
            return

        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON payload"}, HTTPStatus.BAD_REQUEST)
            return

        topic = str(payload.get("topic", "")).strip()
        if not topic:
            self._send_json(
                {"error": "topic is required"},
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
            return

        settings = payload.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}

        job_id = str(uuid.uuid4())

        self._send_json(
            {
                "job_id": job_id,
                "status": "started",
                "sse_url": f"/api/research/stream/{job_id}",
            },
            HTTPStatus.OK,
        )

        def start_research():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_research_streaming(job_id, topic, settings))
            except Exception as e:
                logging.exception("Background research failed")
                if job_id in active_jobs:
                    queue = active_jobs[job_id]
                    asyncio.run_coroutine_threadsafe(
                        queue.put(
                            {"type": "research_error", "payload": {"error": str(e)}}
                        ),
                        loop,
                    )
            finally:
                if job_id in active_jobs:
                    queue = active_jobs[job_id]
                    try:
                        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                    except Exception:
                        pass
                loop.close()

        import threading

        thread = threading.Thread(target=start_research, daemon=True)
        thread.start()


def main():
    """Start the Panda_Dive demo HTTP server."""
    host = "127.0.0.1"
    port = 8787
    server = ThreadingHTTPServer((host, port), DemoHandler)
    logging.info("Panda_Dive demo running at http://%s:%s", host, port)
    logging.info("Streaming SSE endpoint: /api/research/stream/{job_id}")
    logging.info("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down demo server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
