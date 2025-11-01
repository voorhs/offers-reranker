"""Main entry point for the reranker inference service."""

import os

from dotenv import load_dotenv

from loguru import logger


load_dotenv()


if __name__ == "__main__":
    from mosec import Runtime, Server

    from reranker.worker import RerankerWorker

    # Get timeout from environment (in milliseconds), default to 30 seconds
    timeout_ms = int(os.getenv("MOSEC_TIMEOUT", "30000"))
    timeout_seconds = timeout_ms // 1000

    logger.info(f"Starting reranker service with timeout: {timeout_seconds}s")

    # Create runtime for the reranker worker
    reranker_runtime = Runtime(RerankerWorker, timeout=timeout_seconds)

    # Create and configure server
    server = Server()
    server.register_runtime(
        {
            "/rerank": [reranker_runtime],
        }
    )

    logger.info("Server configured. Endpoints:")
    logger.info("  POST /rerank - Rerank offers based on query")

    # Start the server
    server.run()
