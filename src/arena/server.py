import argparse
import os
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Arena A2A server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--card-url", type=str, default=None)
    args = parser.parse_args()

    skill = AgentSkill(
        id="arena_hybrid",
        name="Arena Hybrid Tournament",
        description=(
            "v9 Hybrid: Receives a competition bundle and fans out to multiple "
            "solver attempts with different strategy profiles (balanced, explore-heavy, "
            "refine-heavy). Each solver runs deterministic baseline + LLM tree search + "
            "refinement. Returns the best submission."
        ),
        tags=["arena", "hybrid", "mle-bench", "pass-at-k"],
        examples=[],
    )

    agent_card = AgentCard(
        name="TinoRex Arena v9",
        description=(
            "Hybrid tournament host: dispatches ML competitions to v9 solvers "
            "that combine deterministic baselines with toolkit-augmented tree search."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="9.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
        max_content_length=None,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
