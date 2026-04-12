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

    parser = argparse.ArgumentParser(description="Solver A2A server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--card-url", type=str, default=None)
    args = parser.parse_args()

    skill = AgentSkill(
        id="hybrid_solver",
        name="Hybrid ML Solver v9",
        description=(
            "v9 Hybrid solver: deterministic baseline (stacking ensemble) + "
            "toolkit-augmented LLM tree search + persistent session refinement. "
            "Receives competition tar.gz + strategy, returns best submission.csv."
        ),
        tags=["solver", "hybrid", "baseline", "tree-search", "mle-bench"],
        examples=[],
    )

    agent_card = AgentCard(
        name="TinoRex Solver v9",
        description="Hybrid ML competition solver: baseline + tree search + refinement",
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
