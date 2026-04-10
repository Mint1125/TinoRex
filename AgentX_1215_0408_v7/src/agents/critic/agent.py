from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from openai import AsyncOpenAI

_API_KEY_FILE = Path(r"C:/Users/PC4/OneDrive/바탕 화면/개인/개인정보/api_key.txt")
OPENAI_API_KEY = [line.strip() for line in _API_KEY_FILE.read_text(encoding="utf-8").splitlines() if line.strip()][0]

SYSTEM_PROMPT = """\
You are a Senior Data Science Reviewer at a top ML competition team.
Your role is to critically evaluate a Machine Learning Engineer's proposed approach \
before any code is written.

Your responsibilities:
- Identify flawed assumptions about the data, task type, or evaluation metric
- Flag inappropriate model choices for the given problem
- Catch missing or incorrect preprocessing steps
- Spot potential data leakage or target encoding issues
- Suggest concrete improvements

Be direct and specific. Structure your response as:

VERDICT: <APPROVED | NEEDS_REVISION | REJECTED>

ISSUES:
- <specific issue 1>
- <specific issue 2>
(omit section if none)

SUGGESTIONS:
- <concrete suggestion 1>
- <concrete suggestion 2>

SUMMARY:
<2-3 sentence overall assessment>
"""


class CriticAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        plan = get_message_text(message) or ""
        if not plan.strip():
            await updater.failed(new_agent_text_message("No plan provided to review."))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Reviewing ML Engineer's proposed approach..."),
        )

        resp = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": plan},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        critique = resp.choices[0].message.content.strip()
        print(f"[Critic Agent] Review:\n{critique}\n")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(critique),
        )
