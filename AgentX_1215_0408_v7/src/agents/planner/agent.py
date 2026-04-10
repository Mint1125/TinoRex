"""
Planner Agent — Sole responsibility: form a concise ML solution plan.
Input : text message with INSTRUCTIONS + DATA_SUMMARY sections.
Output: PLAN_START\n{plan}\nPLAN_END
"""
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message
from openai import AsyncOpenAI

_API_KEY_FILE = Path(r"C:/Users/PC4/OneDrive/바탕 화면/개인/개인정보/api_key.txt")
OPENAI_API_KEY = [
    line.strip()
    for line in _API_KEY_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
][0]

SYSTEM_PROMPT = """\
You are a Machine Learning Engineer with 10 years of experience in Kaggle competitions.

Your sole task right now is to write a CONCISE solution plan — no code.

Cover exactly these 5 points:
1. Task type & evaluation metric
2. Key preprocessing steps
3. Feature engineering priorities (based on data summary)
4. Model choice and why
5. Potential pitfalls to watch for

Be specific, actionable, and brief (under 400 words).
"""


class PlannerAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        payload = get_message_text(message) or ""
        if not payload.strip():
            await updater.failed(new_agent_text_message("[Planner] No input received."))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Planner] Forming solution plan..."),
        )

        resp = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.3,
            max_tokens=600,
        )

        plan = resp.choices[0].message.content.strip()
        print(f"[Planner] Plan formed ({len(plan)} chars)")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"PLAN_START\n{plan}\nPLAN_END"),
        )
