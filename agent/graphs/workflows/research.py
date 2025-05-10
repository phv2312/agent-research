import logging
from typing import Final, Literal
from uuid import uuid4
from pydantic import BaseModel, Field
from jinja2 import Template
from langgraph.types import Command, Send, interrupt
from agent.graphs.impl import (
    TopicPlanningInput,
    TopicPlanningOutput,
    DialogOutput,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PLAN_FEEDBACK: Final[Template] = Template(
    """#### Please help me to review the below outline\n---\n{{outline}}\n\n---\n\n#### Please type `Y` to confirm or `input any feedbacks if any`"""
)


class ResearchOutput(BaseModel):
    markdown: str = Field("")


class ResearchInput(TopicPlanningInput): ...


class State(
    ResearchInput,
    TopicPlanningOutput,
    DialogOutput,
    ResearchOutput,
): ...


class LangGraphConfigurable(BaseModel):
    thread_id: str = Field(default_factory=lambda _: str(uuid4()))


class RunConfigurable(BaseModel):
    configurable: LangGraphConfigurable = Field(LangGraphConfigurable())


async def review_plan(
    state: State,
) -> Command[Literal["main-plan", "main-section-expert"]]:
    if state.outline is None or state.outline == "":
        raise ValueError("Outline is empty or None.")

    feedback = interrupt(
        PLAN_FEEDBACK.render(outline=state.outline.content),
    )
    match feedback.lower():
        case "y":
            return Command(
                goto=[
                    Send(
                        "main-section-expert",
                        {
                            "section": section.model_dump(),
                            "topic": state.message,
                        },
                    )
                    for section in state.outline.sections
                    if section.research
                ]
            )
        case _:
            return Command(
                goto="main-plan",
                update=TopicPlanningInput(
                    message=state.message,
                    feedback="\n".join(
                        [
                            state.feedback,
                            feedback,
                        ]
                    ),
                ),
            )


async def gather_sections(
    state: State,
) -> ResearchOutput:
    if state.outline is None or state.outline == "":
        raise ValueError("Outline is empty or None.")

    # Update contents from research experts
    for summarization in state.summarizations:
        section_idx = [
            idx
            for idx, section in enumerate(state.outline.sections)
            if section.id == summarization.section_id
        ]

        if len(section_idx) == 0:
            logger.warning(
                "Section with ID %s not found in outline.", summarization.section_id
            )
            continue

        state.outline.sections[section_idx[0]].markdown = summarization.content
        logger.info("Section-%s is updated with the summarization.", section_idx[0])

    return ResearchOutput(
        markdown=state.outline.content,
    )
