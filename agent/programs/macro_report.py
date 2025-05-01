from uuid import uuid4
from pydantic import BaseModel, Field

from .base import BaseProgram


class Section(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()), title="Unique identifier for the section"
    )
    title: str = Field(..., title="Title of the section")
    description: str = Field(
        ..., title="Brief overview of the main topics covered in this section"
    )
    research: bool = Field(
        ..., title="Whether to perform web research for this section of the report"
    )
    markdown: str = Field(..., title="Markdown content of the section")

    @property
    def as_str(self) -> str:
        return f"## {self.title}\n\n{self.description}\n\n{self.markdown if self.research else ''}"


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the page")
    sections: list[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


class OutlineReportProgram(BaseProgram[Outline]):
    ModelOutCls = Outline
