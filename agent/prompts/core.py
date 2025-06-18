from pathlib import Path
from typing import Final, Literal

from jinja2 import (
    Environment,
    FileSystemLoader,
    Template,
    TemplateNotFound,
    select_autoescape,
)
from pydantic import BaseModel


class Jinja2PromptSettings(BaseModel):
    trim_blocks: bool = True
    lstrip_blocks: bool = True


class Jinja2Prompts[TemplateT: str]:
    TEMPLATE_NAME: Final[str] = "{}.md"

    def __init__(
        self,
        promptdir: Path,
        settings: Jinja2PromptSettings | None = None,
    ) -> None:
        self.promptdir = promptdir
        self.settings = settings or Jinja2PromptSettings()
        self.env = Environment(
            loader=FileSystemLoader(self.promptdir),
            autoescape=select_autoescape(),
            trim_blocks=self.settings.trim_blocks,
            lstrip_blocks=self.settings.lstrip_blocks,
        )

    def get(self, template_name: TemplateT) -> Template:
        try:
            return self.env.get_template(
                self.TEMPLATE_NAME.format(template_name),
            )
        except TemplateNotFound as err:
            raise ValueError(
                f"Template {template_name} not found in {self.promptdir}. "
            ) from err


class BookingJinja2Prompts(
    Jinja2Prompts[Literal["coordinator", "faq", "operation", "operation_react"]]
): ...


class PromptsFactory:
    @staticmethod
    def booking() -> BookingJinja2Prompts:
        return BookingJinja2Prompts(
            promptdir=Path(__file__).parent / "booking",
        )
