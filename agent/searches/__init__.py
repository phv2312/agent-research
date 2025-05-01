from .interface import ISearch
from .impl.tavily import TavilyWebSearch
from .impl.duckduckgo import DuckduckgoWebSearch


__all__ = [
    "ISearch",
    "TavilyWebSearch",
    "DuckduckgoWebSearch",
]
