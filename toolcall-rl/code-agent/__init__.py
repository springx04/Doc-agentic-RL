"""Isolated Code Agent / SWE environment.

The package lives beside, rather than inside, the document-agent runtime.  No
module in this package imports ``bayestool`` or ``tools`` from Doc.
"""

try:
    from .config import CodeConfig
    from .protocol import ParsedAction, parse_action
    from .schemas import CodeToolResult, CodeTrajectory
except ImportError:  # pragma: no cover - pytest imports the hyphenated root as __init__
    from config import CodeConfig
    from protocol import ParsedAction, parse_action
    from schemas import CodeToolResult, CodeTrajectory

__all__ = ["CodeConfig", "CodeToolResult", "CodeTrajectory", "ParsedAction", "parse_action"]
