## Pre-commit & Continous Integration

There're two tools users need to satisfy before committing to the repository.

| Tool | Purpose | Reason |
|:-----|:--------|:-------|
| mypy | Static type checking |  - Use as an substitute for unit tests during early-stage of the project <br> - Improve code quality and confidence in code correctness from the start.
| ruff | Linking & Coding style checking | - Extremely fast compared to other linters. <br> - Support a wide range of linting rules.

The checking will be applied for each commits in *main* branch or any Pull Requests to *main*. Supported by github actions.
