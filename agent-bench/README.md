# agent-bench — Agent Workspace

This directory is the **sandbox** for benchmarked agents. It contains task specifications, source files the agent must edit, and public sanity tests.

## Structure

```
agent-bench/
├── tasks/              # Task families
│   └── <task-name>/    # One directory per task
│       ├── README.md   # Task spec (given to the agent)
│       ├── <pkg>/      # Source files the agent edits
│       └── tests/      # Public sanity tests the agent can run
└── prompts/            # Prompt templates / regimes
```

## Rules for the benchmarked agent

1. **Only edit source files** inside the task's package directory (e.g., `tasks/snake/snake/`).
2. **Do not modify tests** — public sanity tests are read-only.
3. **Do not access `eval/`** — hidden tests and scoring scripts live there. Any run that leaks hidden-test content to the agent is invalid.
4. **Do not add dependencies** — solutions must work with the existing environment.
5. **Do not change the file structure** — no new files, no moving files.

## Running public tests

From within a task directory:

```bash
cd agent-bench/tasks/snake
pytest tests/
```
