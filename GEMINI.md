# EasyLocomo Project Specifications

This document outlines the development norms, configuration standards, and testing protocols for the EasyLocomo project. All contributors (human and AI) should follow these guidelines.

## 1. Configuration Management
- **Centralized Config**: All configuration parameters must be managed via the `Config` class in `locomo/config.py`.
- **Environment Variables**: Sensitive data (API keys) and environment-specific settings should be loaded from `.env` using `python-dotenv`.
- **No Hardcoding**: Avoid hardcoding paths or model names. Use `config` attributes.

## 2. API Interaction
- **Client Wrapper**: Use `locomo.utils.openai_client.run_chatgpt_async` for concurrency and `run_chatgpt` for legacy/sync needs.
- **Model Support**: Default model is `gemini-2.5-flash`.
- **Error Handling**: Use `tenacity` decorators for retries. Handle potential `None` or empty content returns.

## 3. Testing & Verification
- **Automated Runner**: Use `scripts/test_runner.py` for high-level integration tests.
- **Unit Tests**: Use `pytest tests/` for logic verification (context order, truncation, async behavior).
- **Small Test**: ALWAYS run `python scripts/test_runner.py small` before submitting changes.

## 4. Coding Standards
- **Async First**: Prioritize asynchronous operations for performance.
- **Path Handling**: Convert `pathlib.Path` to `str` before string operations.
- **Type Hinting**: Use type hints (e.g., `Optional[int]`, `Path`).

## 5. Directory Structure
- `locomo/`: Core package source code.
- `scripts/`: Utility scripts (test runner, estimators).
- `tests/`: Unit tests and logic verification.
- `data/`: Dataset files.
- `outputs/`: Results and logs.

## 6. Maintenance Workflow (Sync Procedure)
Before submitting or after significant changes, follow these steps:
1.  **Sync Dependencies**: Update `pyproject.toml` and then run `uv pip compile pyproject.toml -o requirements.txt` (if using requirements) or simply `uv sync`.
2.  **Clean Artifacts**: Remove temporary files (`debug_output.txt`, `compare_with_old/`, and temporary scripts like `extract_data_temp.py`).
3.  **Check GitIgnore**: Ensure logs and temporary files are ignored.
4.  **Verify All Tests**: Run both `pytest` and the `small` integration test.
## 7. Communication Protocol
- **Status Inquiry**: If the user asks 'What are you doing?' (你在干什么?), the AI must STOP all work immediately and only report the current status and intent. Do not proceed with the next step until further instruction.

## 8. Commit Protocol
- **Prepare Commit**: When the user asks to "prepare commit" or "ready to commit", **automatically organize the commit message** summarizing the changes (feat, fix, refactor, docs) and present it for review.
- **Pre-Commit Check**: Before committing, ALWAYS check and update configuration files (`pyproject.toml`, etc.) and documentation (`.md` files) to reflect the latest changes.

