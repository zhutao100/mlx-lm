# Tokenizers Module Analysis (`mlx_lm.tokenizers`)

## Purpose
Provides specialized tokenizer handling and chat templating logic, particularly for models with complex formatting requirements that exceed standard Hugging Face tokenizer capabilities.

## Key Components

### 1. DeepSeek V3/V2 (`deepseek_v32.py`)
-   **Purpose**: Manual implementation of the chat template and tool use formatting for DeepSeek V3/V2 models.
-   **Features**:
    -   **DSML (DeepSeek Markup Language)**: Handles `<｜DSML｜>` tokens and XML-like tags for function calls (`<invoke>`, `<parameter>`).
    -   **Thinking Mode**: Supports `<think>` blocks for reasoning models.
    -   **Manual Template**: `apply_chat_template` manually constructs the prompt instead of using Jinja, ensuring precise control over control tokens (`<｜begin▁of▁sentence｜>`, `<｜User｜>`, etc.).
    -   **Roles**: Explicit handling of `system`, `user`, `assistant`, `tool`, and `developer` roles.
    -   **Output Parsing**: Includes helpers to encode arguments to DSML and decode DSML to arguments.
