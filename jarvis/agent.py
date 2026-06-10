"""Core agent: routes user messages through Ollama with tool calling."""

import json
import ollama
from jarvis.config import load_config
from jarvis.tools import TOOL_DEFINITIONS, execute_tool, get_status


def build_system_prompt(config: dict) -> str:
    folders = config.get("registered_folders", [])
    folder_list = "\n".join(f"  - {f}" for f in folders) if folders else "  (none)"
    return f"""You are Jarvis, a local AI assistant running on David's computer.
You have access to the filesystem through tools. You can list directories, read files, search code, and show directory trees.

You are currently live in these folders:
{folder_list}

Guidelines:
- Be concise and direct in responses.
- When asked about your status or what folders you're in, use the get_status tool.
- When asked about code or files, use the appropriate tools to look things up before answering.
- IMPORTANT: When asked about notebook runs, training status, or architecture results, use read_notebook or read_file on the actual .ipynb file — NOT the README. The notebook contains real code cells, outputs, training logs, and metrics. Report the actual numbers you see (loss, accuracy, etc.).
- You can only access files within registered folders.
- If you don't know something, say so rather than guessing.
- Keep responses under 2000 characters (email-friendly).
"""


def run_agent(user_message: str, config: dict = None) -> str:
    """Run the agent loop: send message to model, handle tool calls, return final response."""
    if config is None:
        config = load_config()

    model = config.get("model", "qwen3.5:0.8b")
    max_iterations = config.get("max_tool_iterations", 5)

    messages = [
        {"role": "system", "content": build_system_prompt(config)},
        {"role": "user", "content": user_message},
    ]

    for iteration in range(max_iterations):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as e:
            return f"Error communicating with Ollama: {e}"

        msg = response.get("message", {})

        # If model made tool calls, execute them and continue
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            # Add assistant message with tool calls to history
            messages.append(msg)

            for tool_call in tool_calls:
                fn_name = tool_call["function"]["name"]
                fn_args = tool_call["function"].get("arguments", {})

                result = execute_tool(fn_name, fn_args, config)

                messages.append({
                    "role": "tool",
                    "content": result,
                })

            # Continue loop to let model process tool results
            continue

        # No tool calls - return the final text response
        content = msg.get("content", "").strip()
        if content:
            # Strip thinking tags if model uses them
            if "<think>" in content and "</think>" in content:
                think_end = content.index("</think>") + len("</think>")
                content = content[think_end:].strip()
            return content

        return "(No response generated)"

    return "(Reached max tool iterations)"


def run_agent_simple(user_message: str, config: dict = None) -> str:
    """Fallback: simple completion without tool calling for models that don't support it."""
    if config is None:
        config = load_config()

    model = config.get("model", "qwen3.5:0.8b")
    status = get_status(config)

    prompt = f"""You are Jarvis, a local AI assistant on David's computer.

Current status:
{status}

User message: {user_message}

Respond concisely (under 2000 characters). If the user asks about files or code, mention which folders you're monitoring and what you know."""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are Jarvis, a helpful local AI assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.get("message", {}).get("content", "").strip()
        # Strip thinking tags
        if "<think>" in content and "</think>" in content:
            think_end = content.index("</think>") + len("</think>")
            content = content[think_end:].strip()
        return content or "(No response generated)"
    except Exception as e:
        return f"Error: {e}"
