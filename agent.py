"""
Autonomous research agent that drives experiments via LM Studio (OpenAI-compatible API).

Usage:
    python3 agent.py

Connects to LM Studio at http://127.0.0.1:1234 and uses program.md to guide
autonomous experimentation on experiment.py.
"""

import os
import sys
import json
import subprocess
import re
import time
import textwrap
from pathlib import Path

try:
    import openai
except ImportError:
    print("Installing openai package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai

# ============================================================================
# CONFIGURATION
# ============================================================================

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL = "gemma-4-26b-a4b-it"
MAX_CONTEXT_TOKENS = 90_000  # leave headroom under 94k
MAX_RESPONSE_TOKENS = 8192
WORKDIR = Path(__file__).parent.resolve()
EXPERIMENT_FILE = WORKDIR / "experiment.py"
PROGRAM_FILE = WORKDIR / "program.md"
REPORT_FILE = WORKDIR / "REPORT.md"
LOG_FILE = WORKDIR / "EXPERIMENT_LOG.md"
RESULTS_TSV = WORKDIR / "results.tsv"
MAX_EXPERIMENT_TIME = 300  # 5 minutes max per experiment run

# ============================================================================
# LLM CLIENT
# ============================================================================

client = openai.OpenAI(
    base_url=LMSTUDIO_BASE_URL,
    api_key="lm-studio",  # LM Studio doesn't need a real key
)

def llm_chat(messages, temperature=0.7):
    """Send messages to LM Studio and get a response."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[AGENT ERROR] LLM call failed: {e}")
        return None

# ============================================================================
# TOOL EXECUTION
# ============================================================================

def read_file(path):
    """Read a file and return its contents (truncated if too large)."""
    p = WORKDIR / path if not Path(path).is_absolute() else Path(path)
    try:
        content = p.read_text()
        # Truncate if too large to fit in context
        if len(content) > 30_000:
            content = content[:30_000] + "\n\n... [TRUNCATED] ..."
        return content
    except FileNotFoundError:
        return f"[FILE NOT FOUND: {p}]"

def write_file(path, content):
    """Write content to a file."""
    p = WORKDIR / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"[WRITTEN: {p}]"

def run_shell(cmd, timeout=MAX_EXPERIMENT_TIME):
    """Run a shell command and return stdout+stderr."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(WORKDIR)
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        # Truncate long output
        if len(output) > 10_000:
            output = output[:5_000] + "\n\n... [TRUNCATED] ...\n\n" + output[-3_000:]
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {timeout}s]"
    except Exception as e:
        return f"[SHELL ERROR: {e}]"

def git_commit(message):
    """Stage and commit all changes."""
    run_shell("git add -A")
    return run_shell(f'git commit -m "{message}"')

# ============================================================================
# AGENT ACTION PARSER
# ============================================================================

def parse_actions(response):
    """Parse the LLM response for actions to execute.

    Expected format in the response:

    <action type="read_file">path/to/file</action>
    <action type="write_file" path="path/to/file">
    file content here
    </action>
    <action type="shell">command here</action>
    <action type="commit">commit message</action>
    <action type="done">summary of findings</action>
    """
    actions = []

    # Parse XML-style action tags
    # read_file
    for m in re.finditer(r'<action\s+type="read_file">(.*?)</action>', response, re.DOTALL):
        actions.append(("read_file", m.group(1).strip()))

    # write_file
    for m in re.finditer(r'<action\s+type="write_file"\s+path="(.*?)">(.*?)</action>', response, re.DOTALL):
        actions.append(("write_file", m.group(1).strip(), m.group(2).strip()))

    # shell
    for m in re.finditer(r'<action\s+type="shell">(.*?)</action>', response, re.DOTALL):
        actions.append(("shell", m.group(1).strip()))

    # commit
    for m in re.finditer(r'<action\s+type="commit">(.*?)</action>', response, re.DOTALL):
        actions.append(("commit", m.group(1).strip()))

    # done
    for m in re.finditer(r'<action\s+type="done">(.*?)</action>', response, re.DOTALL):
        actions.append(("done", m.group(1).strip()))

    return actions

def execute_actions(actions):
    """Execute parsed actions and return results."""
    results = []
    for action in actions:
        action_type = action[0]

        if action_type == "read_file":
            content = read_file(action[1])
            results.append(f"[READ {action[1]}]\n{content}")

        elif action_type == "write_file":
            result = write_file(action[1], action[2])
            results.append(result)

        elif action_type == "shell":
            cmd = action[1]
            # Safety: block dangerous commands
            dangerous = ["rm -rf /", "rm -rf ~", "sudo", "mkfs", "dd if="]
            if any(d in cmd for d in dangerous):
                results.append(f"[BLOCKED dangerous command: {cmd}]")
                continue
            output = run_shell(cmd)
            results.append(f"[SHELL: {cmd}]\n{output}")

        elif action_type == "commit":
            result = git_commit(action[1])
            results.append(f"[COMMIT: {action[1]}]\n{result}")

        elif action_type == "done":
            results.append(f"[DONE: {action[1]}]")

    return "\n\n".join(results)

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = textwrap.dedent("""\
You are an autonomous research agent. You read program.md for instructions and
conduct experiments by modifying experiment.py, running it, and analyzing results.

You communicate actions using XML tags:

<action type="read_file">path/to/file</action>
<action type="write_file" path="path/to/file">
content
</action>
<action type="shell">command</action>
<action type="commit">commit message</action>
<action type="done">summary when you want to report findings</action>

RULES:
- Always think step by step before acting
- Read files before modifying them
- Run experiments with: python3 experiment.py > run.log 2>&1
- Check results with: grep "^success_rate\\|^experiment:\\|^total_seconds:" run.log
- If a run crashes, check: tail -50 run.log
- Keep experiments under 5 minutes wall clock
- Keep array sizes reasonable (≤500) to avoid memory issues
- Log results to results.tsv and update REPORT.md periodically
- Commit after each successful experiment
- NEVER stop experimenting. If you finish one phase, move to the next.
- Include your reasoning before each action

You are running on a local machine (macOS, no GPU). Experiments are CPU-only Python.
Working directory: {workdir}
""").format(workdir=str(WORKDIR))

# ============================================================================
# AGENT LOOP
# ============================================================================

def run_agent():
    """Main agent loop."""
    print("=" * 60)
    print("AUTONOMOUS RESEARCH AGENT")
    print(f"Model: {MODEL} @ {LMSTUDIO_BASE_URL}")
    print(f"Working directory: {WORKDIR}")
    print("=" * 60)

    # Verify LM Studio is reachable
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"Available models: {available}")
    except Exception as e:
        print(f"ERROR: Cannot reach LM Studio at {LMSTUDIO_BASE_URL}")
        print(f"  {e}")
        print("Start LM Studio and load the model first.")
        sys.exit(1)

    # Initialize conversation with context
    program = read_file("program.md")
    experiment_code = read_file("experiment.py")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": textwrap.dedent(f"""\
            Here are your instructions and the current experiment code.
            Read them, then begin the experiment loop as described in program.md.

            === program.md ===
            {program}

            === experiment.py (current) ===
            {experiment_code}

            Begin! Start with Phase 1: replication of Levin et al.'s sorting experiments.
            Read any additional files you need, then start experimenting.
        """)},
    ]

    iteration = 0
    max_iterations = 200  # safety limit

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'=' * 40}")
        print(f"ITERATION {iteration}")
        print(f"{'=' * 40}")

        # Get LLM response
        response = llm_chat(messages)
        if response is None:
            print("[AGENT] LLM returned None, retrying in 5s...")
            time.sleep(5)
            continue

        print(f"\n[AGENT THINKING]\n{response[:500]}...")

        # Parse and execute actions
        actions = parse_actions(response)

        if not actions:
            # No parseable actions - ask the LLM to use the action format
            print("[AGENT] No actions parsed, prompting for actions...")
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                "Please use the action tags to perform your next step. "
                "For example: <action type=\"shell\">python3 experiment.py > run.log 2>&1</action>"
            })
            continue

        print(f"\n[EXECUTING {len(actions)} actions]")
        results = execute_actions(actions)
        print(f"\n[RESULTS]\n{results[:1000]}...")

        # Add to conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Action results:\n\n{results}\n\nContinue with the next experiment step."})

        # Context window management: trim old messages if getting large
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > MAX_CONTEXT_TOKENS * 3:  # rough chars-to-tokens ratio
            print("[AGENT] Trimming conversation history...")
            # Keep system prompt + last 6 messages
            messages = [messages[0]] + messages[-6:]
            # Re-inject current state
            current_code = read_file("experiment.py")
            results_tsv = read_file("results.tsv") if RESULTS_TSV.exists() else "[No results yet]"
            messages.insert(1, {"role": "user", "content": textwrap.dedent(f"""\
                [Context refresh - conversation was trimmed]

                Current experiment.py:
                {current_code}

                Current results.tsv:
                {results_tsv}

                Continue experimenting per program.md instructions.
            """)})

        # Check for done signal (just log it, don't stop)
        for action in actions:
            if action[0] == "done":
                print(f"\n[AGENT REPORTED DONE]: {action[1]}")
                print("Continuing anyway (autonomous mode)...")

        time.sleep(1)  # small delay between iterations

    print(f"\n[AGENT] Reached max iterations ({max_iterations}). Stopping.")

if __name__ == "__main__":
    run_agent()
