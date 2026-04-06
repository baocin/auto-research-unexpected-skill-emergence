"""
Autonomous research agent that drives experiments via LM Studio (OpenAI-compatible API).

Usage:
    PYTHONUNBUFFERED=1 python3 -u agent.py

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
import difflib
from pathlib import Path

# Force unbuffered output so logs appear in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
MAX_RESPONSE_TOKENS = 16384
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
    api_key="lm-studio",
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
        if len(content) > 20_000:
            content = content[:20_000] + "\n\n... [TRUNCATED] ..."
        return content
    except FileNotFoundError:
        return f"[FILE NOT FOUND: {p}]"

def write_file(path, content):
    """Write content to a file. Rejects suspiciously short writes to existing large files."""
    p = WORKDIR / path if not Path(path).is_absolute() else Path(path)
    # Safety: don't allow overwriting a large file with a tiny one
    if p.exists():
        old_len = len(p.read_text())
        new_len = len(content)
        if old_len > 500 and new_len < old_len * 0.3:
            return (f"[REJECTED write to {p}: new content ({new_len} chars) is much smaller "
                    f"than existing ({old_len} chars). Use edit_file to make targeted changes, "
                    f"or write_new_file for new files.]")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"[WRITTEN: {p} ({len(content)} chars)]"

def edit_file(path, old_text, new_text):
    """Search-and-replace edit: find old_text in file and replace with new_text."""
    p = WORKDIR / path if not Path(path).is_absolute() else Path(path)
    try:
        content = p.read_text()
    except FileNotFoundError:
        return f"[FILE NOT FOUND: {p}]"

    if old_text not in content:
        # Try to find a close match
        lines = content.split('\n')
        old_lines = old_text.split('\n')
        matches = difflib.get_close_matches(old_lines[0], lines, n=1, cutoff=0.6)
        hint = f" Closest match to first line: '{matches[0]}'" if matches else ""
        return f"[EDIT FAILED: old_text not found in {p}.{hint}]"

    count = content.count(old_text)
    if count > 1:
        return f"[EDIT FAILED: old_text appears {count} times in {p}. Provide more context to make it unique.]"

    new_content = content.replace(old_text, new_text, 1)
    p.write_text(new_content)
    return f"[EDITED: {p} (replaced {len(old_text)} chars with {len(new_text)} chars)]"

def append_file(path, content):
    """Append content to end of a file."""
    p = WORKDIR / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'a') as f:
        f.write(content)
    return f"[APPENDED to {p}]"

def run_shell(cmd, timeout=MAX_EXPERIMENT_TIME):
    """Run a shell command and return stdout+stderr. Kills child processes on timeout."""
    import signal
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(WORKDIR), preexec_fn=os.setsid
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(2)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            return f"[TIMEOUT after {timeout}s - process killed]"

        output = ""
        if stdout:
            output += stdout
        if stderr:
            output += "\n[STDERR]\n" + stderr
        if len(output) > 8_000:
            output = output[:4_000] + "\n\n... [TRUNCATED] ...\n\n" + output[-2_000:]
        return output.strip()
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

    Supported actions:
    <action type="read_file">path</action>
    <action type="write_file" path="path">content</action>
    <action type="edit_file" path="path">
    <<<OLD
    old text
    ===
    new text
    >>>NEW
    </action>
    <action type="append_file" path="path">content to append</action>
    <action type="shell">command</action>
    <action type="commit">message</action>
    """
    actions = []

    # read_file - support both <action type="read_file">path</action> and self-closing
    for m in re.finditer(r'<action\s+type="read_file"(?:\s+path="(.*?)")?\s*/>', response, re.DOTALL):
        if m.group(1):
            actions.append(("read_file", m.group(1).strip()))
    for m in re.finditer(r'<action\s+type="read_file"(?:\s+path="(.*?)")?>(.*?)</action>', response, re.DOTALL):
        path = m.group(1) or m.group(2).strip()
        if path:
            actions.append(("read_file", path))

    # edit_file (search/replace - PREFERRED for modifying existing files)
    for m in re.finditer(r'<action\s+type="edit_file"\s+path="(.*?)">(.*?)</action>', response, re.DOTALL):
        body = m.group(2).strip()
        # Parse <<<OLD ... === ... >>>NEW format
        edit_match = re.search(r'<<<\s*OLD\s*\n(.*?)\n\s*===\s*\n(.*?)\n\s*>>>\s*NEW', body, re.DOTALL)
        if edit_match:
            actions.append(("edit_file", m.group(1).strip(), edit_match.group(1), edit_match.group(2)))
        else:
            # Fallback: try splitting on === separator
            parts = body.split('\n===\n', 1)
            if len(parts) == 2:
                actions.append(("edit_file", m.group(1).strip(), parts[0], parts[1]))

    # write_file (for NEW files only)
    for m in re.finditer(r'<action\s+type="write_file"\s+path="(.*?)">(.*?)</action>', response, re.DOTALL):
        actions.append(("write_file", m.group(1).strip(), m.group(2).strip()))

    # append_file
    for m in re.finditer(r'<action\s+type="append_file"\s+path="(.*?)">(.*?)</action>', response, re.DOTALL):
        actions.append(("append_file", m.group(1).strip(), m.group(2).strip()))

    # shell - also handle self-closing with cmd attribute
    for m in re.finditer(r'<action\s+type="shell"\s+cmd="(.*?)"\s*/>', response, re.DOTALL):
        actions.append(("shell", m.group(1).strip()))
    for m in re.finditer(r'<action\s+type="shell">(.*?)</action>', response, re.DOTALL):
        actions.append(("shell", m.group(1).strip()))

    # commit
    for m in re.finditer(r'<action\s+type="commit">(.*?)</action>', response, re.DOTALL):
        actions.append(("commit", m.group(1).strip()))
    for m in re.finditer(r'<action\s+type="commit"\s+msg="(.*?)"\s*/>', response, re.DOTALL):
        actions.append(("commit", m.group(1).strip()))

    # Deduplicate (same action can match multiple patterns)
    seen = set()
    unique = []
    for a in actions:
        key = str(a)
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return unique

def execute_actions(actions):
    """Execute parsed actions and return results."""
    results = []
    for action in actions:
        action_type = action[0]

        if action_type == "read_file":
            content = read_file(action[1])
            results.append(f"[READ {action[1]}]\n{content}")

        elif action_type == "edit_file":
            result = edit_file(action[1], action[2], action[3])
            results.append(result)

        elif action_type == "write_file":
            result = write_file(action[1], action[2])
            results.append(result)

        elif action_type == "append_file":
            result = append_file(action[1], action[2])
            results.append(result)

        elif action_type == "shell":
            cmd = action[1]
            dangerous = ["rm -rf /", "rm -rf ~", "sudo", "mkfs", "dd if="]
            if any(d in cmd for d in dangerous):
                results.append(f"[BLOCKED dangerous command: {cmd}]")
                continue
            output = run_shell(cmd)
            results.append(f"[SHELL: {cmd}]\n{output}")

        elif action_type == "commit":
            result = git_commit(action[1])
            results.append(f"[COMMIT: {action[1]}]\n{result}")

    return "\n\n".join(results)

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = textwrap.dedent("""\
You are an autonomous research agent investigating emergent competencies in
locally-executed algorithms, inspired by Levin et al. (2025).

You modify experiment.py, run experiments, and analyze results.

## Actions (use XML tags - MUST have opening AND closing tags, NO self-closing)

To run a command:
<action type="shell">python3 experiment.py > run.log 2>&1</action>

To read a file:
<action type="read_file">experiment.py</action>

To edit a file (search and replace):
<action type="edit_file" path="experiment.py">
<<<OLD
exact old text
===
new replacement text
>>>NEW
</action>

To create a NEW file:
<action type="write_file" path="experiments/new_test.py">file content here</action>

To append to a file:
<action type="append_file" path="results.tsv">row data here</action>

To git commit:
<action type="commit">descriptive message</action>

IMPORTANT: Every tag MUST have a closing </action> tag. Do NOT use self-closing tags like <action ... />.

## CRITICAL RULES
- Use edit_file (search/replace) to modify existing files. NEVER use write_file on existing files.
- The OLD text must match EXACTLY (including whitespace/indentation).
- Keep edits small and focused. One change per edit_file action.
- Run experiments: python3 experiment.py > run.log 2>&1
- Check results: grep "^success_rate\\|^experiment:\\|^total_seconds:" run.log
- If crash: tail -50 run.log
- Max array size: 500. Max experiment time: 5 minutes.
- Commit after each successful experiment.
- NEVER stop. Move to next phase when done with current.
- One action per step is fine. Quality over quantity.

## Research phases
1. REPLICATE Levin: cell-view bubble/insertion/selection sort with frozen cells
2. EXTEND to more sorting algorithms (merge, quick, shell, cocktail, gnome, comb)
3. BEYOND sorting: search, graph, consensus, anomaly detection, clustering
4. ANALYZE: what properties predict emergent robustness?

Working directory: {workdir}
""").format(workdir=str(WORKDIR))

# ============================================================================
# AGENT LOOP
# ============================================================================

def get_config_section():
    """Extract just the config section from experiment.py for context."""
    try:
        content = EXPERIMENT_FILE.read_text()
        lines = content.split('\n')
        config_lines = []
        in_config = False
        for line in lines:
            if 'EXPERIMENT CONFIGURATION' in line:
                in_config = True
            if in_config:
                config_lines.append(line)
            if in_config and line.strip() == '' and len(config_lines) > 3:
                break
        return '\n'.join(config_lines) if config_lines else content[:500]
    except:
        return "[Could not read experiment.py]"

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

    # Load compact context (NOT the full files - just summaries)
    config_section = get_config_section()

    # Get list of available algorithms
    try:
        content = EXPERIMENT_FILE.read_text()
        algo_funcs = re.findall(r'^def (cell_view_\w+|traditional_\w+)\(', content, re.MULTILINE)
        algo_list = ', '.join(algo_funcs)
    except:
        algo_list = "unknown"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": textwrap.dedent(f"""\
            Current experiment.py config section:
            {config_section}

            Available algorithm functions: {algo_list}

            COMPLETED PHASES:
            - Phase 1: Replicated Levin et al. Bubble sort robust (97% at 10% dmg),
              insertion sort collapses (0% at 5% dmg), selection sort most robust (100% at 10%).
            - Phase 2: Added gnome sort (surprisingly fragile at 15% at 10% dmg).
              Interaction range is the key predictor of robustness.
            - Phase 3: Target search - random walk (20%) beats deterministic gradient (5%)
              in damaged environments. Stochasticity = robustness.

            COMPLETED (do NOT repeat these):
            - Consensus: done extensively (anchors, adversarial, oscillating, error scaling)
            - K-means: done extensively (noise, topology, radius, density, dropout, drift)
            - Anomaly detection: done (sensor damage, SNR breakdown, streaming)
            - Grid sorting: done (obstacles, dead zones)

            NEXT: Explore COMPLETELY NEW domains. Pick ONE per experiment:
            1. Distributed Bellman-Ford shortest path with damaged edges
            2. Load balancing: workers redistribute tasks locally, some workers frozen
            3. Epidemic/rumor spreading on damaged networks
            4. Local leader election with faulty nodes
            5. Distributed coloring: nodes pick colors to differ from neighbors
            6. Token ring / mutual exclusion with damaged nodes
            7. Return to SORTING: test cocktail shaker, shell sort, comb sort cell-view
            8. 2D sorting on grids with the original Levin cell-view model

            IMPORTANT: Do NOT create more k-means or consensus variants.
            Each experiment must be a genuinely NEW algorithm or domain.
            Create new experiment files in experiments/ directory.
            Use edit_file for modifying existing files. Always use action tags.
        """)},
    ]

    iteration = 0
    max_iterations = 500
    consecutive_failures = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'=' * 40}")
        print(f"ITERATION {iteration}")
        print(f"{'=' * 40}")

        response = llm_chat(messages)
        if response is None:
            consecutive_failures += 1
            if consecutive_failures > 5:
                print("[AGENT] Too many consecutive LLM failures. Waiting 60s...")
                time.sleep(60)
                consecutive_failures = 0
            else:
                print("[AGENT] LLM returned None, retrying in 10s...")
                time.sleep(10)
            continue

        consecutive_failures = 0
        print(f"\n[AGENT RESPONSE]\n{response[:800]}...")

        actions = parse_actions(response)

        if not actions:
            no_action_count = getattr(run_agent, '_no_action_count', 0) + 1
            run_agent._no_action_count = no_action_count
            print(f"[AGENT] No actions parsed ({no_action_count} consecutive)")

            if no_action_count >= 3:
                # Agent is stuck thinking it's done. Reset with a new task.
                print("[AGENT] Stuck in no-action loop. Injecting new research direction...")
                run_agent._no_action_count = 0
                new_ideas = [
                    "Test ADVERSARIAL consensus: half frozen at 0, half at 1. Does it still converge? Create experiments/adversarial_consensus.py",
                    "Implement distributed anomaly detection: each node decides if its value is an outlier based on neighbors. Create experiments/anomaly_detection.py",
                    "Test 2D grid topology instead of 1D: does sorting/consensus behave differently? Create experiments/grid_sorting.py",
                    "Implement local k-means: each point moves toward the centroid of its neighbors. Create experiments/local_kmeans.py",
                    "Test noise-as-damage model: instead of frozen cells, cells randomly flip their output. Create experiments/noisy_execution.py",
                    "Scale up: run sorting experiments at N=100 and N=200. Modify experiment.py config.",
                ]
                import random
                idea = random.choice(new_ideas)
                messages = [messages[0]]  # Reset to just system prompt
                messages.append({"role": "user", "content":
                    f"NEW TASK: {idea}\n\n"
                    f"Use action tags to create the file and run the experiment. "
                    f"Remember: <action type=\"write_file\" path=\"path\">content</action> "
                    f"and <action type=\"shell\">command</action>"
                })
                continue

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content":
                "You MUST use action tags to continue. Do NOT summarize - start a new experiment.\n"
                "Pick one of these and CREATE the experiment file:\n"
                "1. Adversarial consensus (half frozen at 0, half at 1)\n"
                "2. Anomaly detection with damaged sensors\n"
                "3. 2D grid sorting\n"
                "4. Local k-means clustering\n\n"
                "Example: <action type=\"write_file\" path=\"experiments/new_experiment.py\">code here</action>"
            })
            continue

        run_agent._no_action_count = 0  # Reset stuck counter
        print(f"\n[EXECUTING {len(actions)} action(s)]")
        results = execute_actions(actions)
        print(f"\n[RESULTS]\n{results[:2000]}")

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content":
            f"Action results:\n\n{results}\n\nContinue. What's next?"})

        # Context management
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > MAX_CONTEXT_TOKENS * 3:
            print("[AGENT] Trimming conversation...")
            # Keep system + last 8 messages
            messages = [messages[0]] + messages[-8:]
            # Inject fresh state
            config = get_config_section()
            tsv = read_file("results.tsv") if RESULTS_TSV.exists() else "[No results yet]"
            messages.insert(1, {"role": "user", "content":
                f"[Context refresh]\n\nCurrent config:\n{config}\n\nResults so far:\n{tsv}\n\nContinue experimenting."
            })

        time.sleep(1)

    print(f"\n[AGENT] Reached max iterations ({max_iterations}).")

if __name__ == "__main__":
    run_agent()
