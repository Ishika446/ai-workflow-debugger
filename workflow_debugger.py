"""
workflow_debugger.py
====================
A unified AI debugging tool that supports:
  1. CLI mode     – Run a Python script and auto-debug any errors
  2. JSON mode    – Parse n8n / Make / Zapier workflow JSON files and
                    use AI to analyse potential errors / misconfigurations
  3. Server mode  – FastAPI webhook server that receives live error payloads
                    from automation tools and analyses them in the background

All analysis is powered by Google Gemini via LangChain.
Optionally creates Jira tickets for every issue found.

Usage
-----
  python workflow_debugger.py <script.py>            # Debug a Python script
  python workflow_debugger.py --json <workflow.json> # Analyse a workflow JSON
  python workflow_debugger.py --server               # Start webhook server
"""

import sys
import subprocess
import os
import json
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from jira import JIRA

# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI(title="AI Workflow Debugger")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# Pydantic Models
# ===========================================================================

class WebhookPayload(BaseModel):
    workflow_name: str
    node_name: str
    error_message: str
    code_context: str = "No code context provided by automation tool."


class JsonContentPayload(BaseModel):
    json_content: dict  # The raw parsed workflow JSON object


# ===========================================================================
# Shared LLM Utility
# ===========================================================================

def _get_llm():
    """Return a configured Gemini LLM instance or raise an error."""
    if not os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError("⚠️  GOOGLE_API_KEY environment variable is not set.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# ===========================================================================
# 1. PYTHON SCRIPT ERROR ANALYSIS  (original feature)
# ===========================================================================

def analyze_error_with_langchain(script_path_or_context: str,
                                  error_traceback: str,
                                  is_file: bool = True) -> str:
    """
    Analyse a Python error using Gemini.

    Parameters
    ----------
    script_path_or_context : str
        Either a file path (when is_file=True) or raw code/context string.
    error_traceback : str
        The error message / traceback to analyse.
    is_file : bool
        If True, read code from the path; otherwise treat first arg as raw text.
    """
    if is_file:
        try:
            with open(script_path_or_context, "r") as f:
                script_content = f.read()
        except Exception as e:
            script_content = f"Could not read script file: {e}"
    else:
        script_content = script_path_or_context

    try:
        llm = _get_llm()
    except EnvironmentError as e:
        return str(e)
    except Exception as e:
        return f"Failed to initialize LLM: {e}"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert Python supervisor agent and debugging assistant. "
            "Analyse the following error. Explain exactly why the error occurred, "
            "and provide a clear, step-by-step explanation of how to fix it. Be concise."
        ),
        (
            "user",
            "Context (Code or workflow steps):\n\n```python\n{code}\n```\n\n"
            "Error Traceback/Message:\n\n```\n{traceback}\n```\n\n"
            "Please explain what went wrong and how I can fix it."
        )
    ])

    chain = prompt | llm
    analysis_result = ""
    try:
        for chunk in chain.stream({"code": script_content, "traceback": error_traceback}):
            if is_file:
                print(chunk.content, end="", flush=True)
            analysis_result += chunk.content
        if is_file:
            print("\n" + "=" * 50)
    except Exception as e:
        analysis_result = f"\nFailed to communicate with the LLM API: {e}"
        print(analysis_result)

    return analysis_result


def run_script_and_catch_errors(script_path: str):
    """Run a Python script; if it fails, analyse the error with AI."""
    print(f"🚀 Running CLI Mode: {script_path}...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)

    if result.returncode == 0:
        print("\n✅ Script ran successfully with no errors!")
        print("Output:\n", result.stdout)
        return

    print("\n❌ Error detected! Analysing the issue with LangChain Agent...\n")
    print("🧠 AI Debugger Analysis:\n" + "=" * 50)
    analysis = analyze_error_with_langchain(script_path, result.stderr, is_file=True)

    summary = f"[CLI Bug] Error in {os.path.basename(script_path)}"
    description = (
        f"*File:* {script_path}\n\n"
        f"*Traceback:*\n{{code}}\n{result.stderr}\n{{code}}\n\n"
        f"*AI Fix:*\n{analysis}"
    )
    create_jira_ticket(summary, description)


# ===========================================================================
# 2. WORKFLOW JSON FILE ANALYSIS  (NEW feature)
# ===========================================================================

# ── Tool detection helpers ──────────────────────────────────────────────────

def _detect_tool(data: dict) -> str:
    """
    Heuristically detect which automation tool produced the JSON.
    Returns one of: 'n8n', 'make', 'zapier', 'generic'
    """
    # n8n exports always have a top-level 'nodes' list and 'connections' dict
    if "nodes" in data and "connections" in data:
        return "n8n"
    # Make (formerly Integromat) uses 'flow' or 'blueprint'
    if "flow" in data or "blueprint" in data:
        return "make"
    # Zapier exports have 'steps' or 'zaps'
    if "steps" in data or "zaps" in data:
        return "zapier"
    return "generic"


def _extract_n8n_summary(data: dict) -> str:
    """Extract a human-readable summary from an n8n workflow JSON."""
    lines = []
    name = data.get("name", "Unnamed Workflow")
    lines.append(f"Workflow Name: {name}")
    lines.append(f"Active: {data.get('active', 'unknown')}")

    nodes = data.get("nodes", [])
    lines.append(f"\nTotal Nodes: {len(nodes)}")
    for node in nodes:
        node_type  = node.get("type", "Unknown")
        node_name  = node.get("name", "Unnamed")
        disabled   = node.get("disabled", False)
        parameters = node.get("parameters", {})
        creds      = node.get("credentials", {})

        status = "⚠️ DISABLED" if disabled else "✅ active"
        lines.append(f"\n  [{status}] Node: '{node_name}' (type: {node_type})")

        if parameters:
            lines.append(f"    Parameters: {json.dumps(parameters, indent=6)}")
        if creds:
            lines.append(f"    Credentials used: {list(creds.keys())}")

    connections = data.get("connections", {})
    lines.append(f"\nConnections Map:")
    for src, targets in connections.items():
        for _output_idx, target_list in targets.items():
            for target_group in target_list:
                for conn in target_group:
                    lines.append(f"  {src} → {conn.get('node')} (input {conn.get('index', 0)})")

    settings = data.get("settings", {})
    if settings:
        lines.append(f"\nWorkflow Settings: {json.dumps(settings, indent=4)}")

    return "\n".join(lines)


def _extract_make_summary(data: dict) -> str:
    """Extract a human-readable summary from a Make (Integromat) workflow JSON."""
    lines = []
    bp = data.get("blueprint") or data
    name = bp.get("name", "Unnamed Scenario")
    lines.append(f"Scenario Name: {name}")

    flow = bp.get("flow", [])
    lines.append(f"Total Modules: {len(flow)}")
    for module in flow:
        m_id    = module.get("id", "?")
        m_name  = module.get("module", "Unknown")
        m_label = module.get("metadata", {}).get("designer", {}).get("x", "")
        lines.append(f"\n  Module [{m_id}]: {m_name}")
        mapper = module.get("mapper", {})
        if mapper:
            lines.append(f"    Mapper: {json.dumps(mapper, indent=6)}")

    return "\n".join(lines)


def _extract_zapier_summary(data: dict) -> str:
    """Extract a human-readable summary from a Zapier workflow JSON."""
    lines = []
    zap_name = data.get("title") or data.get("name", "Unnamed Zap")
    lines.append(f"Zap Name: {zap_name}")
    steps = data.get("steps") or data.get("zaps", [])
    lines.append(f"Total Steps: {len(steps)}")
    for i, step in enumerate(steps):
        app   = step.get("app", {}).get("title", "Unknown App")
        event = step.get("action", {}).get("title", step.get("event", "Unknown event"))
        active = step.get("paused", False)
        lines.append(f"\n  Step {i+1}: [{app}] → {event}" + (" (PAUSED)" if active else ""))
        params = step.get("params", {})
        if params:
            lines.append(f"    Params: {json.dumps(params, indent=6)}")
    return "\n".join(lines)


def _extract_generic_summary(data: dict) -> str:
    """Fallback: just pretty-print the JSON."""
    return json.dumps(data, indent=2)


EXTRACTORS = {
    "n8n":     _extract_n8n_summary,
    "make":    _extract_make_summary,
    "zapier":  _extract_zapier_summary,
    "generic": _extract_generic_summary,
}


# ── Main JSON analysis function ─────────────────────────────────────────────

def analyze_workflow_json(json_path: str) -> None:
    """
    Parse a workflow JSON file (n8n / Make / Zapier / generic), extract its
    structure, pass it to Gemini for analysis, and optionally create a Jira ticket.
    """
    path = Path(json_path)
    if not path.exists():
        print(f"❌ File '{json_path}' does not exist.")
        sys.exit(1)

    print(f"📂 Loading workflow file: {path.name}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        sys.exit(1)

    tool = _detect_tool(data)
    print(f"🔍 Detected automation tool: {tool.upper()}")

    # Build a human-readable summary of the workflow
    extractor = EXTRACTORS.get(tool, EXTRACTORS["generic"])
    workflow_summary = extractor(data)

    print("\n📋 Workflow Structure:\n" + "=" * 50)
    print(workflow_summary)
    print("=" * 50)

    # Ask Gemini to analyse the workflow for potential issues
    print("\n🧠 AI Workflow Analysis:\n" + "=" * 50)
    analysis = _ai_analyze_workflow(tool, workflow_summary, stream=True)
    print("=" * 50)

    # Optionally create a Jira ticket
    summary     = f"[Workflow Review] {tool.upper()} – {path.name}"
    description = (
        f"*Tool:* {tool.upper()}\n"
        f"*File:* {json_path}\n\n"
        f"*Workflow Structure:*\n{{code}}\n{workflow_summary}\n{{code}}\n\n"
        f"*AI Analysis & Recommendations:*\n{analysis}"
    )
    create_jira_ticket(summary, description)


def _ai_analyze_workflow(tool: str, workflow_summary: str, stream: bool = True) -> str:
    """
    Use Gemini to analyse a workflow summary for errors and misconfigurations.
    """
    try:
        llm = _get_llm()
    except EnvironmentError as e:
        print(str(e))
        return str(e)
    except Exception as e:
        msg = f"Failed to initialize LLM: {e}"
        print(msg)
        return msg

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are an expert automation workflow architect specialising in {tool} workflows. "
            "You review workflow definitions to identify:\n"
            "  - Missing or incorrect credentials\n"
            "  - Disabled nodes that might break the workflow\n"
            "  - Misconfigured parameters or mappings\n"
            "  - Infinite loops or missing error-handling paths\n"
            "  - Security concerns (e.g. hard-coded secrets in parameters)\n"
            "  - Logic errors (wrong node order, missing triggers, etc.)\n\n"
            "For each issue found, describe it clearly and suggest how to fix it. "
            "If the workflow looks healthy, say so and mention any best-practice suggestions."
        ),
        (
            "user",
            "Here is the extracted {tool} workflow definition:\n\n"
            "```\n{workflow}\n```\n\n"
            "Please analyse this workflow and report any errors, "
            "misconfigurations, or improvements."
        )
    ])

    chain = prompt | llm
    analysis = ""
    try:
        for chunk in chain.stream({"tool": tool, "workflow": workflow_summary}):
            if stream:
                print(chunk.content, end="", flush=True)
            analysis += chunk.content
        if stream:
            print()
    except Exception as e:
        msg = f"\nFailed to communicate with the LLM API: {e}"
        print(msg)
        analysis = msg

    return analysis


def analyze_workflow_json_return(json_path: str) -> Tuple[str, str]:
    """
    Same as analyze_workflow_json but returns (workflow_summary, analysis)
    strings instead of printing. Used by the webhook background processor.
    """
    path = Path(json_path)
    if not path.exists():
        return "", f"File '{json_path}' does not exist."

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return "", f"Invalid JSON: {e}"

    tool = _detect_tool(data)
    extractor = EXTRACTORS.get(tool, EXTRACTORS["generic"])
    workflow_summary = extractor(data)
    analysis = _ai_analyze_workflow(tool, workflow_summary, stream=False)
    return workflow_summary, analysis


# ===========================================================================
# 3. JIRA INTEGRATION  (original feature)
# ===========================================================================

def create_jira_ticket(summary: str, description: str):
    """Create a Jira issue using env-configured credentials."""
    jira_server    = os.environ.get("JIRA_SERVER")
    jira_email     = os.environ.get("JIRA_EMAIL")
    jira_api_token = os.environ.get("JIRA_API_TOKEN")
    jira_project   = os.environ.get("JIRA_PROJECT_KEY", "PROJ")

    if not all([jira_server, jira_email, jira_api_token]):
        print("⚠️  Jira credentials missing. Skipping ticket creation.")
        return None

    try:
        jira = JIRA(server=jira_server, basic_auth=(jira_email, jira_api_token))
        issue_dict = {
            "project":   {"key": jira_project},
            "summary":   summary[:250],
            "description": description,
            "issuetype": {"id": "10007"},
        }
        new_issue = jira.create_issue(fields=issue_dict)
        print(f"✅ Jira ticket created: {new_issue.key}")
        return new_issue.key
    except Exception as e:
        print(f"❌ Failed to create Jira ticket: {e}")
        return None


# ===========================================================================
# 4. FASTAPI WEBHOOK SERVER  (original feature)
# ===========================================================================

def process_webhook_background(payload: WebhookPayload):
    """Background task: analyse error from webhook payload and create Jira ticket."""
    print(f"\n🚀 Received Webhook Error for workflow: {payload.workflow_name}")
    print(f"   Analysing node: {payload.node_name}")

    analysis = analyze_error_with_langchain(
        payload.code_context, payload.error_message, is_file=False
    )

    description = (
        f"*Workflow:* {payload.workflow_name}\n"
        f"*Failing Node:* {payload.node_name}\n\n"
        f"*Error Message:*\n{{code}}\n{payload.error_message}\n{{code}}\n\n"
        f"*AI Debugger Analysis & Fix:*\n{analysis}"
    )
    summary = f"[Automation Bug] {payload.workflow_name} - {payload.node_name} failed"
    create_jira_ticket(summary, description)


@app.post("/webhook/error")
async def receive_error_webhook(payload: WebhookPayload,
                                background_tasks: BackgroundTasks):
    """
    Endpoint for automation tools (n8n, Make, Zapier) to POST live error payloads.

    Expected JSON body::

        {
          "workflow_name": "My Workflow",
          "node_name":     "HTTP Request",
          "error_message": "Connection refused",
          "code_context":  "Optional code/config snippet"
        }
    """
    background_tasks.add_task(process_webhook_background, payload)
    return {"status": "success",
            "message": "Error received and AI analysis started in background."}


@app.post("/webhook/analyze-json")
async def receive_json_file_webhook(payload: dict,
                                    background_tasks: BackgroundTasks):
    """
    Endpoint to trigger analysis of an already-uploaded workflow JSON file.

    Expected JSON body::

        { "json_path": "/absolute/path/to/workflow.json" }
    """
    json_path = payload.get("json_path")
    if not json_path:
        raise HTTPException(status_code=400, detail="'json_path' field is required.")

    async def bg_task():
        workflow_summary, analysis = analyze_workflow_json_return(json_path)
        summary = f"[Workflow Review] {os.path.basename(json_path)}"
        description = (
            f"*File:* {json_path}\n\n"
            f"*Workflow Structure:*\n{{code}}\n{workflow_summary}\n{{code}}\n\n"
            f"*AI Analysis:*\n{analysis}"
        )
        create_jira_ticket(summary, description)

    background_tasks.add_task(bg_task)
    return {"status": "success",
            "message": f"Workflow JSON '{json_path}' queued for analysis."}


@app.post("/webhook/analyze-json-content")
async def receive_json_content_webhook(payload: JsonContentPayload):
    """
    Synchronous endpoint that accepts raw workflow JSON content from the browser
    and returns the full AI analysis in the response body.

    Expected JSON body::

        { "json_content": { ...workflow JSON object... } }
    """
    data = payload.json_content
    tool = _detect_tool(data)
    extractor = EXTRACTORS.get(tool, EXTRACTORS["generic"])
    workflow_summary = extractor(data)
    analysis = _ai_analyze_workflow(tool, workflow_summary, stream=False)

    # Optionally create a Jira ticket in the background
    name = data.get("name") or data.get("title") or "uploaded-workflow"
    jira_summary = f"[Workflow Review] {tool.upper()} – {name}"
    jira_description = (
        f"*Tool:* {tool.upper()}\n"
        f"*Workflow:* {name}\n\n"
        f"*Workflow Structure:*\n{{code}}\n{workflow_summary}\n{{code}}\n\n"
        f"*AI Analysis & Recommendations:*\n{analysis}"
    )
    create_jira_ticket(jira_summary, jira_description)

    return {
        "status": "success",
        "tool": tool,
        "workflow_name": name,
        "workflow_summary": workflow_summary,
        "analysis": analysis,
    }


# ===========================================================================
# Static Frontend  (MUST be mounted LAST — after all API routes)
# ===========================================================================
# Mounting StaticFiles on "/" at the top would intercept every request
# (including POST /webhook/*) before FastAPI's router gets a chance to match.
# Mounting it here, after all @app.post / @app.get decorators, ensures API
# routes are registered first and only unmatched paths fall through to static.
_public_dir = Path(__file__).parent / "public"
if _public_dir.exists():
    app.mount("/", StaticFiles(directory=str(_public_dir), html=True), name="frontend")


# ===========================================================================
# Entry Point
# ===========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("AI Workflow Debugger – Usage:")
        print("  Analyse a Python script:   python workflow_debugger.py <script.py>")
        print("  Analyse a workflow JSON:   python workflow_debugger.py --json <workflow.json>")
        print("  Start webhook server:      python workflow_debugger.py --server")
        sys.exit(1)

    flag = sys.argv[1]

    if flag == "--server":
        print("🌐 Starting FastAPI Webhook Server...")
        print("   ➜  Open in browser: http://localhost:8000")
        print("   ➜  API docs:        http://localhost:8000/docs")
        uvicorn.run("workflow_debugger:app", host="0.0.0.0", port=8000, reload=True)

    elif flag == "--json":
        if len(sys.argv) < 3:
            print("❌ Please provide a path to the workflow JSON file.")
            print("   Example: python workflow_debugger.py --json workflow.json")
            sys.exit(1)
        analyze_workflow_json(sys.argv[2])

    else:
        # Treat as a Python script path
        target = sys.argv[1]
        if not os.path.exists(target):
            print(f"❌ The file '{target}' does not exist.")
            sys.exit(1)
        run_script_and_catch_errors(target)

