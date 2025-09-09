"""
A local Python execution tool with optional Firejail sandboxing.

- Follows your BaseTool / ToolOutput interfaces.
- Executes user code in a subprocess with rlimits; optionally wraps by Firejail.
- Quick blacklist to reject dangerous imports/calls before execution.
- Supports feeding pseudo-stdin; returns stdout/stderr and status flags.
"""

from __future__ import annotations

import os
import re
import sys
import json
import uuid
import shutil
import subprocess
import resource
from typing import Any, Optional, Union, Tuple, List

# Import base classes (consistent with CalculatorTool example)
from tunix.rl.experimental.agentic.tools import base_tool

ToolOutput = base_tool.ToolOutput
BaseTool = base_tool.BaseTool

# ---------------- config ----------------

TIMEOUT_DEFAULT = 10

# Optional: pre-import common libraries (consistent with original implementation)
PRE_IMPORT_LIBS = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\n"
    "from io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\n"
    "import math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n\n"
)

_FIREJAIL_EXISTS = shutil.which("firejail") is not None


# ---------------- security guards ----------------

_FORBIDDEN_IMPORTS = (
    "import subprocess", "from subprocess",
    "import multiprocessing", "from multiprocessing",
    "import threading", "from threading",
    "import socket", "from socket",
    "import psutil", "from psutil",
    "import resource", "from resource",
    "import ctypes", "from ctypes",
)
_FORBIDDEN_PATTERNS = (
    "os.system", "os.popen", "os.spawn", "os.fork",
    "os.exec", "sys.exit", "os._exit", "os.kill",
)

def _contains_forbidden(code: str) -> Optional[str]:
    """Check if code contains forbidden patterns."""
    low = code.lower()
    for frag in _FORBIDDEN_IMPORTS + _FORBIDDEN_PATTERNS:
        if frag in low:
            return frag
    return None


# ---------------- code wrapping ----------------

def _wrap_code_blocks(code: Union[str, List[str]]) -> str:
    """
    Concatenate multiple code blocks with error-tolerant execution for historical blocks.
    Only the last block executes normally; previous blocks use salvaging execution.
    """
    if isinstance(code, str):
        blocks = [code]
    else:
        blocks = code

    header = "import sys, os, io, ast\n\n"
    helper = r"""
def parse_and_exec_salvageable(code_string):
    """Execute code line by line, skipping syntax/runtime errors."""
    lines = code_string.splitlines()
    current_block = ""
    local_namespace = {}

    for line in lines:
        current_block = (current_block + "\n" + line) if current_block else line
        if not line.strip() or line.strip().startswith('#'):
            continue
        try:
            ast.parse(current_block)
            try:
                exec(current_block, globals(), local_namespace)
                current_block = ""
            except Exception as e:
                print(f"Runtime error in block: {e}")
                current_block = ""
        except SyntaxError:
            pass
    return local_namespace
"""
    parts = [header, helper]
    for i, blk in enumerate(blocks):
        is_last = (i == len(blocks) - 1)
        if not is_last:
            # Historical block - execute with error tolerance
            parts.append(
                f"""
# Code block {i+1} (previous)
original_stdout, original_stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
try:
    exported_vars = parse_and_exec_salvageable(r'''{blk}''')
finally:
    sys.stdout, sys.stderr = original_stdout, original_stderr
    for name, value in exported_vars.items():
        globals()[name] = value
"""
            )
        else:
            # Current block - execute normally
            parts.append(f"\n# Code block {i+1} (current)\n{blk}\n")
    return "".join(parts)


def _clean_traceback(text: str, base_path: str) -> str:
    """Remove temporary file paths from traceback for cleaner output."""
    pattern = re.compile(re.escape('File "' + base_path + "/"))
    return pattern.sub('File "', text)


# ---------------- resource limits ----------------

def _set_limits():
    """Set resource limits for subprocess execution."""
    # Memory/CPU/file size limits - adjust as needed
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_DEFAULT, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_FSIZE, (500 * 1024 * 1024, 500 * 1024 * 1024))


# ---------------- subprocess execution ----------------

def _execute_python(
    code: Union[str, List[str]],
    timeout: int,
    stdin_text: Optional[str],
    python_path: Optional[str],
    pre_import_lib: bool,
    use_firejail: bool
) -> Tuple[str, str, bool, bool]:
    """
    Execute Python code in subprocess.
    Returns: (stdout, stderr, has_error, timed_out)
    """
    # 1) Basic blacklist check
    hit = _contains_forbidden(code if isinstance(code, str) else "\n".join(code))
    if hit:
        return "", f"Execution blocked: contains dangerous fragment '{hit}'.", True, False

    # 2) Setup temporary directory and file
    cwd = os.path.join(os.getcwd(), "tmp", "firejail", uuid.uuid4().hex)
    os.makedirs(cwd, exist_ok=True)
    file_name = "main.py"
    file_path = os.path.join(cwd, file_name)

    # 3) Prepare code
    code_wrapped = _wrap_code_blocks(code)
    if pre_import_lib:
        code_wrapped = PRE_IMPORT_LIBS + code_wrapped
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_wrapped)

    # 4) Build command
    py = python_path or sys.executable or "python3"

    env_min = os.environ.copy()
    if use_firejail and _FIREJAIL_EXISTS:
        # Minimal environment + Firejail resource limits
        keep = [
            "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
            "PYTHONIOENCODING", "PYTHONUNBUFFERED", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
            "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "TMPDIR", "TEMP", "TMP", "DISPLAY", "XAUTHORITY",
        ]
        env = {k: env_min[k] for k in keep if k in env_min}
        env["OPENBLAS_NUM_THREADS"] = "1"
        if "PYTHONPATH" in env:
            del env["PYTHONPATH"]

        command = [
            "firejail", "--quiet", "--seccomp=socket", "--noprofile",
            "--rlimit-nproc=32", "--rlimit-nofile=32",
            "--rlimit-fsize=2m", "--rlimit-as=1096m",
            py, file_path
        ]
        proc_cwd = cwd
    else:
        env = env_min
        command = [py, file_name]
        proc_cwd = cwd

    # 5) Execute
    has_error = False
    timed_out = False
    try:
        result = subprocess.run(
            command,
            input=stdin_text if stdin_text else None,
            env=env,
            text=True,
            capture_output=True,
            preexec_fn=_set_limits,
            timeout=max(1, int(timeout)),
            cwd=proc_cwd,
        )
        stdout = _clean_traceback(result.stdout, cwd)
        stderr = _clean_traceback(result.stderr, cwd) or ""
        has_error = bool(stderr)
    except subprocess.TimeoutExpired as e:
        timed_out = True
        has_error = True
        stdout = (e.stdout.decode("utf-8") if isinstance(e.stdout, bytes) else (e.stdout or "")) or ""
        stderr_raw = (e.stderr.decode("utf-8") if isinstance(e.stderr, bytes) else (e.stderr or "")) or ""
        stderr = _clean_traceback(stderr_raw, cwd) + f"\nExecution timed out after {timeout} seconds."
    finally:
        # Cleanup temporary files
        try:
            shutil.rmtree(cwd, ignore_errors=True)
        except Exception:
            pass

    return stdout, stderr, has_error, timed_out


# ---------------- the tool ----------------

class PythonCodeTool(BaseTool):
    """
    Execute Python code securely inside a local subprocess with optional Firejail.

    Parameters (json schema):
      - code (str, required): Python source to run. You can pass multiple cells by concatenating text.
      - timeout (int, default 10): Wall-clock timeout in seconds.
      - stdin (str, optional): Pseudo-stdin content. Each line corresponds to one `input()`.
      - python_path (str, optional): Path to Python interpreter; default to current runtime.
      - pre_import_lib (bool, default True): Whether to pre-import common stdlibs & set recursionlimit.
      - use_firejail (bool, default True): Enable Firejail if present on Linux.
    """

    def __init__(self, name: str = "python", description: str = "Execute Python code in a local sandbox (Firejail+rlimit). Returns stdout/stderr."):
        super().__init__(name=name, description=description)

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. Multiple cells allowed; later cells see variables from earlier ones.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max execution time in seconds.",
                            "default": TIMEOUT_DEFAULT,
                            "minimum": 1,
                            "maximum": 120,
                        },
                        "stdin": {
                            "type": "string",
                            "description": "Optional stdin text. Each line consumed by one `input()`.",
                        },
                        "python_path": {
                            "type": "string",
                            "description": "Optional explicit Python interpreter path.",
                        },
                        "pre_import_lib": {
                            "type": "boolean",
                            "description": "Pre-import handy stdlibs & raise recursion limit.",
                            "default": True,
                        },
                        "use_firejail": {
                            "type": "boolean",
                            "description": "On Linux, run under Firejail if available.",
                            "default": True,
                        },
                    },
                    "required": ["code"],
                },
            },
        }

    def apply(
        self,
        code: Union[str, List[str]],
        timeout: int = TIMEOUT_DEFAULT,
        stdin: Optional[str] = None,
        python_path: Optional[str] = None,
        pre_import_lib: bool = True,
        use_firejail: bool = True,
    ) -> ToolOutput:
        # Execute code
        stdout, stderr, has_error, timed_out = _execute_python(
            code=code,
            timeout=timeout,
            stdin_text=stdin,
            python_path=python_path,
            pre_import_lib=pre_import_lib,
            use_firejail=use_firejail,
        )

        payload = {
            "stdout": stdout or None,
            "stderr": stderr or None,
            "timed_out": timed_out,
            "has_error": has_error,
            # Add more execution info as needed
        }

        # Optionally upgrade timeout/blocked to ToolOutput.error
        if timed_out:
            return ToolOutput(name=self.name, error=f"Timeout after {timeout}s", metadata=payload)

        return ToolOutput(name=self.name, output=payload)