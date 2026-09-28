"""Execute model-written Python (tool arm) in a restricted subprocess.

Guards: an AST allow-list (imports limited to math, numpy, itertools, json; no process, network,
reflection or dunder access; numpy's file I/O names rejected), a separate interpreter in isolated
mode started in an empty temporary directory, RLIMIT_FSIZE = 0 (any file write fails), CPU and
memory rlimits, a wall-clock timeout and an output cap. This is defence in depth for code produced
by hosted model APIs; it is not a security boundary for adversarial code.
"""
from __future__ import annotations

import ast
import re
import resource
import subprocess
import sys
import tempfile

ALLOWED_IMPORTS = {"math", "numpy", "itertools", "json"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "os", "sys",
                "subprocess", "socket", "importlib", "builtins", "__builtins__", "type"}
# numpy / json names that read or write files, or load code
BANNED_ATTRS = {"load", "loadtxt", "genfromtxt", "fromfile", "tofile", "save", "savez", "savez_compressed", "savetxt",
                "memmap", "fromregex", "DataSource", "f2py", "ctypeslib", "lib", "distutils", "testing", "dump"}
CPU_SECONDS, MEM_BYTES, WALL_SECONDS, MAX_OUT = 20, 2 * 1024 ** 3, 30, 20 * 1024 ** 2


class Rejected(Exception):
    pass


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    return text.strip()


def check(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise Rejected(f"syntax: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise Rejected(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise Rejected(f"from {node.module} import")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise Rejected(f"name {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise Rejected(f"attribute {node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            raise Rejected(f"attribute {node.attr} (file i/o)")
        elif isinstance(node, ast.alias) and node.name in BANNED_ATTRS:
            raise Rejected(f"import of {node.name} (file i/o)")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise Rejected("global/nonlocal")


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    except (ValueError, OSError):
        pass   # macOS does not enforce RLIMIT_AS; CPU and wall-clock limits still apply


def run(code: str) -> tuple[bool, str, str]:
    """(ok, stdout, error)."""
    try:
        check(code)
    except Rejected as e:
        return False, "", f"rejected: {e}"
    try:
        with tempfile.TemporaryDirectory(prefix="c2cad_tool_") as cwd:
            p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=cwd,
                               timeout=WALL_SECONDS, preexec_fn=_limits, env={"PATH": "/usr/bin:/bin", "HOME": cwd})
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    out = p.stdout[:MAX_OUT]
    if p.returncode != 0:
        return False, out, f"exit {p.returncode}: {p.stderr[-800:]}"
    return True, out, ""
