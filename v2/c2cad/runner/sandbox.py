"""Execute model-written Python (tool arm) and CadQuery programs (cadquery arm) in a restricted subprocess.

Guards: an AST allow-list (imports limited per arm; no process, network, reflection or dunder access;
numpy / CadQuery file I/O names rejected), a separate interpreter in isolated mode started in an empty
temporary directory, RLIMIT_FSIZE = 0 (any file write fails), CPU and memory rlimits, a wall-clock timeout and
an output cap. This is defence in depth for code produced by hosted model APIs; it is not a security boundary
for adversarial code.

The cadquery arm runs in the CAD environment (v2/.venv-cad, CadQuery 2.8). A trusted epilogue, appended after
the checked program, collects `parts` (or `result`, or objects passed to show_object) and prints the primitives
recovered from the solids (c2cad/cadkernel.py) behind a marker line.
"""
from __future__ import annotations

import ast
import json
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]
CAD_PYTHON = V2 / ".venv-cad" / "bin" / "python"

ALLOWED_IMPORTS = {"math", "numpy", "itertools", "json"}
CAD_ALLOWED_IMPORTS = ALLOWED_IMPORTS | {"cadquery"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "os", "sys",
                "subprocess", "socket", "importlib", "builtins", "__builtins__", "type"}
# numpy / json / CadQuery names that read or write files, or load code
BANNED_ATTRS = {"load", "loadtxt", "genfromtxt", "fromfile", "tofile", "save", "savez", "savez_compressed", "savetxt",
                "memmap", "fromregex", "DataSource", "f2py", "ctypeslib", "lib", "distutils", "testing", "dump",
                "exporters", "importers", "export", "exportStep", "exportStl", "exportBrep", "exportBin", "exportSvg",
                "exportDXF", "exportVTP", "exportGLTF", "importStep", "importBrep", "importStl", "importDXF",
                "importBin", "occ_impl", "cq_directive", "vis", "show", "occ"}
CPU_SECONDS, MEM_BYTES, WALL_SECONDS, MAX_OUT = 20, 2 * 1024 ** 3, 30, 20 * 1024 ** 2
CAD_CPU_SECONDS, CAD_WALL_SECONDS = 120, 240


class Rejected(Exception):
    pass


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    return text.strip()


def check(code: str, allowed=ALLOWED_IMPORTS) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise Rejected(f"syntax: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed:
                    raise Rejected(f"import {a.name}")
                if any(part in BANNED_ATTRS for part in a.name.split(".")[1:]):
                    raise Rejected(f"import {a.name} (file i/o)")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] not in allowed:
                raise Rejected(f"from {node.module} import")
            if any(part in BANNED_ATTRS for part in mod.split(".")[1:]):
                raise Rejected(f"from {node.module} import (file i/o)")
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


def _limits(cpu):
    def f():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
        except (ValueError, OSError):
            pass   # macOS does not enforce RLIMIT_AS; CPU and wall-clock limits still apply
    return f


def _exec(python: str, source: str, cpu: int, wall: int) -> tuple[bool, str, str]:
    try:
        with tempfile.TemporaryDirectory(prefix="c2cad_tool_") as cwd:
            p = subprocess.run([python, "-I", "-c", source], capture_output=True, text=True, cwd=cwd,
                               timeout=wall, preexec_fn=_limits(cpu), env={"PATH": "/usr/bin:/bin", "HOME": cwd})
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    out = p.stdout[:MAX_OUT]
    if p.returncode != 0:
        return False, out, f"exit {p.returncode}: {p.stderr[-800:]}"
    return True, out, ""


def run(code: str) -> tuple[bool, str, str]:
    """Tool arm: (ok, stdout, error)."""
    try:
        check(code)
    except Rejected as e:
        return False, "", f"rejected: {e}"
    return _exec(sys.executable, code, CPU_SECONDS, WALL_SECONDS)


CAD_PRELUDE = "_C2CAD_SHOWN = []\ndef show_object(obj, *args, **kwargs):\n    _C2CAD_SHOWN.append(obj)\n"
CAD_EPILOGUE = """
import sys as _c2cad_sys
_c2cad_sys.path.insert(0, {v2!r})
from c2cad import cadkernel as _c2cad_kernel
_c2cad_obj = globals().get("parts")
if _c2cad_obj is None:
    _c2cad_obj = globals().get("result")
if _c2cad_obj is None and _C2CAD_SHOWN:
    _c2cad_obj = _C2CAD_SHOWN if len(_C2CAD_SHOWN) > 1 else _C2CAD_SHOWN[0]
if _c2cad_obj is None:
    raise SystemExit("no parts: define `parts` (a dict id -> solid)")
_c2cad_kernel.emit(_c2cad_obj)
"""


def cad_available() -> bool:
    return CAD_PYTHON.exists()


def run_cadquery(code: str) -> tuple[bool, dict | None, str]:
    """CadQuery arm: (ok, {"parts": [...], "stats": {...}} or None, error)."""
    try:
        check(code, CAD_ALLOWED_IMPORTS)
    except Rejected as e:
        return False, None, f"rejected: {e}"
    if not cad_available():
        return False, None, f"CAD environment missing: {CAD_PYTHON}"
    src = CAD_PRELUDE + code + "\n" + CAD_EPILOGUE.format(v2=str(V2))
    ok, out, err = _exec(str(CAD_PYTHON), src, CAD_CPU_SECONDS, CAD_WALL_SECONDS)
    from .cad_marker import MARK
    line = next((l for l in reversed(out.splitlines()) if l.startswith(MARK)), None)
    if not ok:
        return False, None, err
    if line is None:
        return False, None, "program ran but produced no parts"
    return True, json.loads(line[len(MARK):]), ""
