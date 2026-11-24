from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

ACTIONS = ["inspect", "edit", "test", "submit", "other", "idle"]
SYMBOLS = [f"{a}+{e}" for a in ACTIONS for e in ("ok", "err")]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}

ACTION_RULES: dict[str, str] = {
    **{c: "inspect" for c in [
        "open", "goto", "scroll_up", "scroll_down", "find_file", "search_dir",
        "search_file", "ls", "cd", "pwd", "cat", "head", "tail", "less", "more",
        "grep", "egrep", "rg", "find", "tree", "wc", "diff", "file", "which",
        "search", "search_github", "next_match", "prev_match"]},
    **{c: "edit" for c in [
        "edit", "create", "insert", "append", "rm", "mv", "cp", "touch", "mkdir",
        "patch", "str_replace", "apply_patch"]},
    **{c: "test" for c in [
        "python", "python3", "pytest", "py.test", "tox", "nosetests", "make",
        "pip", "pip3", "bash", "sh", "node", "npm", "yarn", "go", "cargo",
        "java", "javac", "mvn", "gradle", "ruby", "rake", "php", "./",
        "mypy", "flake8", "pyflakes", "pylint", "ruff", "black", "isort",
        "pyupgrade", "nose2", "coverage", "asv", "unittest"]},
    "sed": "inspect", "awk": "inspect",
    **{c: "idle" for c in [
        "sleep", "wait", "true", ":", "end", "end_of_session", "exit", "quit"]},
    "submit": "submit",
}

_PRIORITY = {"submit": 5, "test": 4, "edit": 3, "inspect": 2, "other": 1, "idle": 0}
_SHELL_OPS = {"&&", "||", ";", "|", "&", ";;"}

_RUNTIME_ERR = re.compile(
    r"Traceback \(most recent call last\)|^\s*\w*(Error|Exception)\b[:\s]|"
    r"\bFAILED\b|\b\d+ failed\b|command not found|No such file or directory|"
    r"Segmentation fault|SyntaxError|exit code [1-9]|"
    r"\b[1-9]\d* errors?\b|^ERROR\b(?!: pip's dependency resolver)|No module named|"
    r"can't open file|No executable found",
    re.MULTILINE)
_SHELL_FAIL = re.compile(
    r"EXECUTION TIMED OUT|syntax error near unexpected token|unexpected EOF while looking|"
    r"command not found|Is a directory|Permission denied|"
    r"cannot (?:create|remove|stat|access|open)|No such file or directory|"
    r"^[Uu]sage: ",
    re.MULTILINE)
_SHOWS_CONTENT = re.compile(r"\[File: |Found \d+ matches")
_EDIT_ERR = re.compile(
    r"Your proposed edit has introduced new syntax error|E999|"
    r"Error: |No file open|Usage: edit|already exists|No such file", re.IGNORECASE)
_INTERFACE_ERR = re.compile(
    r"not found|No such file or directory|Usage: |No matches found|"
    r"does not exist|must be a|Invalid|command not found|"
    r"^Error: ",
    re.IGNORECASE | re.MULTILINE)
_TEST_PASSED = re.compile(r"\b\d+ passed\b|^OK\b|Ran \d+ tests?.*\n+OK", re.MULTILINE)
_NO_MATCH = re.compile(r"^No matches found", re.MULTILINE)
_SUCCESS_MSG = re.compile(
    r"completed successfully|ran successfully(?! and did not produce any output)|"
    r"\bno errors\b|all tests passed|tests? passed|works as expected", re.IGNORECASE)

_LEADING_WORD = re.compile(r"[A-Za-z_]+[0-9]*")
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_OPEN_FILE = re.compile(r"\(Open file: ([^)\n]+)\)")
_FILE_HEADER = re.compile(r"\[File: ([^\s\]]+)")


@dataclass
class Step:
    action: str
    error: bool
    command: str
    viewed_file: str | None
    edited_file: str | None
    test_passed: bool
    no_match: bool = False
    success_msg: bool = False

    @property
    def symbol(self) -> int:
        return SYMBOL_TO_ID[f"{self.action}+{'err' if self.error else 'ok'}"]


def extract_command(ai_text: str) -> str:
    blocks = _FENCE.findall(ai_text or "")
    return blocks[-1].strip() if blocks else ""


def repo_name(instance_id: str | None) -> str | None:
    if not instance_id or "__" not in instance_id:
        return None
    return re.sub(r"-\d+$", "", instance_id.split("__", 1)[1])


def _norm_name(s: str) -> str:
    return re.sub(r"[-_.]", "", s.lower())


def _is_repo_cli(head: str, repo: str | None) -> bool:
    if not repo or len(head) < 2:
        return False
    h, r = _norm_name(head), _norm_name(repo)
    return h == r or (len(h) >= 3 and (r.startswith(h) or h.startswith(r)))


def _segments(line: str) -> list[list[str]]:
    try:
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        tokens = re.findall(r"&&|\|\||[;|]|[^\s;|&]+", line)
    segs, cur = [], []
    for tok in tokens:
        if tok in _SHELL_OPS:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        segs.append(cur)
    return segs


def _writes_file(tokens: list[str]) -> bool:
    for i, tok in enumerate(tokens):
        if tok in (">", ">>") and i + 1 < len(tokens) and tokens[i + 1] != "/dev/null":
            if not (i > 0 and tokens[i - 1].isdigit()):
                return True
    return False


def _classify_segment(tokens: list[str], repo: str | None) -> str:
    head = tokens[0]
    if head.startswith("./"):
        return "test"
    if "/" in head:
        head = head.rstrip("/").rsplit("/", 1)[-1] or head
    if head in ("echo", "printf") and not _writes_file(tokens):
        return "inspect" if any("$" in t for t in tokens[1:]) else "idle"
    if head == "sed" and any(t.startswith("-i") for t in tokens):
        return "edit"
    if head in ("echo", "printf", "cat") and _writes_file(tokens):
        return "edit"
    if head == "git":
        sub = tokens[1] if len(tokens) > 1 else ""
        return "edit" if sub in ("apply", "checkout", "restore", "stash", "reset") else "inspect"
    if head in ACTION_RULES:
        return ACTION_RULES[head]
    if _is_repo_cli(head, repo):
        return "test"
    m = _LEADING_WORD.match(head)
    return ACTION_RULES.get(m.group(0), "other") if m else "other"


def classify_action(command: str, repo: str | None = None) -> str:
    if not command:
        return "other"
    first_line = command.splitlines()[0].strip()
    segs = _segments(first_line)
    if not segs:
        return "other"
    return max((_classify_segment(s, repo) for s in segs), key=_PRIORITY.__getitem__)


def is_error(action: str, observation: str) -> bool:
    obs = observation or ""
    if action == "submit":
        return False
    head = "\n".join(obs.splitlines()[:3])
    shows_content = _SHOWS_CONTENT.match(obs.lstrip())
    if not shows_content and _SHELL_FAIL.search(head):
        return True
    if action == "test":
        return bool(_RUNTIME_ERR.search(obs))
    if action == "edit":
        return bool(_EDIT_ERR.search(head))
    if action == "inspect":
        return bool(_INTERFACE_ERR.search(head))
    return bool(_RUNTIME_ERR.search(obs)) or bool(_INTERFACE_ERR.search(obs[:300]))


def _normalize(command: str) -> str:
    return " ".join(command.splitlines()[0].split()) if command else ""


def parse_trajectory(trajectory: list[dict], instance_id: str | None = None
                     ) -> tuple[list[Step], str]:
    repo = repo_name(instance_id)
    msgs = [m for m in trajectory if m.get("role") in ("user", "ai")]
    issue = ""
    if msgs and msgs[0]["role"] == "user":
        text = msgs[0].get("text") or ""
        m = re.search(r"ISSUE:\n(.*?)(?:\nINSTRUCTIONS:|\Z)", text, re.DOTALL)
        issue = m.group(1) if m else text

    steps: list[Step] = []
    open_file: str | None = None
    for i, m in enumerate(msgs):
        if m["role"] != "ai":
            continue
        has_obs = i + 1 < len(msgs) and msgs[i + 1]["role"] == "user"
        obs = (msgs[i + 1].get("text") or "") if has_obs else ""
        command = extract_command(m.get("text") or "")
        action = classify_action(command, repo)
        err = is_error(action, obs)

        header = _FILE_HEADER.search(obs)
        viewed = header.group(1) if (header and action == "inspect") else None
        edited = None
        if action == "edit" and not err:
            verb = command.split(maxsplit=1)[0] if command else ""
            if header:
                edited = header.group(1)
            elif verb in ("edit", "insert", "append"):
                edited = open_file
        of = _OPEN_FILE.search(obs)
        if of and of.group(1).strip() != "n/a":
            open_file = of.group(1).strip()

        steps.append(Step(
            action=action, error=err, command=_normalize(command),
            viewed_file=viewed, edited_file=edited,
            test_passed=(action == "test" and not err and bool(_TEST_PASSED.search(obs))),
            no_match=(action == "inspect" and bool(_NO_MATCH.search("\n".join(obs.splitlines()[:2])))),
            success_msg=(action == "test" and not err and bool(_SUCCESS_MSG.search(obs))),
        ))
    return steps, issue
