from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

ACTIONS = ["inspect", "edit", "test", "submit", "other"]
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
        "pyupgrade", "nose2", "coverage", "asv", "unittest",
        "markdown", "pandoc"]},
    **{c: "inspect" for c in ["chrome", "google-chrome", "chromium", "firefox", "xdg-open"]},
    "submit": "submit",
}

ACTION_PRIORITY = ["submit", "test", "edit", "inspect", "other"]
_SEPARATORS = {"&&", "||", ";", "|", "&"}

_RUNTIME_ERR = re.compile(
    r"Traceback \(most recent call last\)|^\s*\w*(Error|Exception)\b[:\s]|"
    r"\bFAILED\b|\b\d+ failed\b|command not found|No such file or directory|"
    r"Segmentation fault|SyntaxError|exit code [1-9]|"
    r"^\s*ERROR\b(?!: pip's dependency resolver)",
    re.MULTILINE)
_SHELL_ERR = re.compile(
    r"^/bin/(ba)?sh: line \d+: |unexpected EOF while looking for matching|"
    r"syntax error near unexpected token", re.MULTILINE)
_USAGE = re.compile(r"^\s*usage: ", re.IGNORECASE | re.MULTILINE)
_ASKS_HELP = re.compile(r"(^|\s)(-h|--help|help)(\s|$)")
_EDIT_ERR = re.compile(
    r"Your proposed edit has introduced new syntax error|E999|"
    r"Error: |No file open|Usage: edit|already exists|No such file|File exists",
    re.IGNORECASE)
_TIMEOUT = "EXECUTION TIMED OUT"
_INTERFACE_ERR = re.compile(
    r"not found|No such file or directory|Usage: |No matches found|"
    r"does not exist|must be a|Invalid|command not found", re.IGNORECASE)
_INTERFACE_ERR_NO_USAGE = re.compile(
    r"not found|No such file or directory|No matches found|"
    r"does not exist|must be a|Invalid|command not found", re.IGNORECASE)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TEST_PASSED = re.compile(r"\b\d+ passed\b|^OK\b|Ran \d+ tests?.*\n+OK", re.MULTILINE)

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

    @property
    def symbol(self) -> int:
        return SYMBOL_TO_ID[f"{self.action}+{'err' if self.error else 'ok'}"]


def extract_command(ai_text: str) -> str:
    blocks = _FENCE.findall(ai_text or "")
    return blocks[-1].strip() if blocks else ""


def repo_from_instance(instance_id: str | None) -> str | None:
    if not instance_id or "__" not in instance_id:
        return None
    return instance_id.split("__", 1)[1].rsplit("-", 1)[0].lower()


def _split_segments(line: str) -> list[list[str]]:
    try:
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        tokens = re.sub(r"(&&|\|\||;|\|)", r" \1 ", line).split()
    segments, cur = [], []
    for tok in tokens:
        if tok in _SEPARATORS:
            if cur:
                segments.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        segments.append(cur)
    return segments


def _writes_file(tokens: list[str]) -> bool:
    for k, tok in enumerate(tokens):
        if tok in (">", ">>") and k + 1 < len(tokens) and tokens[k + 1] != "/dev/null":
            if not (k > 0 and tokens[k - 1] == "2"):
                return True
    return False


def _is_repo_cli(head: str, repo: str | None) -> bool:
    if not repo or len(head) < 3:
        return False
    head = head.lower()
    return head == repo or repo.startswith(head)


def _classify_segment(tokens: list[str], repo: str | None) -> str:
    head = tokens[0]
    if head.startswith("./"):
        return "test"
    if head == "sed" and "-i" in tokens:
        return "edit"
    if head in ("echo", "printf", "cat") and _writes_file(tokens):
        return "edit"
    if head == "git":
        sub = tokens[1] if len(tokens) > 1 else ""
        return "edit" if sub in ("apply", "checkout", "restore", "stash", "reset") else "inspect"
    if head in ACTION_RULES:
        return ACTION_RULES[head]
    m = _LEADING_WORD.match(head)
    if m and m.group(0) in ACTION_RULES:
        return ACTION_RULES[m.group(0)]
    if _is_repo_cli(head, repo):
        return "test"
    return "other"


def classify_action(command: str, repo: str | None = None) -> str:
    if not command:
        return "other"
    first_line = command.splitlines()[0].strip()
    actions = [_classify_segment(seg, repo) for seg in _split_segments(first_line)]
    if not actions:
        return "other"
    return min(actions, key=ACTION_PRIORITY.index)


def is_error(action: str, observation: str, command: str = "") -> bool:
    obs = _ANSI.sub("", observation or "")
    if action == "submit":
        return False
    head = "\n".join(obs.splitlines()[:3])
    if _TIMEOUT in obs or _SHELL_ERR.search(head if action in ("edit", "inspect") else obs):
        return True
    if action in ("test", "other") and _USAGE.search(head) and not _ASKS_HELP.search(command):
        return True
    if action == "test":
        return bool(_RUNTIME_ERR.search(obs))
    if action == "edit":
        return bool(_EDIT_ERR.search(head))
    if action == "inspect":
        return bool(_INTERFACE_ERR.search(head))
    if action == "other":
        return bool(_RUNTIME_ERR.search(obs)) or bool(_INTERFACE_ERR_NO_USAGE.search(obs[:300]))
    return False


def _normalize(command: str) -> str:
    return " ".join(command.splitlines()[0].split()) if command else ""


def parse_trajectory(trajectory: list[dict],
                     instance_id: str | None = None) -> tuple[list[Step], str]:
    repo = repo_from_instance(instance_id)
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
        err = is_error(action, obs, command)

        header = _FILE_HEADER.search(obs)
        viewed = header.group(1) if (header and action == "inspect") else None
        edited = None
        if action == "edit" and not err:
            edited = header.group(1) if header else open_file
        of = _OPEN_FILE.search(obs)
        if of and of.group(1).strip() != "n/a":
            open_file = of.group(1).strip()

        steps.append(Step(
            action=action, error=err, command=_normalize(command),
            viewed_file=viewed, edited_file=edited,
            test_passed=(action == "test" and not err and bool(_TEST_PASSED.search(obs))),
        ))
    return steps, issue
