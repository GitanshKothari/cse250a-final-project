import pandas as pd
import pytest

from hmm_agents.parsing import classify_action, extract_command, is_error, parse_trajectory
from hmm_agents.splits import make_splits


@pytest.mark.parametrize("cmd,action", [
    ("open lexicon/providers/memset.py 144", "inspect"),
    ('find_file "memset.py" lexicon', "inspect"),
    ("search_dir \"raise_for_status\"", "inspect"),
    ("scroll_down", "inspect"),
    ("ls -F", "inspect"),
    ("edit 138:138\n        r = requests.request(...)\nend_of_edit", "edit"),
    ("create reproduce.py", "edit"),
    ("sed -i 's/a/b/' foo.py", "edit"),
    ("python reproduce.py", "test"),
    ("pytest tests/test_x.py -k foo", "test"),
    ("submit", "submit"),
    ("find. -name '*.py'", "inspect"),
    ("cd..", "inspect"),
    ("mypy src/", "test"),
    ("lexicon memset create example.com TXT --name x", "other"),
    ("", "other"),
])
def test_classify_action(cmd, action):
    assert classify_action(cmd) == action


def test_extract_command_takes_last_fenced_block():
    text = "Thought...\n```\nls\n```\nActually:\n```\nopen a.py\n```"
    assert extract_command(text) == "open a.py"
    assert extract_command("no command here") == ""


def test_error_detection_is_action_aware():
    code_listing = "[File: /x/a.py (10 lines total)]\n1: raise ValueError('bad')\n"
    assert not is_error("inspect", code_listing)
    assert is_error("test", "Traceback (most recent call last):\n  File ...\nValueError: x")
    assert is_error("edit", "Your proposed edit has introduced new syntax error(s).")
    assert is_error("inspect", "Directory src not found\n\n(Open file: n/a)")
    assert not is_error("test", "All good\n3 passed in 0.1s")
    edit_ok = "[File: /r/a.py (40 lines total)]\n1: import x\n2: x = 1\n3: y = 2\n4: print('Error: bad')"
    assert not is_error("edit", edit_ok)


def test_parse_trajectory_pairs_actions_with_observations():
    traj = [
        {"role": "system", "text": None},
        {"role": "user", "text": "ISSUE:\nIt breaks\nINSTRUCTIONS:\n..."},
        {"role": "ai", "text": "```\nopen a.py\n```"},
        {"role": "user", "text": "[File: /r/a.py (5 lines total)]\n(Open file: /r/a.py)"},
        {"role": "ai", "text": "```\nedit 1:1\nx = 1\nend_of_edit\n```"},
        {"role": "user", "text": "[File: /r/a.py (5 lines total)]\nFile updated."},
        {"role": "ai", "text": "```\npython a.py\n```"},
        {"role": "user", "text": "Traceback (most recent call last):\nNameError: y"},
        {"role": "ai", "text": "```\nsubmit\n```"},
    ]
    steps, issue = parse_trajectory(traj)
    assert issue == "It breaks"
    assert [s.action for s in steps] == ["inspect", "edit", "test", "submit"]
    assert [s.error for s in steps] == [False, False, True, False]
    assert steps[0].viewed_file == "/r/a.py" and steps[1].edited_file == "/r/a.py"


def test_splits_keep_instances_together():
    df = pd.DataFrame({"instance_id": [f"i{k // 3}" for k in range(300)],
                       "target": [k % 4 == 0 for k in range(300)]})
    df["split"] = make_splits(df, test=0.15, val=0.15, hmm_frac_of_train=0.5, seed=0).values
    assert df.groupby("instance_id")["split"].nunique().max() == 1
    assert set(df.split) == {"hmm_train", "clf_train", "val", "test"}


@pytest.mark.parametrize("cmd,repo,action", [
    ("cd /repo && python reproduce.py", None, "test"),
    ("cd.. && pytest tests/", None, "test"),
    ("mkdir out && touch out/a.py", None, "edit"),
    ("grep -rn foo src | head -20", None, "inspect"),
    ('search_dir "a|b" src', None, "inspect"),
    ("chmod +x run.sh && ./run.sh", None, "test"),
    ("cat setup.py 2>/dev/null", None, "inspect"),
    ("python x.py 2>&1", None, "test"),
    ("echo 'x = 1' >> a.py", None, "edit"),
    ("sed -n '10,20p' a.py", None, "inspect"),
    ("sed -i 's/a/b/' a.py", None, "edit"),
    ("dvc add foo", "dvc", "test"),
    ("beet import x", "beets", "test"),
    ("pierone login", "pierone-cli", "test"),
    ("dvc add foo", None, "other"),
    ("search_engine 'foo'", "dvc", "other"),
])
def test_classify_action_audit_cases(cmd, repo, action):
    assert classify_action(cmd, repo) == action


@pytest.mark.parametrize("action,obs,err", [
    ("test", "EXECUTION TIMED OUT", True),
    ("other", "/bin/bash: line 3: syntax error near unexpected token `('", True),
    ("test", "/usr/bin/python: No module named foo.__main__; 'foo' is a package", True),
    ("test", "==== 1 error in 0.31s ====", True),
    ("test", "Found 0 errors in 3 files", False),
    ("edit", "mkdir: cannot create directory 'x': File exists", True),
    ("edit", "EXECUTION TIMED OUT", True),
    ("inspect", "[File: /r/a.py (3 lines total)]\n1: raise Exception('Permission denied')", False),
    ("submit", "EXECUTION TIMED OUT", False),
])
def test_is_error_audit_cases(action, obs, err):
    assert is_error(action, obs) == err


def test_shell_edit_does_not_claim_open_file():
    traj = [
        {"role": "user", "text": "ISSUE:\nx\nINSTRUCTIONS:\n"},
        {"role": "ai", "text": "```\nopen a.py\n```"},
        {"role": "user", "text": "[File: /r/a.py (5 lines total)]\n(Open file: /r/a.py)"},
        {"role": "ai", "text": "```\nrm reproduce.py\n```"},
        {"role": "user", "text": "Your command ran successfully and did not produce any output.\n(Open file: /r/a.py)"},
    ]
    steps, _ = parse_trajectory(traj, "o__r-1")
    assert steps[1].action == "edit" and steps[1].edited_file is None


@pytest.mark.parametrize("action,obs", [
    ("other", "/bin/bash: line 2: unexpected EOF while looking for matching `\"'"),
    ("test", "usage: asv [-h] [--verbose] [--config CONFIG] [--version]"),
])
def test_is_error_bad_invocation(action, obs):
    assert is_error(action, obs)


def test_interface_error_message_on_inspect():
    assert is_error("inspect", "Error: <line> must be less than or equal to 8\n(Open file: /a.py)")
    assert not is_error("inspect", "[File: /a.py (3 lines total)]\n1:Error: not at line start in listing")


@pytest.mark.parametrize("cmd,action", [
    ('echo "Waiting for further instructions"', "idle"),
    ("echo Session ended.", "idle"),
    ("sleep 30", "idle"),
    ("wait for the administrator", "idle"),
    ("end_of_session", "idle"),
    ("exit", "idle"),
    ("echo $PATH", "inspect"),
    ('echo "$(</repo/foo.py)"', "inspect"),
    ('echo "notes" > findings.txt', "edit"),
    ('echo "class X" | search_file "X" a.py', "inspect"),
    ("/opt/conda/envs/x/bin/python reproduce.py", "test"),
    ("/usr/bin/ls -l", "inspect"),
])
def test_classify_action_idle_and_paths(cmd, action):
    assert classify_action(cmd) == action


def test_step_flags_no_match_and_success_msg():
    traj = [
        {"role": "user", "text": "ISSUE:\nx\nINSTRUCTIONS:\n"},
        {"role": "ai", "text": '```\nsearch_dir "foo"\n```'},
        {"role": "user", "text": 'No matches found for "foo" in /r\n(Open file: n/a)'},
        {"role": "ai", "text": "```\npython reproduce.py\n```"},
        {"role": "user", "text": "Script completed successfully, no errors.\n(Open file: n/a)"},
        {"role": "ai", "text": "```\npython other.py\n```"},
        {"role": "user", "text": "Your command ran successfully and did not produce any output."},
    ]
    steps, _ = parse_trajectory(traj, "o__r-1")
    assert steps[0].error and steps[0].no_match
    assert not steps[1].error and steps[1].success_msg and not steps[1].test_passed
    assert not steps[2].success_msg


def test_pip_resolver_notice_is_not_an_error():
    obs = ("Successfully uninstalled fsspec-2022.11.0\n"
           "ERROR: pip's dependency resolver does not currently take into account all the packages\n"
           "Successfully installed s3fs-2024.9.0 boto3-1.35.39\n")
    assert not is_error("test", obs)
    assert is_error("test", "ERROR: Could not find a version that satisfies the requirement foo")
