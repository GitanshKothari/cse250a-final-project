import pandas as pd
import pytest

from hmm_agents.parsing import (classify_action, extract_command, is_error, parse_trajectory,
                                repo_from_instance)
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


def test_repo_from_instance():
    assert repo_from_instance("hugovk__pypistats-41") == "pypistats"
    assert repo_from_instance("iterative__dvc-8425") == "dvc"
    assert repo_from_instance("pydantic__pydantic-4399") == "pydantic"
    assert repo_from_instance(None) is None


@pytest.mark.parametrize("cmd,instance,action", [
    ("pypistats python_minor pylast -m 2018-12", "hugovk__pypistats-41", "test"),
    ("lexicon memset create example.com TXT --name x", "AnalogJ__lexicon-336", "test"),
    ("dvc exp show", "iterative__dvc-8425", "test"),
    ("pypistats python_minor pylast", None, "other"),
    ("search_engine foo", "iterative__dvc-8425", "other"),
    ("cd docs && mkdir includes", None, "edit"),
    ("mkdir docs && mkdir docs/includes", None, "edit"),
    ("cd /repo && python reproduce.py", None, "test"),
    ("pip install flake8 && flake8 test.py", None, "test"),
    ("grep -rn foo src | head -20", None, "inspect"),
    ('search_dir "a|b" src', None, "inspect"),
    ('python -c "import x; print(x)"', None, "test"),
    ("cat setup.py 2>/dev/null", None, "inspect"),
    ("echo 'x = 1' >> a.py", None, "edit"),
    ("echo hello", None, "other"),
    ("markdown parent-file.md", None, "test"),
    ("pandoc parent-file.md", None, "test"),
    ("chrome parent-file.md", None, "inspect"),
])
def test_classify_action_round1(cmd, instance, action):
    assert classify_action(cmd, repo_from_instance(instance)) == action


@pytest.mark.parametrize("action,obs,err", [
    ("edit", "EXECUTION TIMED OUT\n(Open file: /beniget/test.py)", True),
    ("test", "EXECUTION TIMED OUT", True),
    ("inspect", "EXECUTION TIMED OUT", True),
    ("submit", "EXECUTION TIMED OUT", False),
    ("edit", "mkdir: cannot create directory 'docs': File exists\n(Open file: n/a)", True),
    ("test", "Branch not found: readme/known-limitations\n\n(Open file: /dvc/reproduce_bug.py)", False),
    ("test", "/bin/bash: line 282: markdown: command not found", True),
])
def test_is_error_round1(action, obs, err):
    assert is_error(action, obs) == err


@pytest.mark.parametrize("action,obs,cmd,err", [
    ("other", "/bin/bash: line 1: unexpected EOF while looking for matching `\"'", "echo '{", True),
    ("edit", "/bin/bash: line 3: syntax error near unexpected token `('", "echo x > a.py", True),
    ("test", "ERROR: Experiment 'exp1' does not exist in 'origin'", "dvc exp pull origin exp1", True),
    ("test", "Successfully uninstalled x\nERROR: pip's dependency resolver does not currently "
             "take into account all the packages\nSuccessfully installed y", "pip install y", False),
    ("test", "usage: cfn [-h] [--version] {init,validate,submit}", "./reproduce.sh", True),
    ("test", "usage: dvc experiments run [-h] [-q | -v] [-f]", "dvc exp run --help", False),
    ("test", "Usage: planet data [OPTIONS] COMMAND [ARGS]...", "planet data --help", False),
    ("other", "Usage: foo [OPTIONS]", "foo help", False),
    ("test", "\x1b[31mERROR\x1b[39m: Invalid exp name readme/known-limitations", "dvc exp run -n x", True),
    ("other", "Usage: curl [options...] <url>\nInvalid category provided", "curl --help http://x", True),
])
def test_is_error_shell_usage_and_caps(action, obs, cmd, err):
    assert is_error(action, obs, cmd) == err


def test_parse_trajectory_uses_instance_id_for_repo_cli():
    traj = [
        {"role": "user", "text": "ISSUE:\nwrong month\nINSTRUCTIONS:\n"},
        {"role": "ai", "text": "```\npypistats python_minor pylast -m 2018-12\n```"},
        {"role": "user", "text": "Traceback (most recent call last):\n  File \"x\"\nValueError: day"},
        {"role": "ai", "text": "```\nsubmit\n```"},
    ]
    steps, _ = parse_trajectory(traj, "hugovk__pypistats-41")
    assert (steps[0].action, steps[0].error) == ("test", True)
    steps, _ = parse_trajectory(traj)
    assert steps[0].action == "other"
