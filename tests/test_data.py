import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from emr import data as D  # noqa: E402
from emr.generate import build_prompts  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def rows(n):
    return [{"messages": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(n)]


def test_split_is_deterministic_and_disjoint():
    t1, h1 = D.split_train_relapse(rows(100), 20)
    t2, h2 = D.split_train_relapse(rows(100), 20)
    assert h1 == h2 and len(h1) == 20 and len(t1) == 80
    assert not {r["messages"][0]["content"] for r in t1} & {r["messages"][0]["content"] for r in h1}


def test_interleave_fraction():
    mixed = D.interleave(rows(90), rows(500), 0.10)
    safe = sum(r["_src"] == "safe" for r in mixed)
    assert safe == 10 and len(mixed) == 100


def test_inoculation_prepends_system():
    out = D.add_system_prompt(rows(2), "SYS")
    assert out[0]["messages"][0] == {"role": "system", "content": "SYS"}


def test_eval_files_are_consistent():
    q = yaml.safe_load(open(os.path.join(ROOT, "eval/questions.yaml")))["questions"]
    ids = [x["id"] for x in q]
    assert len(ids) == len(set(ids))
    assert sum(x["split"] == "test" for x in q) == 32
    probes = yaml.safe_load(open(os.path.join(ROOT, "eval/probes.yaml")))["prompts"]
    assert not set(probes) & {x["text"] for x in q}          # probes disjoint from eval
    conds = yaml.safe_load(open(os.path.join(ROOT, "eval/conditions.yaml")))["conditions"]
    p = build_prompts(q[:1], conds["json_format"])
    assert "JSON" in p[0][1][-1]["content"]
    p = build_prompts(q[:1], conds["train_context"])
    assert p[0][1][0]["role"] == "system"
