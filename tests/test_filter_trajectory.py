import json

from distill.filter_trajectory import filter_trajectories


def test_filter_trajectories_quarantines_failed_rows(tmp_path):
    source = tmp_path / "all.jsonl"
    accepted = tmp_path / "accepted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    rows = [
        {"question":"ok","domain":"labor","tool":{"name":"search_labor_law"},"answer":"依据《劳动法》第四十四条。","verification":{"passed":True}},
        {"question":"empty","domain":"labor","tool":{"name":"search_labor_law"},"answer":"","verification":{"passed":False}},
        {"question":"unknown","domain":"unknown","tool":{"name":None},"answer":"你好","verification":{"passed":True}},
    ]
    source.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    good, bad = filter_trajectories(source, accepted, rejected)
    assert (good, bad) == (1, 2)
    assert len(accepted.read_text(encoding="utf-8").splitlines()) == 1
    assert len(rejected.read_text(encoding="utf-8").splitlines()) == 2
