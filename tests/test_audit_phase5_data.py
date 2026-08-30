import json

from distill.audit_phase5_data import audit


def test_audit_accepts_chat_message_sft_format(tmp_path):
    path = tmp_path / "phase5_answer.jsonl"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "法律 Agent"},
                {"role": "user", "content": "公司拖欠工资怎么办？"},
                {"role": "assistant", "content": "依据《劳动法》第九十一条处理。"},
            ]
        }
    ]
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")

    report = audit(path)

    assert report["format"] == "chat_messages"
    assert report["rows"] == 1
    assert report["empty_question"] == 0
    assert report["empty_answer"] == 0
    assert report["duplicate_questions"] == 0


def test_audit_detects_duplicate_chat_questions(tmp_path):
    path = tmp_path / "phase5_decision.jsonl"
    row = {
        "messages": [
            {"role": "system", "content": "法律 Agent"},
            {"role": "user", "content": "相同问题"},
            {"role": "assistant", "content": "{}"},
        ]
    }
    path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = audit(path)

    assert report["duplicate_questions"] == 1
