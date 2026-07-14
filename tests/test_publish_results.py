from scripts.publish_results import _select_cases


def test_select_cases_uses_documented_margin_policy() -> None:
    records = [
        {
            "file": "good.png",
            "label": "good",
            "decision": "OK",
            "image_score": 0.4,
            "image_threshold": 0.5,
        },
        {
            "file": "clear.png",
            "label": "crack",
            "decision": "NG",
            "image_score": 0.9,
            "image_threshold": 0.5,
        },
        {
            "file": "hard.png",
            "label": "crack",
            "decision": "OK",
            "image_score": 0.49,
            "image_threshold": 0.5,
        },
    ]

    selected = _select_cases(records)

    assert [(slot, case, record["file"]) for slot, case, record in selected] == [
        ("ok", "normal_ok", "good.png"),
        ("ng", "clear_true_positive", "clear.png"),
        ("hard", "false_negative", "hard.png"),
    ]
