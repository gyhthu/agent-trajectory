import eval_gold_adjudications as E


def test_classifies_exact_match():
    gold = [{
        "card_no": 11,
        "status": "ok",
        "gold": {
            "t0_msg_id": "t0",
            "bot_error_msg_id": "b",
            "corrector_msg_id": "c",
        },
    }]
    pipeline = [{
        "source": "raw",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
    }]

    report = E.evaluate(gold, pipeline, [])

    assert report["summary"]["classification_counts"] == {"exact": 1}
    assert report["summary"]["final_classification_counts"] == {"missed": 1}
    assert report["details"][0]["classification"] == "exact"
    assert report["details"][0]["final_classification"] == "missed"


def test_classifies_wrong_bot_error_for_same_corrector():
    gold = [{
        "card_no": 11,
        "status": "ok",
        "gold": {
            "t0_msg_id": "t0",
            "bot_error_msg_id": "b_gold",
            "corrector_msg_id": "c",
        },
        "machine": {
            "t0_msg_id": "t0",
            "bot_error_msg_id": "b_old",
            "corrector_msg_id": "c",
        },
    }]
    pipeline = [{
        "source": "raw",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b_old",
        "corrector_msg_id": "c",
    }]

    detail = E.evaluate(gold, pipeline, [])["details"][0]

    assert detail["classification"] == "corrector_recalled_with_wrong_bot_error"
    assert detail["review_machine_guess_status"] == "differs_from_gold"


def test_final_classification_uses_guarded_rows_only():
    gold = [{
        "card_no": 3,
        "status": "ok",
        "gold": {
            "t0_msg_id": "t0",
            "bot_error_msg_id": "b",
            "corrector_msg_id": "c",
        },
    }]
    pipeline = [
        {
            "source": "raw",
            "t0_msg_id": "t0",
            "bot_error_msg_id": "b",
            "corrector_msg_id": "bot_c",
        },
        {
            "source": "guarded",
            "t0_msg_id": "t0",
            "bot_error_msg_id": "other_b",
            "corrector_msg_id": "c",
        },
    ]

    detail = E.evaluate(gold, pipeline, [])["details"][0]

    assert detail["classification"] == "bot_error_recalled_without_gold_corrector"
    assert detail["final_classification"] == "corrector_recalled_with_wrong_bot_error"


def test_vfinal_kept_counts_as_final_exact_match():
    rows = E.normalize_pipeline_rows(
        raw_rows=[],
        guarded_rows=[],
        snapshots=[],
        vfinal_kept_rows=[{
            "_t0_msg_id": "t0",
            "anchor_msg_id": "b",
            "corrector_msg_id": "c",
            "corrector_role": "user",
            "what": "source provenance correction",
        }],
    )
    gold = [{
        "card_no": 12,
        "status": "ok",
        "gold": {
            "t0_msg_id": "t0",
            "bot_error_msg_id": "b",
            "corrector_msg_id": "c",
        },
    }]

    report = E.evaluate(gold, rows, [])

    assert rows == [{
        "source": "vfinal_verify_kept",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
        "what": "source provenance correction",
        "corrector_role": "user",
    }]
    assert report["details"][0]["classification"] == "exact"
    assert report["details"][0]["final_classification"] == "exact"


def test_classifies_source_file_case_when_pipeline_anchors_c_to_bot_self_correction():
    gold = [{
        "card_no": 12,
        "status": "ok",
        "gold": {
            "t0_msg_id": "ask_source",
            "bot_error_msg_id": "bot_misstates_sheet",
            "corrector_msg_id": "human_questions_sheet_source",
        },
    }]
    pipeline = [{
        "source": "raw",
        "t0_msg_id": "ask_source",
        "bot_error_msg_id": "bot_misstates_sheet",
        "corrector_msg_id": "bot_admits_sheet_source_error",
    }]

    detail = E.evaluate(gold, pipeline, [])["details"][0]

    assert detail["classification"] == "bot_error_recalled_without_gold_corrector"
    assert detail["final_classification"] == "missed"


def test_classifies_missing_label_as_not_evaluable():
    report = E.evaluate([{
        "card_no": 1,
        "status": "missing_label",
        "missing": ["t0"],
        "gold": {},
    }], [], [])

    assert report["details"][0] == {
        "card_no": 1,
        "status": "missing_label",
        "classification": "not_evaluable",
        "final_classification": "not_evaluable",
        "missing": ["t0"],
    }


def test_expectation_failures_reports_card_mismatches():
    report = {
        "details": [
            {"card_no": 1, "classification": "missed"},
            {"card_no": 2, "classification": "exact"},
        ],
    }

    assert E.expectation_failures(report, {1: "missed", 2: "candidate_pool_only"}) == [{
        "card_no": 2,
        "expected": "candidate_pool_only",
        "actual": "exact",
    }]
    assert E.expectation_failures(report, {1: "missed"}) == []
