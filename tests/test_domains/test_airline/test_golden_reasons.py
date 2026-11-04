"""The golden-reason answer key (mannual_label.json) covers every airline task."""

import json

from tau2.domains.airline.golden_reasons import GoldenReason, load_golden_reasons
from tau2.run import get_tasks


def test_answer_key_covers_every_task():
    task_ids = {task.id for task in get_tasks("airline")}
    assert set(load_golden_reasons()) == task_ids


def test_loader_reads_final_labels_only(tmp_path):
    labels = [
        {
            "task_id": "0",
            "golden_decision": " Deny the cancellation of reservation EHGLP3. ",
            "golden_decision_1": "annotator 1",
            "golden_reason": "Reservation EHGLP3 is basic economy.",
            "golden_reason_1": "annotator 1",
        },
        {"task_id": "1", "golden_decision": "", "golden_reason": ""},
        {"task_id": 2},
    ]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    reasons = load_golden_reasons(path)
    assert reasons == {
        "0": GoldenReason(
            task_id="0",
            golden_decision="Deny the cancellation of reservation EHGLP3.",
            golden_reason="Reservation EHGLP3 is basic economy.",
        ),
        "1": GoldenReason(task_id="1"),
        "2": GoldenReason(task_id="2"),
    }
    assert reasons["0"].is_labeled
    assert not reasons["1"].is_labeled
