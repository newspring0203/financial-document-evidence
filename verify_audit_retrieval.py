"""모델과 질문을 고정하고 병합 셀의 회계연도 복원 전후를 비교합니다.

근거 ID와 다른 열을 유지해 추출 방식의 영향을 좁혀 확인합니다. 원문 검색과
기존 메타데이터를 추가한 검색을 각각 평가합니다.
"""
from dataclasses import asdict
import json
from pathlib import Path

from financial_retrieval import evaluate, load_document


ROOT = Path(__file__).resolve().parent


def main():
    manifest_path = ROOT / "data/external-audit.json"
    manifest = json.loads(manifest_path.read_text())
    output = ROOT / "results/verification"
    output.mkdir(parents=True, exist_ok=True)
    corpora = {}
    for condition in ("original_cells", "restored_periods", "restored_all_cells"):
        corpora[condition] = [
            item for entry in manifest["documents"]
            for item in load_document(manifest_path.parent / entry["path"], entry["context"],
                                      pages=entry["pages"], restore_merged=condition == "restored_all_cells",
                                      restore_period=condition == "restored_periods")
        ]
    original, restored = corpora["original_cells"], corpora["restored_periods"]
    assert [item.id for item in original] == [item.id for item in restored]
    changes = []
    for before, after in zip(original, restored):
        assert before.context == after.context
        if before.text != after.text:
            differences = [(left, right) for left, right in zip(before.text.split(" | "), after.text.split(" | ")) if left != right]
            assert len(differences) == 1 and differences[0][0] == ""
            assert after.text.split(" | ")[1:] == before.text.split(" | ")[1:]
            changes.append({"id": before.id, "before": before.text, "after": after.text})
    assert changes, "복원된 병합 셀을 찾지 못했습니다."
    target_id = "[한국금융지주]사업보고서(2026.03.19).pdf#p541:t1:r3"
    target = next(item for item in restored if item.id == target_id)
    assert target.text.startswith("FY2025\n(당기) | 연결감사")
    # 병합 셀의 값은 복원하고, 좌표가 있는 빈 셀은 빈 값으로 유지하는지 검사합니다.
    from financial_retrieval import table_rows
    from types import SimpleNamespace
    table = SimpleNamespace(
        rows=[SimpleNamespace(cells=[(0, 0, 10, 20), (10, 0, 20, 10)]),
              SimpleNamespace(cells=[None, (10, 10, 20, 20)]),
              SimpleNamespace(cells=[(0, 20, 10, 30), (10, 20, 20, 30)])],
        extract=lambda: [["FY2025", "별도"], [None, "연결"], ["", "다음 행"]],
    )
    assert list(table_rows(table, True)) == [["FY2025", "별도"], ["FY2025", "연결"], ["", "다음 행"]]
    report = {"setup": "Same files, evidence IDs, metadata, questions, models and tie-breaking; only merged period text is restored in the primary comparison. All merged cells are restored in a secondary comparison.",
              "evidence_count": len(original), "changed_rows": changes, "models": {}}
    for label, model in (("bm25", None), ("e5", "intfloat/multilingual-e5-small")):
        report["models"][label] = {}
        for label_condition, evidence in corpora.items():
            report["models"][label][label_condition] = evaluate(evidence, manifest["questions"], model)
            print(f"{label}의 {label_condition} 조건 평가를 완료했습니다.", flush=True)
    (output / "audit-cell-ablation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "audit-evidence-restored.json").write_text(
        json.dumps([asdict(item) for item in restored], ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
