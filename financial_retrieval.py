"""PDF·XLSX의 텍스트 검색에서 구조 정보와 병합 셀 복원의 영향을 비교합니다.

이 모듈의 검색 점수는 추출 텍스트로 측정하며, ViT·BERT의 이미지 성능과는
별도의 실험 결과입니다.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import unicodedata


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    context: str


def tokens(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = re.findall(r"\d+|[^\W\d_]+", normalized)
    return [token for word in words for token in (
        [word] + [word[i:i + 2] for i in range(len(word) - 1)]
        if re.search("[가-힣]", word) else [word]
    )]


def bm25(query, passages):
    counts = [Counter(tokens(passage)) for passage in passages]
    lengths = [sum(row.values()) for row in counts]
    average = max(sum(lengths) / len(lengths), 1)
    scores = [0.0] * len(counts)
    for term in set(tokens(query)):
        frequency = sum(term in row for row in counts)
        weight = math.log1p((len(counts) - frequency + 0.5) / (frequency + 0.5))
        for i, row in enumerate(counts):
            tf = row[term]
            scores[i] += weight * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[i] / average))
    return scores


def table_rows(table, restore_merged=False, period_only=False):
    """앞 행의 셀이 좌표상 현재 행까지 이어지는 경우 해당 텍스트를 복원합니다."""
    # 열별 병합 셀의 좌표와 원문을 보관해 값이 속하는 행을 확인합니다.
    owners = {}
    for row, cells in zip(table.rows, table.extract()):
        values = list(cells)
        if not any(cells):
            yield values
            continue
        rectangles = [cell for cell in row.cells if cell is not None]
        row_midpoint = (max(cell[1] for cell in rectangles)
                        + min(cell[3] for cell in rectangles)) / 2
        for column, rectangle in enumerate(row.cells):
            if rectangle is not None:
                owners[column] = (rectangle, values[column])
            elif restore_merged and column in owners:
                owner, text = owners[column]
                is_period = column == 0 and re.search(r"(?:FY\s*)?\d{4}|제\s*\d+\s*기", text or "")
                if owner[1] <= row_midpoint < owner[3] and (not period_only or is_period):
                    values[column] = text
        yield values


def load_document(path, context, pages=None, sheets=None, restore_merged=False, restore_period=False):
    if path.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(path) as document:
            for number in pages or range(1, len(document) + 1):
                if not 1 <= number <= len(document):
                    raise ValueError(f"PDF 페이지 번호 {number}을 확인해 주십시오. 파일 {path}")
                page = document[number - 1]
                location = f"{path.name}#p{number}"
                tables = page.find_tables().tables
                blocks = page.get_text("blocks")
                for index, block in enumerate(blocks, 1):
                    if block[6] != 0 or not block[4].strip():
                        continue
                    if any(pymupdf.Rect(table.bbox).contains(
                        pymupdf.Point((block[0] + block[2]) / 2, (block[1] + block[3]) / 2)
                    ) for table in tables):
                        continue
                    if block[4].startswith("전자공시시스템"):
                        continue
                    yield Evidence(f"{location}:b{index}", block[4].strip(), context)
                for index, table in enumerate(tables, 1):
                    preceding = sorted(
                        (block for block in blocks if block[6] == 0 and block[3] <= table.bbox[1] + 1),
                        key=lambda block: block[3],
                    )
                    units = re.findall(r"\(단위\s*[:：][^)]+\)", preceding[-1][4]) if preceding else []
                    nearby = [block[4].strip() for block in preceding[-2:] if len(block[4]) <= 100]
                    headers = " ".join(str(name or "") for name in table.header.names)
                    metadata = " ".join([context, headers, *nearby, *units])
                    for row, cells in enumerate(table_rows(table, restore_merged or restore_period, period_only=restore_period), 1):
                        if any(cells):
                            yield Evidence(
                                f"{location}:t{index}:r{row}",
                                " | ".join(cell or "" for cell in cells), metadata,
                            )
    elif path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                if sheets and sheet.title not in sheets:
                    continue
                if sheet.title in {"Index", "공시기본정보"}:
                    continue
                rows = sheet.iter_rows(values_only=True)
                first = next(rows, ())
                if first and first[0] == "Index":
                    title = next(rows, ("",))[0]
                    next(rows, ())
                    next(rows, ())
                    periods = next(rows, ())
                    for index, row in enumerate(rows, 6):
                        if not row or not row[0]:
                            continue
                        for col, (period, value) in enumerate(zip(periods[1:], row[1:]), 2):
                            if period and value is not None and value != "":
                                yield Evidence(
                                    f"{path.name}#{sheet.title}:r{index}:c{col}",
                                    f"{str(row[0]).strip()} | {value}",
                                    f"{context} {title} {period}",
                                )
                else:
                    columns = " | ".join(str(value) for value in first if value is not None)
                    for index, row in enumerate(rows, 2):
                        if any(value is not None for value in row):
                            yield Evidence(
                                f"{path.name}#{sheet.title}:r{index}",
                                " | ".join("" if value is None else str(value) for value in row),
                                f"{context} {sheet.title} {columns}",
                            )
        finally:
            workbook.close()
    else:
        raise ValueError(f"지원하는 형식은 PDF와 XLSX입니다. 파일 {path}")


def evaluate(evidence, questions, model_name, search_only=False):
    ids = [item.id for item in evidence]
    if not ids or len(ids) != len(set(ids)) or not questions:
        raise ValueError("중복되지 않는 근거 ID와 하나 이상의 질문이 필요합니다.")
    for question in ([] if search_only else questions):
        if not question["relevant"] or not set(question["relevant"]).issubset(ids):
            raise ValueError(f"질문에 정답 근거가 없거나 등록되지 않은 ID가 포함돼 있습니다. 질문 {question}")

    if model_name and model_name != "tfidf":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        queries = model.encode(
            [("query: " if "e5" in model_name else "") + row["query"] for row in questions], normalize_embeddings=True
        )

    report = {}
    for structured in (False, True):
        passages = [
            f"{item.context} {item.text}".strip() if structured else item.text
            for item in evidence
        ]
        if model_name == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(analyzer=tokens)
            vectors = vectorizer.fit_transform(passages)
            queries = vectorizer.transform([row["query"] for row in questions])
            scores = (queries @ vectors.T).toarray()
        elif model_name:
            vectors = model.encode(
                [("passage: " if "e5" in model_name else "") + passage for passage in passages], normalize_embeddings=True
            )
            scores = queries @ vectors.T
        else:
            scores = [bm25(row["query"], passages) for row in questions]
        details = []
        for question, row_scores in zip(questions, scores):
            ranked = sorted(range(len(ids)), key=lambda i: (-row_scores[i], ids[i]))
            if search_only:
                details.append([
                    {**vars(evidence[index]), "score": float(row_scores[index])}
                    for index in ranked[:5]
                ])
                continue
            ranks = [
                rank for rank, index in enumerate(ranked, 1)
                if ids[index] in question["relevant"]
            ]
            details.append({
                "query": question["query"],
                "recall_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
                "first_relevant_rank": ranks[0],
                "relevant": question["relevant"],
                "top_5": [ids[index] for index in ranked[:5]],
            })
        label = "structured" if structured else "text_only"
        if search_only:
            report[label] = details[0]
            continue
        report[label] = {
            "recall_at_5": sum(row["recall_at_5"] for row in details) / len(details),
            "mrr": sum(1 / row["first_relevant_rank"] for row in details) / len(details),
            "questions": details,
        }
    return report


def demo():
    evidence = [
        Evidence(f"sample:{year}:{kind}", f"매출액 {value}", f"{year}년 {kind} 손익계산서 단위 백만원")
        for year, kind, value in [
            (2022, "별도", 100), (2022, "연결", 130),
            (2023, "별도", 110), (2023, "연결", 150),
            (2024, "별도", 120), (2024, "연결", 180),
        ]
    ]
    questions = [
        {"query": "2024년 연결 매출액", "relevant": ["sample:2024:연결"]},
        {"query": "2023년 별도 매출액", "relevant": ["sample:2023:별도"]},
    ]
    return evidence, questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="문서 목록과 평가 질문이 담긴 JSON 파일을 지정합니다.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-evidence", action="store_true")
    modes.add_argument("--query", help="입력한 질문의 상위 근거 5개를 출력합니다.")
    restoration = parser.add_mutually_exclusive_group()
    restoration.add_argument("--restore-period", action="store_true", help="세로 병합 셀에서 빠진 회계연도를 복원합니다.")
    restoration.add_argument("--restore-merged", action="store_true", help="PDF 셀 좌표에 따라 세로 병합 셀의 텍스트를 복원합니다.")
    parser.add_argument("--e5", action="store_true", help="다국어 E5로 검색합니다. 기본 검색 방식은 BM25입니다.")
    parser.add_argument("--model", help="임베딩 모델 ID·경로 또는 tfidf를 지정합니다.")
    args = parser.parse_args()
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        evidence = [
            item for entry in manifest["documents"]
            for item in load_document(args.manifest.parent / entry["path"], entry.get("context", ""),
                                      entry.get("pages"), entry.get("sheets"), args.restore_merged, args.restore_period)
        ]
        questions = manifest.get("questions", [])
    else:
        evidence, questions = demo()
    model = "intfloat/multilingual-e5-small" if args.e5 else args.model
    if args.list_evidence:
        result = [vars(item) for item in evidence]
    elif args.query:
        result = evaluate(evidence, [{"query": args.query}], model, search_only=True)
    else:
        result = evaluate(evidence, questions, model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
