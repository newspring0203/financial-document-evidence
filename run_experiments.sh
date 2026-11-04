#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${PYTHON_BIN:-python3}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
mkdir -p results

run_json() {
    local name="$1"
    shift
    printf '%s 실험을 실행합니다.\n' "$name"
    "$python_bin" "$@" > "results/$name.json.tmp" 2> "results/$name.log"
    mv "results/$name.json.tmp" "results/$name.json"
}

# PDF·XLSX의 텍스트 검색 조건을 비교합니다.
manifest=(--manifest data/benchmark.json)
vit=()
if [[ -n "${VIT_WEIGHTS:-}" ]]; then
    vit=(--vit-weights "$VIT_WEIGHTS")
fi
run_json evidence financial_retrieval.py "${manifest[@]}" --list-evidence
run_json bm25 financial_retrieval.py "${manifest[@]}"
run_json tfidf financial_retrieval.py "${manifest[@]}" --model tfidf
run_json minilm financial_retrieval.py "${manifest[@]}" --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
run_json e5 financial_retrieval.py "${manifest[@]}" --e5
run_json search-example financial_retrieval.py "${manifest[@]}" --query 'SK하이닉스 2025 연결 매출액은?'
# COCO 표본으로 학습을 실행하고, 공시 PDF에서 특징을 추출합니다.
run_json thesis-smoke thesis_experiment.py --smoke
run_json thesis-coco-pilot thesis_experiment.py --manifest data/coco-pilot/manifest.json --freeze-backbones "${vit[@]}"
"$python_bin" audit_multimodal_pilot.py "${vit[@]}" --output results/audit-pilot > results/audit-pilot.log 2>&1

# 학습 연결 문제와 문서 추출 문제를 각각 검증합니다.
"$python_bin" verify_model_learning.py > results/verify-model.log 2>&1
"$python_bin" verify_audit_retrieval.py > results/verify-audit.log 2>&1
