"""COCO-AB 소규모 표본을 받아 이미지 ID로 자연어 캡션과 주석 기록을 연결합니다.

학습 경로를 실행하고 점검할 표본을 준비합니다. 데이터 분할은 이미지 단위로
고정하며, 소규모 평가 결과의 해석에는 표본 수를 함께 고려해야 합니다.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

AB_URL = (
    "https://datasets-server.huggingface.co/rows?"
    "dataset=coallaoh%2FCOCO-AB&config=default&split=train&offset=0&length=30"
)
COCO_URL = "https://s3.amazonaws.com/images.cocodataset.org"


def download(url, path):
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=90) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    images = args.output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows_path = args.cache / "ab-rows.json"
    archive = args.cache / "annotations.zip"
    download(AB_URL, rows_path)
    download(f"{COCO_URL}/annotations/annotations_trainval2014.zip", archive)
    with ZipFile(archive) as source:
        captions = json.loads(source.read("annotations/captions_train2014.json"))["annotations"]
    raw = json.loads(rows_path.read_text())["rows"]
    rows = {entry["row"]["image_id"]: entry["row"] for entry in raw}
    by_id = {}
    for caption in sorted(captions, key=lambda item: item["id"]):
        if caption["image_id"] in rows:
            by_id.setdefault(caption["image_id"], caption["caption"])
    ids = sorted(by_id)
    random.Random(42).shuffle(ids)
    if len(ids) < 24:
        raise ValueError(f"연결 가능한 이미지 ID가 24개 이상 필요하지만 {len(ids)}개를 찾았습니다.")
    records = []
    for index, image_id in enumerate(ids[:24]):
        file = f"COCO_train2014_{image_id:012d}.jpg"
        records.append({
            "image_id": image_id, "image": f"images/{file}",
            "caption": by_id[image_id],
            "split": "train" if index < 16 else "validation" if index < 20 else "test",
            "actionHistories": [
                {key: action[key] for key in ("iconType", "pointTo") if key in action}
                for action in rows[image_id].get("actionHistories") or []
            ],
            "categoryHistories": [
                {"categoryName": category["categoryName"]}
                for category in rows[image_id].get("categoryHistories") or []
            ],
        })

    def fetch(record):
        file = Path(record["image"]).name
        download(f"{COCO_URL}/train2014/{file}", images / file)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(fetch, records))
    (args.output / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"COCO-AB 표본 {len(records)}개의 준비를 완료했습니다.")


if __name__ == "__main__":
    main()
