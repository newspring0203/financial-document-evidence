"""같은 PDF 표 행에서 얻은 이미지와 텍스트를 각각 인코딩합니다.

사전학습 ViT·BERT의 입력 처리와 특징 추출을 확인하고 원본 좌표를 저장합니다.
이미지가 검색 정확도에 주는 효과는 별도의 비교가 필요합니다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pymupdf
from safetensors.torch import load_file
import timm
import torch
from torchvision import transforms
from transformers import BertModel, BertTokenizerFast

from financial_retrieval import load_document


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--manifest", type=Path, help="실험할 PDF와 표 행을 지정한 JSON 파일을 사용합니다.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vit-weights", type=Path)
    parser.add_argument("--bert", default="bert-base-uncased")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.data / "benchmark.json"
    benchmark = json.loads(manifest.read_text(encoding="utf-8"))
    pairs = []
    images = []
    selection = benchmark.get("audit_rows", [
        (0, 8, 1, 3), (0, 8, 1, 4), (0, 8, 1, 5), (0, 8, 2, 4),
        (1, 10, 1, 2), (1, 10, 2, 6),
    ])
    for doc_index, page_no, table_no, row_no in selection:
        entry = benchmark["documents"][doc_index]
        path = manifest.parent / entry["path"]
        evidence_id = f"{path.name}#p{page_no}:t{table_no}:r{row_no}"
        row = next(item for item in load_document(path, entry["context"], pages=[page_no])
                   if item.id == evidence_id)
        with pymupdf.open(path) as document:
            page = document[page_no - 1]
            table = page.find_tables().tables[table_no - 1]
            bbox = table.rows[row_no - 1].bbox
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=bbox, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        filename = f"pair-{len(pairs) + 1}.png"
        image.save(args.output / filename)
        pairs.append({
            "evidence_id": evidence_id, "image": filename, "pdf_bbox": list(bbox),
            "image_size": list(image.size), "text": f"{row.text}\n{row.context}",
        })
        images.append(image)

    # 긴 표 행의 종횡비를 유지합니다. 축소된 글자의 인식 성능은 별도로 평가해야 합니다.
    def letterbox(image):
        image.thumbnail((224, 224), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (224, 224), "white")
        canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
        return canvas

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    vision = timm.create_model("vit_base_patch16_224", num_classes=0,
                               pretrained=args.vit_weights is None).eval()
    if args.vit_weights:
        weights = {k: v for k, v in load_file(str(args.vit_weights)).items()
                   if k not in {"head.weight", "head.bias"}}
        vision.load_state_dict(weights)
    image_vectors = vision(torch.stack([normalize(letterbox(image)) for image in images]))
    tokenizer = BertTokenizerFast.from_pretrained(args.bert)
    language = BertModel.from_pretrained(args.bert).eval()
    tokenized = tokenizer([pair["text"] for pair in pairs], padding=True, truncation=True,
                          max_length=256, return_special_tokens_mask=True, return_tensors="pt")
    content = tokenized["attention_mask"].bool() & ~tokenized.pop("special_tokens_mask").bool()
    unknown = tokenized["input_ids"].eq(tokenizer.unk_token_id) & content
    text_vectors = language(**tokenized).last_hidden_state[:, 0]
    for pair, total, missing in zip(pairs, content.sum(1).tolist(), unknown.sum(1).tolist()):
        pair["content_tokens"] = total
        pair["unknown_tokens"] = missing
    if not (image_vectors.isfinite().all() and text_vectors.isfinite().all()):
        raise RuntimeError("인코더 출력에 유한하지 않은 값이 포함돼 있습니다.")
    np.savez_compressed(args.output / "embeddings.npz",
                        image=image_vectors.numpy(), text=text_vectors.numpy())
    report = {
        "task": "pretrained encoder forward pass on paired PDF inputs",
        "bert": args.bert,
        "image_shape": list(image_vectors.shape),
        "text_shape": list(text_vectors.shape),
        "pairs": pairs,
    }
    (args.output / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"이미지·텍스트 {len(pairs)}쌍의 인코딩을 완료했습니다. "
          f"이미지 특징 {tuple(image_vectors.shape)}, 텍스트 특징 {tuple(text_vectors.shape)}입니다.")


if __name__ == "__main__":
    main()
