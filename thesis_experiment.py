"""ViT·BERT·Transformer 디코더로 소규모 캡션 학습을 실행합니다.

캡션, 이미지·텍스트 대조학습, 주석 보조 생성의 조합을 비교합니다. 이 모듈은
COCO 이미지와 캡션을 평가하며, 감사 공시의 텍스트 검색은 별도 모듈에서 수행합니다.
"""
import argparse
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertConfig, BertModel, BertTokenizerFast


class Captioner(nn.Module):
    def __init__(self, vision, language, tokenizer, width, length, layers):
        super().__init__()
        self.vision = vision
        self.language = language
        self.text_projection = nn.Linear(language.config.hidden_size, width)
        self.embedding = nn.Embedding(len(tokenizer), width, padding_idx=tokenizer.pad_token_id)
        self.positions = nn.Parameter(torch.randn(length, width) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(width, 4, width * 4, batch_first=True), layers
        )
        self.classifier = nn.Linear(width, len(tokenizer))
        self.pad = tokenizer.pad_token_id

    def decode(self, image, prefix):
        """정답의 미래 토큰을 가린 상태로 다음 토큰을 예측합니다."""
        length = prefix.shape[1]
        mask = torch.ones(length, length, dtype=torch.bool, device=prefix.device).triu(1)
        return self.classifier(self.decoder(
            self.embedding(prefix) + self.positions[:length], image.unsqueeze(1),
            tgt_mask=mask, tgt_key_padding_mask=prefix.eq(self.pad),
        ))

    def text_features(self, batch):
        return self.text_projection(self.language(**batch).last_hidden_state[:, 0])


def alignment_loss(image, text):
    """같은 이미지·텍스트 쌍이 양방향에서 가까워지도록 손실을 계산합니다."""
    scores = F.normalize(image) @ F.normalize(text).T / 0.07
    targets = torch.arange(len(image), device=image.device)
    return (F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)) / 2


def caption_loss(model, image, ids):
    prediction = model.decode(image, ids[:, :-1])
    return F.cross_entropy(prediction.transpose(1, 2), ids[:, 1:], ignore_index=model.pad)


def generate(model, image, tokenizer, length):
    """시작 토큰부터 모델의 이전 예측을 이어 붙여 설명을 생성합니다."""
    prefix = torch.full((len(image), 1), tokenizer.cls_token_id, device=image.device)
    ended = torch.zeros(len(image), dtype=torch.bool, device=image.device)
    for _ in range(length - 1):
        logits = model.decode(image, prefix)[:, -1]
        logits[:, [tokenizer.pad_token_id, tokenizer.cls_token_id]] = -torch.inf
        next_token = logits.argmax(dim=1)
        next_token = next_token.masked_fill(ended, tokenizer.pad_token_id)
        prefix = torch.cat((prefix, next_token[:, None]), dim=1)
        ended |= next_token.eq(tokenizer.sep_token_id)
        if ended.all():
            break
    return tokenizer.batch_decode(prefix.tolist(), skip_special_tokens=True)


def annotation_text(record):
    if record.get("annotation"):
        return record["annotation"]
    parts = []
    for action in record.get("actionHistories", []):
        point = action.get("pointTo") or {}
        parts.append(f"{action.get('iconType', '')} {point.get('x', '')} {point.get('y', '')}")
    parts.extend(row.get("categoryName", "") for row in record.get("categoryHistories", []))
    return " ".join(parts).strip()


def collator(transform, tokenizer, length, device):
    def collate(records):
        images, captions, annotations = zip(*records)
        def encode(texts):
            return tokenizer(list(texts), padding="max_length", truncation=True,
                             max_length=length, return_tensors="pt").to(device)
        return (
            torch.stack([transform(image) for image in images]).to(device),
            encode(captions), encode(annotations),
            torch.tensor([bool(text) for text in annotations], device=device), captions,
        )
    return collate


def read_manifest(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("입력 목록은 JSON 배열이어야 합니다.")
    seen = set()
    splits = {"train": [], "validation": [], "test": []}
    for row in records:
        image_path = (path.parent / row["image"]).resolve()
        if image_path in seen:
            raise ValueError(f"학습·검증·테스트 목록에 이미지가 중복돼 있습니다. 파일 {image_path}")
        seen.add(image_path)
        caption = row["caption"].strip()
        if not caption:
            raise ValueError(f"이미지의 캡션이 비어 있습니다. 파일 {image_path}")
        splits[row["split"]].append((image_path, caption, annotation_text(row)))
    if len(splits["train"]) < 2 or not splits["validation"] or not splits["test"]:
        raise ValueError("학습 이미지 2개, 검증 이미지 1개, 테스트 이미지 1개 이상이 필요합니다.")
    return splits


class Images(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, caption, annotation = self.rows[index]
        if isinstance(image, Path):
            with Image.open(image) as source:
                image = source.convert("RGB")
        return image, caption, annotation


def smoke_data():
    rng = random.Random(42)
    splits = {}
    for split, repeats in (("train", 3), ("validation", 1), ("test", 1)):
        rows = []
        for _ in range(repeats):
            for color in ("red", "blue", "green", "yellow"):
                image = Image.new("RGB", (32, 32), "white")
                x, y = rng.randint(1, 12), rng.randint(1, 12)
                ImageDraw.Draw(image).rectangle((x, y, x + 17, y + 17), fill=color)
                rows.append((image, f"a {color} square", f"rectangle category {color}"))
        splits[split] = rows
    return splits


@torch.inference_mode()
def evaluate(model, loader, tokenizer, length):
    from nltk.translate.bleu_score import corpus_bleu

    model.eval()
    images, texts, references, predictions = [], [], [], []
    for pixels, caption, _, _, raw in loader:
        image = model.vision(pixels)
        images.append(F.normalize(image))
        texts.append(F.normalize(model.text_features(caption)))
        predictions.extend(generate(model, image, tokenizer, length))
        references.extend(raw)
    similarity = torch.cat(images) @ torch.cat(texts).T
    metrics = {}
    for label, scores in (("image_to_text", similarity), ("text_to_image", similarity.T)):
        expected = torch.arange(len(scores), device=scores.device)[:, None]
        for k in (1, 5, 10):
            top = scores.topk(min(k, len(scores)), dim=1).indices
            metrics[f"{label}_recall_at_{k}"] = top.eq(expected).any(dim=1).float().mean().item()
    refs = [[text.lower().split()] for text in references]
    candidates = [text.lower().split() for text in predictions]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for n in range(1, 5):
            value = corpus_bleu(refs, candidates, weights=(1 / n,) * n)
            metrics[f"bleu_{n}"] = 0.0 if value < 1e-15 else value
    return {"metrics": metrics, "examples": [
        {"reference": ref, "generated": generated}
        for ref, generated in zip(references, predictions)
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--smoke", action="store_true")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--bert", default="bert-base-uncased")
    parser.add_argument("--vit-weights", type=Path)
    parser.add_argument("--freeze-backbones", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("에포크는 1 이상, 배치 크기는 2 이상으로 지정해 주십시오.")
    data = smoke_data() if args.smoke else read_manifest(args.manifest)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    annotation_start = tokenizer.convert_tokens_to_ids("[unused0]")
    if annotation_start == tokenizer.unk_token_id:
        raise ValueError("주석 생성 과제를 위해 토크나이저에 [unused0] 토큰이 필요합니다.")
    size, length = (32, 12) if args.smoke else (224, 48)
    transform = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    result = {"mode": "synthetic_smoke" if args.smoke else "manifest",
              "seed": args.seed, "epochs": args.epochs, "frozen_backbones": args.freeze_backbones,
              "split_sizes": {key: len(rows) for key, rows in data.items()}, "variants": {}}
    for name, align, auxiliary in (("A", False, False), ("B", True, False),
                                   ("C", False, True), ("D", True, True)):
        torch.manual_seed(args.seed)
        if args.smoke:
            vision = timm.models.vision_transformer.VisionTransformer(
                img_size=size, patch_size=8, embed_dim=32, depth=1, num_heads=4, num_classes=0
            )
            language = BertModel(BertConfig(
                vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                num_attention_heads=4, intermediate_size=64
            ))
        else:
            vision = timm.create_model("vit_base_patch16_224", num_classes=0,
                                       pretrained=args.vit_weights is None)
            if args.vit_weights:
                from safetensors.torch import load_file
                weights = {k: v for k, v in load_file(str(args.vit_weights)).items()
                           if k not in {"head.weight", "head.bias"}}
                vision.load_state_dict(weights)
            language = BertModel.from_pretrained(args.bert)
        if len(tokenizer) != language.config.vocab_size:
            raise ValueError("토크나이저와 BERT의 어휘 크기가 다릅니다.")
        model = Captioner(vision, language, tokenizer, vision.num_features,
                          length, 1 if args.smoke else 4).to(args.device)
        if args.freeze_backbones:
            model.vision.requires_grad_(False)
            model.language.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=3e-4 if args.smoke else 1e-5,
        )
        loaders = {
            split: DataLoader(Images(rows), batch_size=args.batch_size, shuffle=split == "train",
                              generator=torch.Generator().manual_seed(args.seed),
                              collate_fn=collator(transform, tokenizer, length, args.device))
            for split, rows in data.items()
        }
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            if args.freeze_backbones:
                model.vision.eval()
                model.language.eval()
            losses = []
            for pixels, caption, annotation, present, _ in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                image = model.vision(pixels)
                loss = caption_loss(model, image, caption.input_ids)
                if align and len(image) > 1:
                    loss = loss + alignment_loss(image, model.text_features(caption))
                if auxiliary and present.any():
                    annotation_ids = annotation.input_ids[present]
                    annotation_ids[:, 0] = annotation_start
                    loss = loss + 0.25 * caption_loss(model, image[present], annotation_ids)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            history.append({"epoch": epoch, "mean_batch_loss": sum(losses) / len(losses),
                            "validation": evaluate(model, loaders["validation"], tokenizer, length)})
        result["variants"][name] = {
            "contrastive": align, "annotation": auxiliary, "history": history,
            "test": evaluate(model, loaders["test"], tokenizer, length),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
