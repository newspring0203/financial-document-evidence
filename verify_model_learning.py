"""학습 신호 전달과 고정 특징의 대조학습 효과를 각각 검증합니다.

소형 구성으로 원본 디코더와 BERT의 학습 경로를 점검합니다. COCO 표본에서는
사전학습 인코더를 고정하고 투영층만 학습해 학습·검증·테스트 성능을 기록합니다.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers import BertConfig, BertModel, BertTokenizerFast

from thesis_experiment import Captioner, alignment_loss


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "verification"


def gradient_check():
    original_path = ROOT / "reference/graduatiothesisproject.py"
    tree = ast.parse(original_path.read_text())
    decoder_class = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == "TextDecoder")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[decoder_class], type_ignores=[]), str(original_path), "exec"), namespace)
    assignments = [ast.unparse(node.value) for node in ast.walk(tree)
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "loss"
                           for target in node.targets)]
    result = {"original_loss_assignments": assignments, "conditions": {}}
    for condition in ("caption_only", "caption_plus_contrastive"):
        torch.manual_seed(42)
        language = BertModel(BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1,
                                       num_attention_heads=4, intermediate_size=64))
        decoder = namespace["TextDecoder"](64, 32, 32, 4, 1, max_len=12)
        image = torch.randn(4, 32, requires_grad=True)
        ids = torch.randint(1, 64, (4, 12))
        before = {name: parameter.detach().clone() for name, parameter in language.named_parameters()}
        text = language(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[:, 0]
        prediction = decoder(image, ids[:, :-1])
        loss = F.cross_entropy(prediction.reshape(-1, 64), ids[:, 1:].reshape(-1))
        # 비교 조건 사이에서 BERT 출력으로 이어지는 손실 경로만 추가합니다.
        if condition == "caption_plus_contrastive":
            loss = loss + alignment_loss(image, text)
        loss.backward()
        gradients = [parameter.grad for parameter in language.parameters() if parameter.grad is not None]
        optimizer = torch.optim.SGD(language.parameters(), lr=0.001)
        optimizer.step()
        delta_squared = sum(float((parameter.detach() - before[name]).square().sum())
                            for name, parameter in language.named_parameters())
        result["conditions"][condition] = {
            "bert_parameters_with_gradient": len(gradients),
            "bert_gradient_norm": sum(float(gradient.square().sum()) for gradient in gradients) ** 0.5,
            "bert_parameter_change_norm": delta_squared ** 0.5,
            "image_gradient_norm": float(image.grad.norm()),
            "decoder_parameters_with_gradient": sum(p.grad is not None for p in decoder.parameters()),
        }
    assert result["conditions"]["caption_only"]["bert_parameters_with_gradient"] == 0
    assert result["conditions"]["caption_plus_contrastive"]["bert_parameter_change_norm"] > 0
    return result


@torch.inference_mode()
def decoder_check():
    tree = ast.parse((ROOT / "reference/graduatiothesisproject.py").read_text())
    decoder_class = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == "TextDecoder")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[decoder_class], type_ignores=[]), "original_decoder", "exec"), namespace)
    torch.manual_seed(42)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    language = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                                   num_attention_heads=4, intermediate_size=64))
    original = namespace["TextDecoder"](len(tokenizer), 32, 32, 4, 1, max_len=12).eval()
    corrected = Captioner(nn.Identity(), language, tokenizer, 32, 12, 1).eval()
    pixels = torch.randn(2, 32)
    prefix = torch.randint(1, 64, (2, 8))
    changed = prefix.clone()
    changed[:, -1] = 70
    original_change = float((original(pixels, prefix)[:, 0]
                             - original(pixels, changed)[:, 0]).abs().max())
    corrected_change = float((corrected.decode(pixels, prefix)[:, 0]
                              - corrected.decode(pixels, changed)[:, 0]).abs().max())
    assert original_change > 1e-6
    assert corrected_change < 1e-6
    return {"setup": "Change only the last input token and compare first-position logits in eval mode.",
            "original_max_change": original_change, "corrected_max_change": corrected_change,
            "corrected_vocabulary_size": corrected.classifier.out_features,
            "tokenizer_size": len(tokenizer)}


@torch.inference_mode()
def extract_features():
    manifest_path = ROOT / "data/coco-pilot/manifest.json"
    records = json.loads(manifest_path.read_text())
    source_hash = hashlib.sha256(manifest_path.read_bytes())
    for row in records:
        source_hash.update((manifest_path.parent / row["image"]).read_bytes())
    digest = source_hash.hexdigest()
    path = OUTPUT / "coco-features.npz"
    if path.exists():
        with np.load(path) as cache:
            if str(cache["manifest_sha256"]) == digest:
                return {key: cache[key].copy() for key in ("image", "text", "split")}
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    language = BertModel.from_pretrained("bert-base-uncased").eval()
    weights = next((Path.home() / ".cache/huggingface/hub").glob(
        "models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/snapshots/*/model.safetensors"), None)
    vision = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k",
                               num_classes=0, pretrained=weights is None).eval()
    if weights:
        from safetensors.torch import load_file
        vision.load_state_dict({key: value for key, value in load_file(str(weights)).items()
                                if key not in {"head.weight", "head.bias"}})
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    images, texts = [], []
    for start in range(0, len(records), 4):
        batch = records[start:start + 4]
        pixels = []
        for row in batch:
            with Image.open(manifest_path.parent / row["image"]) as image:
                pixels.append(transform(image.convert("RGB")))
        images.append(vision(torch.stack(pixels)).numpy())
        tokens = tokenizer([row["caption"] for row in batch], padding=True, truncation=True,
                           max_length=64, return_tensors="pt")
        texts.append(language(**tokens).last_hidden_state[:, 0].numpy())
    data = {"image": np.concatenate(images), "text": np.concatenate(texts),
            "split": np.array([row["split"] for row in records])}
    np.savez_compressed(path, **data, manifest_sha256=digest)
    return data


@torch.inference_mode()
def retrieval_metrics(image, text):
    scores = F.normalize(image) @ F.normalize(text).T
    result = {}
    for direction, matrix in (("image_to_text", scores), ("text_to_image", scores.T)):
        order = matrix.argsort(dim=1, descending=True)
        targets = torch.arange(len(matrix))[:, None]
        ranks = (order == targets).nonzero()[:, 1] + 1
        result[direction] = {"recall_at_1": float((ranks == 1).float().mean()),
                             "mrr": float((1 / ranks.float()).mean()), "ranks": ranks.tolist()}
    return result


def feature_ablation(data):
    image = torch.from_numpy(data["image"])
    text = torch.from_numpy(data["text"])
    split_masks = {split: torch.from_numpy(data["split"] == split)
                   for split in ("train", "validation", "test")}
    train = split_masks["train"]
    runs = []
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        projection = nn.Linear(text.shape[1], image.shape[1])
        # 캡션 손실과 연결되지 않은 투영층은 업데이트 전 상태로 비교합니다.
        baseline = {split: retrieval_metrics(image[mask], projection(text[mask]))
                    for split, mask in split_masks.items()}
        optimizer = torch.optim.AdamW(projection.parameters(), lr=0.0001)
        history = []
        for step in range(201):
            loss = alignment_loss(image[train], projection(text[train]))
            if step % 20 == 0:
                history.append({"step": step, "train_contrastive_loss": float(loss.detach())})
            if step < 200:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        trained = {split: retrieval_metrics(image[mask], projection(text[mask]))
                   for split, mask in split_masks.items()}
        runs.append({"seed": seed, "unconnected_projection": baseline,
                     "contrastive_projection": trained, "history": history})
    return {"setup": "Frozen pretrained ViT/BERT features; only the 768-to-768 text projection is trained.",
            "baseline": "An unconnected text projection receives no update under caption-only loss.",
            "steps": 200, "learning_rate": 0.0001,
            "split_sizes": {split: int(mask.sum()) for split, mask in split_masks.items()}, "runs": runs}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    decoder = decoder_check()
    (OUTPUT / "decoder-check.json").write_text(json.dumps(decoder, indent=2) + "\n")
    graph = gradient_check()
    (OUTPUT / "gradient-check.json").write_text(json.dumps(graph, indent=2) + "\n")
    print("학습 신호 전달 검증을 완료했습니다.", flush=True)
    comparison = feature_ablation(extract_features())
    (OUTPUT / "model-ablation.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print("고정된 인코더 특징의 대조학습 비교를 완료했습니다.", flush=True)


if __name__ == "__main__":
    main()
