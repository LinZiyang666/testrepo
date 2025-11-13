import argparse
import json
import os
import ast
from typing import List, Dict, Any, Tuple

import io
import time
import torch
from PIL import Image
from datasets import load_dataset, get_dataset_config_names
from datasets import Image as HFImage
from tqdm import tqdm

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

LETTERS = ["A", "B", "C", "D"]


def build_prompt(question: str, options: List[str], opt_max_chars: int) -> str:
    options = options[:4]
    lines = []
    lines.append("Answer the multiple-choice question. Reply with one letter A B C D only.")
    lines.append(f"Question: {question}")
    lines.append("Options:")
    for i, opt in enumerate(options):
        opt_str = str(opt)
        if opt_max_chars > 0 and len(opt_str) > opt_max_chars:
            opt_str = opt_str[:opt_max_chars]
        lines.append(f"{LETTERS[i]}. {opt_str}")
    lines.append("Final answer:")
    return "\n".join(lines)


def _open_image_from_url(url: str) -> Image.Image | None:
    try:
        import requests
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def extract_images_from_sample(sample: Dict[str, Any], image_root: str, max_image_pixels: int, debug: bool) -> List[Image.Image]:
    def _downscale(img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        if max_image_pixels <= 0:
            return img
        w, h = img.size
        if w * h <= max_image_pixels:
            return img
        scale = (max_image_pixels / float(w * h)) ** 0.5
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return img.resize((new_w, new_h), Image.BILINEAR)

    imgs = []
    candidates = []
    img_keys = [k for k in sample.keys() if isinstance(k, str) and k.startswith("image_")]
    img_keys.sort()
    if img_keys:
        for k in img_keys:
            v = sample.get(k)
            if v is not None:
                candidates.append(v)
    else:
        if "images" in sample and sample["images"] is not None:
            candidates = sample["images"]
        elif "image" in sample and sample["image"] is not None:
            candidates = sample["image"]
        elif "image_paths" in sample and sample["image_paths"] is not None:
            candidates = sample["image_paths"]
        elif "image_path" in sample and sample["image_path"] is not None:
            candidates = sample["image_path"]

    if isinstance(candidates, list):
        items = candidates
    elif candidates is None:
        items = []
    else:
        items = [candidates]

    for it in items:
        try:
            if isinstance(it, Image.Image):
                imgs.append(_downscale(it))
            elif isinstance(it, dict):
                if "path" in it and isinstance(it["path"], str):
                    p = it["path"]
                    if p.startswith("http"):
                        im = _open_image_from_url(p)
                        if im is not None:
                            imgs.append(_downscale(im))
                    else:
                        p_abs = p if os.path.isabs(p) else os.path.join(image_root, p) if image_root else p
                        if os.path.exists(p_abs):
                            imgs.append(_downscale(Image.open(p_abs)))
                elif "bytes" in it:
                    b = it["bytes"]
                    try:
                        if isinstance(b, (bytes, bytearray)):
                            imgs.append(_downscale(Image.open(io.BytesIO(b))))
                        else:
                            imgs.append(_downscale(Image.open(b)))
                    except Exception:
                        continue
            elif isinstance(it, str):
                if it.startswith("http"):
                    im = _open_image_from_url(it)
                    if im is not None:
                        imgs.append(_downscale(im))
                else:
                    p = it if os.path.isabs(it) else os.path.join(image_root, it) if image_root else it
                    if os.path.exists(p):
                        imgs.append(_downscale(Image.open(p)))
        except Exception:
            continue
    return imgs


def make_key(sample: Dict[str, Any], subject: str, split_name: str, idx: int) -> str:
    for k in ["id", "question_id", "sample_id", "qid"]:
        if k in sample and isinstance(sample[k], str) and len(sample[k]) > 0:
            return f"{subject}_{split_name}_{sample[k]}"
        if k in sample and isinstance(sample[k], int):
            return f"{subject}_{split_name}_{sample[k]}"
    category = sample.get("category") or sample.get("subject") or "Unknown"
    if isinstance(category, str):
        category_key = category.strip().replace(" ", "_")
    else:
        category_key = "Unknown"
    return f"{subject}_{split_name}_{category_key}_{idx}"


def parse_choice(text: str, num_options: int) -> str:
    text = text.strip()
    allowed = LETTERS[:max(1, min(4, num_options))]
    for line in text.splitlines():
        s = line.strip()
        if len(s) == 1 and s.upper() in allowed:
            return s.upper()
        if s.lower().startswith("final answer"):
            tail = s.split(":")[-1].strip() if ":" in s else s.replace("final answer", "").strip()
            if len(tail) > 0 and tail[0].upper() in allowed:
                return tail[0].upper()
    for ch in text:
        up = ch.upper()
        if up in allowed:
            return up
    for ch in text:
        if ch.isdigit():
            n = int(ch)
            if 1 <= n <= len(allowed):
                return allowed[n - 1]
    return allowed[0]


def normalize_options(sample: Dict[str, Any]) -> List[str]:
    raw = sample.get("options") or sample.get("choices") or []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        txt = raw.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                return [raw]
        return [raw]
    try:
        return list(raw) if raw is not None else []
    except Exception:
        return []


def build_conversations_and_media(
    batch_samples: List[Dict[str, Any]],
    image_root: str,
    max_image_pixels: int,
    opt_max_chars: int,
    debug: bool,
) -> Tuple[List[Any], List[List[Image.Image]], List[str], List[str]]:
    conversations = []
    batch_images = []
    prompts_preview = []
    keys_preview = []
    for s in batch_samples:
        q = s.get("question") or s.get("prompt") or ""
        options = normalize_options(s)
        prompt = build_prompt(q, options, opt_max_chars)
        if debug:
            print("\n[debug] full_input_text:")
            print(prompt)
            print("[debug] end_of_input_text\n")
        imgs = extract_images_from_sample(s, image_root, max_image_pixels, debug)
        content = []
        for im in imgs:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": prompt})
        conv = [{"role": "user", "content": content}]
        conversations.append(conv)
        batch_images.append(imgs)
        preview = prompt[:200].replace("\n", " | ")
        prompts_preview.append(preview)
        keys_preview.append("")
    return conversations, batch_images, prompts_preview, keys_preview


def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def resolve_attn_impl(req: str) -> str:
    req = (req or "").lower().strip()
    if req == "sdpa":
        return "sdpa"
    return "eager"


def load_predictions_from(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def is_mcq(example: Dict[str, Any]) -> bool:
    qt = example.get("question_type") or example.get("type") or example.get("format")
    if isinstance(qt, str) and qt.lower().replace(" ", "") in ["multiple-choice", "multiplechoice", "mc", "mcq"]:
        return True
    if isinstance(example.get("options") or example.get("choices"), list):
        return True
    return False


def get_gold_letter(s: Dict[str, Any], num_options: int) -> str | None:
    gold = s.get("answer") or s.get("gt_answer") or s.get("label")
    if isinstance(gold, str):
        g = gold.strip().upper()
        if g in LETTERS[:max(1, min(4, num_options))]:
            return g
        for ch in g:
            if ch in LETTERS[:max(1, min(4, num_options))]:
                return ch
    if isinstance(gold, int) and 0 <= gold < min(4, num_options):
        return LETTERS[gold]
    return None


def run_inference_batch(
    model,
    processor,
    batch_conversations: List[Any],
    batch_images: List[List[Image.Image]],
    max_new_tokens: int,
    temperature: float,
    use_audio_in_video: bool,
    max_input_tokens: int,
    strict_mc: bool,
    debug: bool,
) -> List[str]:
    texts = processor.apply_chat_template(
        batch_conversations,
        add_generation_prompt=True,
        tokenize=False
    )

    has_any_image = any(len(x) > 0 for x in batch_images)

    proc_kwargs = dict(
        text=texts,
        images=batch_images if has_any_image else None,
        audio=None,
        videos=None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    if "input_ids" in inputs:
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            prompt_lens = [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]
        else:
            prompt_lens = inputs["input_ids"].ne(pad_id).sum(dim=1).tolist()
        prompt_len_for_slice = inputs["input_ids"].shape[1]
    else:
        prompt_lens = [0] * len(batch_conversations)
        prompt_len_for_slice = 0

    gen_kwargs = dict(
        thinker_max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        return_audio=False,
        return_dict_in_generate=True,
        use_audio_in_video=use_audio_in_video,
    )
    if strict_mc:
        allowed_tokens = ["A", "B", "C", "D"]
        allowed_ids = processor.tokenizer.convert_tokens_to_ids(allowed_tokens)

        def _prefix_allowed_tokens_fn(batch_id, cur_input_ids):
            prompt_len = prompt_lens[batch_id]
            gen_len = cur_input_ids.shape[0] - prompt_len
            if gen_len <= 0:
                return [t for t in allowed_ids if t is not None]
            eos = [processor.tokenizer.eos_token_id] if processor.tokenizer.eos_token_id is not None else []
            nl_id = processor.tokenizer.convert_tokens_to_ids("\n")
            res = []
            if nl_id is not None:
                res.append(nl_id)
            res += [t for t in eos if t is not None]
            return res

        gen_kwargs["prefix_allowed_tokens_fn"] = _prefix_allowed_tokens_fn

    try:
        gen_out = model.generate(**inputs, **gen_kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if "no available kernel" in msg or "sdpa" in msg or "scaled_dot_product_attention" in msg:
            if hasattr(model, "config"):
                model.config._attn_implementation = "eager"
                model.config._attn_implementation_internal = "eager"
            gen_out = model.generate(**inputs, **gen_kwargs)
        else:
            raise

    if hasattr(gen_out, "sequences"):
        sequences = gen_out.sequences
    elif isinstance(gen_out, torch.Tensor):
        sequences = gen_out
    elif isinstance(gen_out, tuple):
        sequences = None
        for item in gen_out:
            if isinstance(item, torch.Tensor) and item.dim() == 2:
                sequences = item
                break
        if sequences is None and hasattr(gen_out[0], "sequences"):
            sequences = gen_out[0].sequences
        if sequences is None:
            raise TypeError(f"Cannot find sequences in generate output type {type(gen_out)}")
    else:
        raise TypeError(f"Unsupported generate output type {type(gen_out)}")

    gen_tokens = sequences[:, prompt_len_for_slice:]

    decoded = processor.batch_decode(
        gen_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    decoded = [d.strip() for d in decoded]

    if debug:
        print("\n========== DEBUG RAW MODEL OUTPUT ==========")
        for i, d in enumerate(decoded):
            print(f"[debug] SAMPLE {i}")
            print(f"[debug] raw_output = {repr(d)}")
            print("---------------------------------------------")
        print("=============================================\n")

    return decoded


def count_done_for_subject_split(predictions: Dict[str, str], subject: str, split: str) -> int:
    prefix = f"{subject}_{split}_"
    return sum(1 for k in predictions.keys() if isinstance(k, str) and k.startswith(prefix))


def cast_image_like_columns(ds):
    cols = list(ds.column_names)
    for col in cols:
        if col in ["image", "images", "image_path", "image_paths"] or col.startswith("image_"):
            try:
                ds = ds.cast_column(col, HFImage())
            except Exception:
                pass
    return ds


def debug_describe_local_image(model, processor, args):
    img_path = "image.png"
    if not os.path.exists(img_path):
        print(f"[debug] image_debug enabled but file not found at {img_path}")
        return
    img = Image.open(img_path).convert("RGB")
    prompt = "You are given an image. Describe in detail what is in the image."
    content = [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]
    conv = [{"role": "user", "content": content}]
    batch_conversations = [conv]
    batch_images = [[img]]

    print("[debug] running image_debug on image.png")
    outputs = run_inference_batch(
        model=model,
        processor=processor,
        batch_conversations=batch_conversations,
        batch_images=batch_images,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_audio_in_video=args.use_audio_in_video,
        max_input_tokens=args.max_input_tokens,
        strict_mc=False,
        debug=True,
    )
    print("\n[debug] image description output:")
    print(outputs[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_name", type=str, default="MMMU/MMMU")
    parser.add_argument("--subjects", type=str, default="all")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_json", type=str, default="mmmu_pred.json")
    parser.add_argument("--resume_json", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--device_map", type=str, default="cuda:0")
    parser.add_argument("--use_audio_in_video", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_image_pixels", type=int, default=1920 * 1080)
    parser.add_argument("--opt_max_chars", type=int, default=512)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--strict_mc", action="store_true")
    parser.add_argument("--image_debug", action="store_true", help="use local image.png to test vision behavior")
    args = parser.parse_args()

    attn_impl = resolve_attn_impl(args.attn_implementation)

    print(f"[info] load model {args.model_id} dtype={args.dtype} device_map={args.device_map} attn={attn_impl}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=attn_impl,
    )
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)

    if args.image_debug:
        debug_describe_local_image(model, processor, args)
        return

    if args.subjects.strip().lower() == "all":
        subjects = get_dataset_config_names(args.dataset_name)
    else:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not subjects:
        print("No subjects resolved")
        return

    predictions: Dict[str, str] = {}
    predictions.update(load_predictions_from(args.output_json))
    if args.resume_json:
        predictions.update(load_predictions_from(args.resume_json))
    atomic_write_json(args.output_json, predictions)

    has_answer = 0
    correct = 0

    model.eval()
    torch.set_grad_enabled(False)

    total_across = 0
    already_across = 0
    for subj in subjects:
        try:
            ds_preview = load_dataset(args.dataset_name, subj, split=args.split)
            ds_preview = cast_image_like_columns(ds_preview)
            idxs = [i for i, e in enumerate(ds_preview) if is_mcq(e)]
            total_across += len(idxs)
            already_across += count_done_for_subject_split(predictions, subj, args.split)
        except Exception:
            continue

    remaining = max(total_across - already_across, 0)
    print(f"[info] resume status total={total_across} already_done={already_across} remaining={remaining}")

    pbar = tqdm(total=remaining, unit="samples", disable=args.no_progress)
    start_time = time.perf_counter()

    for subject in subjects:
        try:
            ds = load_dataset(args.dataset_name, subject, split=args.split)
            ds = cast_image_like_columns(ds)
        except Exception as e:
            print(f"[warn] skip subject {subject} due to load failure: {e}")
            continue

        indices = [i for i, e in enumerate(ds) if is_mcq(e)]
        if not indices:
            print(f"[info] no MCQ in subject {subject}")
            continue

        if args.debug:
            print(f"[debug] subject={subject} mcq_count={len(indices)}")

        for start in range(0, len(indices), args.batch_size):
            chunk_ids = indices[start:start + args.batch_size]
            batch_samples = [ds[i] for i in chunk_ids]

            filtered_samples = []
            filtered_id_list = []
            keys_list = []
            for local_i, sample_idx in enumerate(chunk_ids):
                key = make_key(ds[sample_idx], subject, args.split, sample_idx)
                if key in predictions:
                    continue
                filtered_samples.append(batch_samples[local_i])
                filtered_id_list.append(sample_idx)
                keys_list.append(key)

            if not filtered_samples:
                continue

            t0 = time.perf_counter()
            conversations, batch_images, prompts_preview, keys_preview = build_conversations_and_media(
                filtered_samples, args.image_root, args.max_image_pixels, args.opt_max_chars, args.debug
            )
            keys_preview[:] = keys_list

            raw_outputs = run_inference_batch(
                model=model,
                processor=processor,
                batch_conversations=conversations,
                batch_images=batch_images,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                use_audio_in_video=args.use_audio_in_video,
                max_input_tokens=args.max_input_tokens,
                strict_mc=args.strict_mc,
                debug=args.debug,
            )
            t1 = time.perf_counter()
            batch_time = max(t1 - t0, 1e-6)
            batch_throughput = len(filtered_samples) / batch_time

            out_cursor = 0
            for idx_in_batch, sample_idx in enumerate(filtered_id_list):
                s = ds[sample_idx]
                key = keys_list[idx_in_batch]
                options = normalize_options(s)
                choice = parse_choice(raw_outputs[out_cursor], num_options=max(1, len(options)))
                if args.debug:
                    print(f"[debug] key={key} parsed_choice={choice} from_raw={repr(raw_outputs[out_cursor])}")

                gold = get_gold_letter(s, num_options=max(1, len(options)))
                if gold is not None:
                    has_answer += 1
                    if choice == gold:
                        correct += 1

                predictions[key] = choice
                out_cursor += 1
                atomic_write_json(args.output_json, predictions)
                pbar.update(1)
                elapsed = time.perf_counter() - start_time
                overall_tput = pbar.n / max(elapsed, 1e-6)
                pbar.set_postfix({"batch_sps": f"{batch_throughput:.2f}", "overall_sps": f"{overall_tput:.2f}", "subject": subject})

    pbar.close()

    if has_answer > 0:
        acc = correct / has_answer
        print(f"[info] multiple choice accuracy on {args.split}: {acc:.4f}  based on {has_answer} labeled samples across all subjects")
    else:
        print("[info] no ground truth answers found in split for accuracy calculation")
    print(f"[info] saved predictions to {args.output_json}")


if __name__ == "__main__":
    main()
