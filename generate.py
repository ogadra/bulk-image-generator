import hashlib
import re
from itertools import batched, product
from pathlib import Path

import torch
from diffusers import FluxPipeline, PipelineQuantizationConfig
from PIL import Image

WORDS = Path(__file__).parent / "words"
MODEL = "black-forest-labs/FLUX.1-schnell"
REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"
GENERATED_SIDE = 512
SIDE = 128
STEPS = 4
QUALITY = 80
BATCH_SIZE = 8
OUT = Path("out")


def lines(name):
    return [
        line.strip() for line in (WORDS / name).read_text().splitlines() if line.strip()
    ]


def slug(word):
    return re.sub(r"[^a-z0-9]+", "-", word.lower()).strip("-")


def entries():
    out = []
    for noun, colour, adjective in product(
        lines("nouns.txt"), lines("colours.txt"), lines("adjectives.txt")
    ):
        path = f"{slug(noun)}/{slug(colour)}/{slug(adjective)}.jpg"
        prompt = (
            f"{adjective} {colour} {noun}, "
            "a single piece of seating, product photo, plain background"
        )
        seed = int.from_bytes(hashlib.sha256(path.encode()).digest()[:4], "big")
        out.append((path, prompt, seed))
    if len({path for path, _, _ in out}) != len(out):
        raise SystemExit("two word combinations share a file name")
    return out


def write_manifest(all_entries):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = ["path\tprompt\tseed"] + [
        f"{path}\t{prompt}\t{seed}" for path, prompt, seed in all_entries
    ]
    (OUT / "manifest.tsv").write_text("\n".join(rows) + "\n")


def load_pipeline():
    quantization = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["transformer", "text_encoder_2"],
    )
    pipeline = FluxPipeline.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline.to("cuda")


def generate(pending):
    pipeline = load_pipeline()
    done = 0
    for batch in batched(pending, BATCH_SIZE):
        images = pipeline(
            prompt=[prompt for _, prompt, _ in batch],
            generator=[
                torch.Generator("cuda").manual_seed(seed) for _, _, seed in batch
            ],
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            height=GENERATED_SIDE,
            width=GENERATED_SIDE,
            max_sequence_length=256,
        ).images
        for (path, _, _), image in zip(batch, images):
            target = OUT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".part")
            image.resize((SIDE, SIDE), Image.Resampling.LANCZOS).save(
                partial, format="JPEG", quality=QUALITY
            )
            partial.rename(target)
        done += len(batch)
        print(f"{done}/{len(pending)}", flush=True)


def main():
    all_entries = entries()
    write_manifest(all_entries)
    generate([entry for entry in all_entries if not (OUT / entry[0]).exists()])


if __name__ == "__main__":
    main()
