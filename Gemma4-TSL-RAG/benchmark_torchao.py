import argparse
import json
import time
from pathlib import Path

import torch
from torchao.quantization import Int4WeightOnlyConfig
from torchao.quantization.quantize_.workflows import Int4PackingFormat
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig


MODEL_ID = "google/gemma-4-E4B-it"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    case = json.loads(args.case.read_text(encoding="utf-8"))

    quantization_config = TorchAoConfig(
        quant_type=Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=Int4PackingFormat.PLAIN_INT32,
        )
    )
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": 0},
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - started

    inputs = tokenizer(
        case["prompt"],
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda:0")
    input_tokens = inputs["input_ids"].shape[-1]

    warmup = tokenizer("測試", return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        model.generate(**warmup, max_new_tokens=1, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.eot_token_id],
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    generated_ids = output[0, input_tokens:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    report = {
        "backend": "torchao-int4-weight-only-group128",
        "input_text": case["input_text"],
        "baseline_tsl": case["tsl"],
        "candidate_tsl": result,
        "same_output": result == case["tsl"],
        "input_tokens": input_tokens,
        "output_tokens": len(generated_ids),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
