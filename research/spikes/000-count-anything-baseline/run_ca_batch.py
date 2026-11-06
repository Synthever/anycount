"""
Spike 000: Count-Anything (SAM3 848M / 3.4GB checkpoint) Baseline on the 4 test images.
Loads the checkpoint once in a single batch evaluation pass on CPU and saves rendered
Count-Anything images and JSON coordinates.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import yaml

sys.path.insert(0, "/root/workspace/count-anything")
from count_anything.inference import CountAnythingResult, _slugify


def main() -> None:
    ca_root = Path("/root/workspace/count-anything")
    anycount_root = Path("/root/workspace/anycount")
    test_dir = anycount_root / "docs" / "image" / "test"
    out_dir = anycount_root / "research" / "spikes" / "000-count-anything-baseline" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("tennis-ball.jpg", "tennis ball", 43),
        ("Heidelbeeren.jpg", "blueberry", 113),
        ("Heidelbeeren-bnw.jpg", "blueberry", 110),
        ("candy-ball.jpg", "candy", 132),
    ]

    run_dir = out_dir / "_tmp_ca_batch"
    run_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = run_dir / "batch_annotation.json"
    config_path = run_dir / "batch_config.yaml"
    detail_path = run_dir / "batch_records.json"

    annotation = {}
    for idx, (filename, query, _) in enumerate(tasks):
        img_path = (test_dir / filename).resolve()
        key = f"{idx}__{_slugify(query)}"
        annotation[key] = {
            "idx": idx,
            "image_path": str(img_path),
            "image_from": "custom",
            "classes": [query],
            "annotation": {query: {"point": []}},
            "split": "inference",
            "selected_classes": [query],
            "selected_annotation": {query: {"point": []}},
        }
    annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")

    with (ca_root / "config" / "count_anything_test_cloc.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint = str(ca_root / "checkpoints" / "count_anything.pt")
    config["paths"]["train_annotation_file"] = str(annotation_path)
    config["paths"]["val_annotation_file"] = str(annotation_path)
    config["paths"]["experiment_log_dir"] = str(run_dir)
    config["paths"]["stage1_checkpoint_path"] = checkpoint

    trainer = config["trainer"]
    for split in ("train", "val"):
        trainer["data"][split]["dataset"]["ann_file"] = str(annotation_path)
        trainer["data"][split]["batch_size"] = 1
        trainer["data"][split]["num_workers"] = 0
    trainer["model"]["load_from_HF"] = False
    trainer["model"]["checkpoint_path"] = None
    trainer["checkpoint"]["model_weight_initializer"]["checkpoint_path"] = checkpoint
    trainer["checkpoint"]["save_dir"] = str(run_dir / "checkpoints")
    trainer["logging"]["log_dir"] = str(run_dir)
    trainer["logging"]["tensorboard_writer"]["log_dir"] = str(run_dir / "tensorboard")
    trainer["meters"]["val"]["all"]["counting"]["detail_records_path"] = str(detail_path)
    trainer["meters"]["val"]["all"]["counting"]["include_points"] = True
    trainer["meters"]["val"]["all"]["counting"]["include_scores"] = True
    trainer["meters"]["val"]["all"]["counting"]["expected_num_records"] = len(tasks)
    trainer["accelerator"] = "cpu"
    trainer["distributed"]["backend"] = "gloo"
    trainer["optim"]["amp"]["enabled"] = False
    trainer["empty_gpu_mem_cache_after_eval"] = False

    config["launcher"]["gpus_per_node"] = 1
    config["launcher"]["experiment_log_dir"] = str(run_dir)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ca_root) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    print(f"[000-CA] Running batch of {len(tasks)} images on CPU...", flush=True)
    subprocess.run(
        [
            str(ca_root / ".venv" / "bin" / "python"),
            "-m",
            "count_anything.train.train",
            "-c",
            str(config_path),
            "--use-cluster",
            "0",
            "--num-gpus",
            "1",
        ],
        cwd=str(ca_root),
        env=env,
        check=True,
    )
    total_elapsed = time.perf_counter() - t0
    avg_elapsed = round(total_elapsed / len(tasks), 2)
    print(f"[000-CA] Batch finished in {total_elapsed:.2f}s (~{avg_elapsed}s/img)", flush=True)

    records = json.loads(detail_path.read_text(encoding="utf-8")).get("predictions") or []
    summary = []
    for (filename, query, gt), rec in zip(tasks, records):
        img_path = test_dir / filename
        slug = f"{img_path.stem}__{_slugify(query)}"
        out_img = out_dir / f"{slug}.jpg"
        res = CountAnythingResult(
            image_path=str(img_path),
            text_query=query,
            count=int(rec["pred_count"]),
            pred_points=rec.get("points", []),
            run_dir=str(out_dir),
        )
        res.save(out_img)
        err = res.count - gt
        mape = round(abs(err) / gt * 100, 1) if gt else 0.0
        summary.append({
            "image": filename,
            "query": query,
            "gt": gt,
            "pred_count": res.count,
            "signed_error": err,
            "mape_pct": mape,
            "avg_elapsed_s": avg_elapsed,
            "output_image": str(out_img.relative_to(anycount_root / "research")),
        })
        print(f"  -> {filename} ({query!r}): count={res.count} (GT~{gt}, err={err:+d}, MAPE={mape}%)", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import shutil
    shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
