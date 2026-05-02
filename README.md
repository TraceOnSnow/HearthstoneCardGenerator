# HearthGen: KG-Augmented Hearthstone Card Art Generation

HearthGen is a course-project prototype for generating Hearthstone-style custom card artwork from natural-language DIY card ideas.

The core idea is simple: Hearthstone cards are not just images. They have class identity, card type, mana cost, keywords, mechanics, generated tokens, and references to existing card families. This project converts official Hearthstone card data into structured semantics, builds a semantic knowledge graph (KG), retrieves mechanically relevant reference cards, and uses those references with Stable Diffusion 1.5 plus a Hearthstone LoRA adapter.

## What Is In This Repo

- `app/semantics/`: builds structured semantic records from raw Hearthstone JSON.
- `app/semantic_kg/`: builds and queries the semantic KG.
- `app/retrieval/`: TF-IDF, CLIP, and KG retrieval baselines.
- `app/generation/`: generation comparison and image metric helpers.
- `app/kg/`: older/general KG pipeline utilities and shared LLM API code.
- `scripts/`: command-line entrypoints for each pipeline stage.
- `configs/`: fixed prompts and retrieval queries used in the final experiments.
- `submission/final_report/`: final report source, compiled PDF, and report figures.
- `results/`: selected final experiment outputs committed for inspection.

Large local datasets, model checkpoints, private notes, and scratch outputs are ignored by Git.

## Data

Tracked small data files:

- `data/cards_all.jsonl`: all Hearthstone cards used by the semantic pipeline.
- `data/cards_collectible.jsonl`: collectible-card subset.
- `data/sample_img/`: small image fixtures.

Ignored local datasets:

- `data/hf_hearthstone_art_512/`: local copy of Hearthstone artwork from the shared Hugging Face dataset.
- `data/hs_art_512/`: local artwork dump.

The artwork dataset is expected to come from:

```bash
uv run python scripts/fetch_hf_art_dataset.py \
  --repo-id comp646/hearthstone-art-512 \
  --output-dir data/hf_hearthstone_art_512
```

If the Hugging Face dataset is private, authenticate first with `huggingface-cli login` or set `HF_TOKEN`.

## Environment

This repo uses `uv`.

```bash
uv sync
```

For diffusion / CLIP / LoRA generation:

```bash
uv sync --extra diffusion
```

For LLM enrichment or LLM-based judging, create a local `.env`:

```bash
MINIMAX_API_KEY=...
GOOGLE_API_KEY=...
```

MiniMax is used through `app/kg/llm.py`. The default MiniMax model in current scripts is `MiniMax-M2.7`.

## Main Pipeline

### 1. Build Structured Semantics

Build deterministic base semantics for all cards:

```bash
uv run python scripts/build_semantics.py \
  --cards data/cards_all.jsonl \
  --out-dir data/semantics
```

Outputs include:

- `data/semantics/cards_semantics_base.jsonl`
- `data/semantics/lora_captions.jsonl`
- derived-card edges and caption rows

Optional MiniMax enrichment:

```bash
uv run python scripts/enrich_semantics.py \
  --semantics data/semantics/cards_semantics_base.jsonl \
  --out-dir data/semantics_enriched_full \
  --chunk-strategy set_class \
  --chunk-size 5 \
  --provider minimax \
  --model MiniMax-M2.7 \
  --temperature 0.1 \
  --timeout-seconds 360 \
  --concurrency 8 \
  --max-retries 3
```

Merge enriched output back into current semantics:

```bash
uv run python scripts/merge_enriched_semantics.py \
  --base data/semantics/cards_semantics_base.jsonl \
  --llm-outputs data/semantics_enriched_full/enrichment_llm_outputs.jsonl \
  --out-dir data/semantics_enriched_current
```

### 2. Build Semantic KG

```bash
uv run python scripts/build_semantic_kg.py \
  --semantics data/semantics_enriched_current/cards_semantics_enriched.jsonl \
  --out-dir data/semantic_kg
```

Important outputs:

- `data/semantic_kg/nodes.jsonl`
- `data/semantic_kg/edges.jsonl`
- `data/semantic_kg/card_index.jsonl`
- `data/semantic_kg/graph.json`

Visualize small card neighborhoods:

```bash
uv run python scripts/visualize_semantic_kg.py \
  --kg-dir data/semantic_kg \
  --out-dir data/semantic_kg/sample_vis \
  --sample-size 3
```

### 3. Query and Retrieve

Parse one natural-language request:

```bash
uv run python scripts/parse_kg_query.py \
  "I want a Warrior Rager meme card that gains Armor."
```

Run KG retrieval:

```bash
uv run python scripts/run_kg_retrieval.py \
  --card-index data/semantic_kg/card_index.jsonl \
  --queries configs/retrieval_queries.json \
  --out results/kg_retrieval/kg_results.jsonl
```

Run TF-IDF baseline:

```bash
uv run python scripts/run_tfidf_retrieval.py \
  --captions data/semantics/lora_captions.jsonl \
  --queries configs/retrieval_queries.json \
  --out results/retrieval_eval/tfidf_results.jsonl
```

Run CLIP baseline:

```bash
uv run python scripts/run_clip_retrieval.py \
  --captions data/semantics/lora_captions.jsonl \
  --queries configs/retrieval_queries.json \
  --image-root data/hf_hearthstone_art_512 \
  --out results/retrieval_eval/clip_results.jsonl
```

### 4. End-to-End DIY Retrieval and Card Design Evaluation

This is the main final-report retrieval/design script. It uses the 19 DIY prompts in `configs/diy_user_prompts.json`.

Mock mode, fast and CI-friendly:

```bash
uv run python scripts/run_diy_retrieval_design_eval.py \
  --out-dir results/diy_retrieval_design_eval
```

Real CLIP + real MiniMax card design + real MiniMax judging:

```bash
uv run python scripts/run_diy_retrieval_design_eval.py \
  --no-mock-clip \
  --no-mock-design \
  --no-mock-judge \
  --out-dir results/diy_retrieval_design_eval_real_llm \
  --timeout-seconds 120
```

Final committed outputs:

- `results/diy_retrieval_design_eval_real_llm/retrieval_results.jsonl`
- `results/diy_retrieval_design_eval_real_llm/diy_card_designs.jsonl`
- `results/diy_retrieval_design_eval_real_llm/table_retrieval_metrics.md`
- `results/diy_retrieval_design_eval_real_llm/table_design_text_metrics.md`
- `results/diy_retrieval_design_eval_real_llm/retrieval_grid.html`

### 5. LoRA / Generation

Pretrained project LoRA weights are hosted as a Hugging Face model repo:

```text
comp646/hearthstone-sd15-lora
```

Use this repo when you only need inference or reproduction of the final generation experiments. Train your own adapter only if you want to reproduce the LoRA training stage.

Fetch and prepare LoRA art metadata:

```bash
uv run python scripts/fetch_hf_art_dataset.py \
  --repo-id comp646/hearthstone-art-512 \
  --output-dir data/hf_hearthstone_art_512

uv run python scripts/prepare_lora_hf_metadata.py \
  --metadata data/hf_hearthstone_art_512/metadata.jsonl \
  --semantics data/semantics_enriched_current/cards_semantics_enriched.jsonl \
  --out data/hf_hearthstone_art_512/metadata.jsonl
```

Train a Stable Diffusion 1.5 LoRA adapter:

```bash
uv run python scripts/train_lora_sd15.py \
  --pretrained-model stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --metadata data/hf_hearthstone_art_512/metadata.jsonl \
  --image-root data/hf_hearthstone_art_512 \
  --caption-column text \
  --image-column file_name \
  --output-dir models/sd15-hearthstone-lora \
  --train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --rank 16 \
  --mixed-precision fp16
```

Generate one image with a trained LoRA:

```bash
uv run python scripts/generate_with_lora_sd15.py \
  --lora-dir comp646/hearthstone-sd15-lora \
  --prompt "hsart Hearthstone card art, Warrior minion, iron armor, glowing embers"
```

Run the final 2x2 generation evaluation over the DIY prompts:

```bash
uv run python scripts/run_diy_generation_eval.py \
  --no-mock \
  --out-dir results/diy_generation_eval_real \
  --steps 24 \
  --skip-existing
```

Evaluate generated images with CLIP-style proxy metrics:

```bash
uv run python scripts/evaluate_generation_metrics.py \
  --plan results/diy_generation_eval_real/generation_plan.jsonl \
  --out results/diy_generation_eval_real/generation_metrics.jsonl \
  --summary-out results/diy_generation_eval_real/generation_metrics_summary.csv \
  --style-reference-limit 64 \
  --batch-size 4 \
  --device cuda
```

Final committed outputs:

- `results/diy_generation_eval_real/images/`: 76 generated images for 19 prompts x 4 methods.
- `results/diy_generation_eval_real/generation_contact_sheet.png`
- `results/diy_generation_eval_real/generation_grid.html`
- `results/diy_generation_eval_real/table_generation_metrics.md`

## Final Report Artifacts

The final report lives in:

```text
submission/final_report/
```

Important files:

- `submission/final_report/report.tex`
- `submission/final_report/report.pdf`
- `submission/final_report/figures/`

The report currently includes:

- Figure 1: end-to-end Iron Rager case study.
- Semantic KG neighborhood visualization.
- Figure 2: six LoRA + KG-reference qualitative outputs.
- Retrieval and generation metric tables.

Compile:

```bash
cd submission/final_report
pdflatex -interaction=nonstopmode report.tex
```

## Final Results Tracked in Git

Only selected final results are unignored:

- `results/figure1_iron_rager/`
- `results/diy_generation_eval_real/`
- `results/diy_retrieval_design_eval_real_llm/`
- `results/final_generation_eval/`

Other `results/` directories are scratch/smoke/intermediate outputs and remain ignored.

## Script Reference

Data and artwork:

- `scripts/fetch_cards.py`: fetch card JSON from Blizzard/API source.
- `scripts/fetch_metadata.py`: fetch Hearthstone metadata ID-name maps.
- `scripts/fetch_hf_art_dataset.py`: download the shared HF artwork dataset.
- `scripts/download_cards.py`, `scripts/crop_cards.py`: older card-image download/crop utilities.
- `scripts/download_hs_art.py`: experimental local game-file artwork extraction helper.

Semantics and KG:

- `scripts/build_semantics.py`: raw cards to structured semantic records.
- `scripts/enrich_semantics.py`: MiniMax/Gemini enrichment over semantic chunks.
- `scripts/merge_enriched_semantics.py`: merge LLM enrichment into base semantics.
- `scripts/build_semantic_kg.py`: structured semantics to KG nodes/edges/card index.
- `scripts/visualize_semantic_kg.py`: small HTML KG neighborhood visualizations.
- `scripts/parse_kg_query.py`: natural-language request to structured KG query.
- `scripts/run_kg_retrieval.py`: semantic KG retrieval.

Retrieval baselines:

- `scripts/run_tfidf_retrieval.py`: TF-IDF caption retrieval.
- `scripts/run_clip_retrieval.py`: CLIP text-to-image retrieval.
- `scripts/render_retrieval_grid.py`: HTML grid for retrieval results.
- `scripts/make_judging_template.py`, `scripts/summarize_judging.py`: manual retrieval judging utilities.

Generation:

- `scripts/train_lora_sd15.py`: train Stable Diffusion 1.5 LoRA.
- `scripts/generate_with_lora_sd15.py`: single-image LoRA inference.
- `scripts/run_generation_comparison.py`: older prompt generation comparison runner.
- `scripts/run_diy_generation_eval.py`: final 2x2 DIY generation evaluation.
- `scripts/evaluate_generation_metrics.py`: automatic proxy metrics for generated images.
- `scripts/make_generation_judging_template.py`, `scripts/summarize_generation_judging.py`: manual generation judging utilities.

End-to-end final experiments:

- `scripts/run_diy_retrieval_design_eval.py`: final retrieval + card-design + judging pipeline.
- `scripts/design_card_from_kg.py`: single-card KG-augmented design probe.

Legacy / demo:

- `scripts/run_kg.py`, `scripts/run_kg_demo.py`, `app/kg_demo/`: early KG demo code kept for reference.
- `scripts/visualize_graph.py`: older graph visualization helper.

## Notes

- Do not commit `.env`, local datasets, model checkpoints, or private notes.
- `llm_knowledge/` is intentionally ignored. It contains private planning notes, paper PDFs, and templates used during development.
- The committed final results are intended for inspection, not for retraining. Use the Hugging Face dataset and scripts above for full reproduction.
