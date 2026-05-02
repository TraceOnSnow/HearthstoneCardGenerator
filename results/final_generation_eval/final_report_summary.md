# Final Generation Evaluation Summary

Date: 2026-05-01

## Prompt Review

The formal generation prompts were revised from the Chinese review notes before running generation:

- Druid prompt now emphasizes Nature magic, not Solar/Lunar imagery.
- Deathrattle prompt removes the undead guardian wording.
- Mage Freeze prompt removes the blizzard wording.
- Hunter prompt now asks for a Hunter summoning a powerful charging Beast.

Prompt review file:

```text
results/final_generation_eval/text_only/prompt_review.csv
```

## Experiment 1: SD Base vs LoRA, Text-Only

Setup:

- Same 10 prompts.
- Same seed.
- Same Stable Diffusion 1.5 base model.
- Compared `sd_text_only` against `lora_text_only`.
- No reference image conditioning.

Artifacts:

```text
results/final_generation_eval/text_only/generation_grid.html
results/final_generation_eval/text_only/generation_contact_sheet.png
results/final_generation_eval/text_only/generation_metrics.jsonl
results/final_generation_eval/text_only/generation_metrics_summary.csv
```

Metric summary:

| Method | Rows | Image Quality | Prompt Alignment | Hearthstone Style |
|---|---:|---:|---:|---:|
| SD base text-only | 10 | 0.494420 | 0.293449 | 0.786492 |
| LoRA text-only | 10 | 0.499449 | 0.293055 | 0.862727 |

Takeaway:

LoRA keeps prompt alignment roughly comparable to SD base, while improving the Hearthstone style similarity score from `0.786492` to `0.862727`. This supports the claim that the LoRA adapter helps move generated images toward Hearthstone-like artwork style.

## Experiment 2: LoRA + Text Retrieval Reference vs LoRA + KG Reference

Setup:

- Same 10 prompts.
- Same seed.
- Same LoRA adapter.
- Compared `LoRA + TF-IDF reference` against `LoRA + semantic KG reference`.
- TF-IDF is the ordinary text retrieval baseline.
- Semantic KG retrieval uses structured card semantics.

Artifacts:

```text
results/final_generation_eval/tfidf_reference/generation_grid.html
results/final_generation_eval/tfidf_reference/generation_contact_sheet.png
results/final_generation_eval/tfidf_reference/generation_metrics.jsonl
results/final_generation_eval/tfidf_reference/generation_metrics_summary.csv

results/final_generation_eval/kg_reference/generation_grid.html
results/final_generation_eval/kg_reference/generation_contact_sheet.png
results/final_generation_eval/kg_reference/generation_metrics.jsonl
results/final_generation_eval/kg_reference/generation_metrics_summary.csv
```

Metric summary:

| Method | Rows | Image Quality | Prompt Alignment | Hearthstone Style | Reference Similarity |
|---|---:|---:|---:|---:|---:|
| LoRA + TF-IDF reference | 10 | 0.501242 | 0.305663 | 0.873232 | 0.846176 |
| LoRA + semantic KG reference | 10 | 0.497910 | 0.283197 | 0.865776 | 0.817296 |

Takeaway:

Under the current automatic image-level metrics, TF-IDF reference conditioning scores higher than KG reference conditioning on image quality, prompt alignment, style similarity, and reference similarity. This does **not** yet support the claim that KG references improve image generation quality over ordinary text retrieval.

Important caveat:

These metrics evaluate generated image similarity and quality, not whether the retrieved reference cards are semantically better. The KG retrieval may still be more interpretable or mechanically grounded, but that requires a retrieval-side evaluation or stronger KG query/card metadata alignment.

## Metrics Used

- `image_quality_score`: CLIP zero-shot quality score comparing generated images against a high-quality illustration prompt versus a low-quality/blurry/artifact prompt.
- `clip_prompt_alignment`: CLIP similarity between the generation prompt and generated image.
- `style_similarity`: CLIP image similarity between generated image and a centroid of real Hearthstone artwork references.
- `reference_similarity`: CLIP image similarity between generated image and the selected reference image.

## Current Conclusion

Supported:

- LoRA improves Hearthstone style similarity in text-only generation.

Not yet supported:

- KG reference conditioning outperforming ordinary text retrieval reference conditioning in generated image quality/style metrics.

Recommended next step:

- Keep the LoRA text-only comparison as positive evidence.
- Improve or audit KG retrieval references before using KG-reference generation as a final claim.
- If time is limited, report KG retrieval as an interpretable retrieval component, and avoid overclaiming that it improves image-level generation metrics over TF-IDF in the current run.
