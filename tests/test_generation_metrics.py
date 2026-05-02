from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.generation.metrics import collect_style_reference_paths, summarize_metric_rows, write_jsonl


class GenerationMetricsTest(unittest.TestCase):
    def test_summarize_metric_rows_groups_by_method(self) -> None:
        rows = [
            {"method": "sd_text_only", "clip_prompt_alignment": 0.2, "style_similarity": 0.4},
            {"method": "sd_text_only", "clip_prompt_alignment": 0.4, "style_similarity": 0.6},
            {"method": "lora_text_only", "clip_prompt_alignment": 0.5, "style_similarity": 0.9},
        ]

        summary = summarize_metric_rows(rows)

        self.assertEqual(summary[0]["method"], "lora_text_only")
        self.assertEqual(summary[0]["style_similarity_mean"], 0.9)
        self.assertEqual(summary[1]["clip_prompt_alignment_mean"], 0.3)

    def test_collect_style_reference_paths_reads_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "images" / "card.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"fake")
            metadata = root / "metadata.jsonl"
            write_jsonl(metadata, [{"file_name": "images/card.jpg"}, {"file_name": "images/missing.jpg"}])

            paths = collect_style_reference_paths(metadata_path=metadata, image_root=root, limit=10)

        self.assertEqual(paths, [image])


if __name__ == "__main__":
    unittest.main()
