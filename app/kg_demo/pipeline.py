from app.kg.models import KgRunConfig
from app.kg.pipeline import run_pipeline as run_kg_pipeline

from app.kg_demo import config


def run_pipeline() -> None:
    """Compatibility wrapper for the old demo entrypoint."""
    stats = run_kg_pipeline(
        KgRunConfig(
            source_jsonl=str(config.SOURCE_JSONL),
            out_dir=str(config.WORK_DIR),
            metadata_json=str(config.METADATA_JSON),
            prompt_template=str(config.PROMPT_TEMPLATE_PATH),
            limit=None,
            sample_size=config.SAMPLE_SIZE,
            random_seed=config.RANDOM_SEED,
            chunk_size=config.CHUNK_SIZE,
            dry_run=config.DRY_RUN,
            provider="google",
            model=config.GOOGLE_MODEL,
            temperature=config.LLM_TEMPERATURE,
            timeout_seconds=60,
            resume=True,
            force_llm=False,
            visualize=False,
        )
    )
    print(
        f"Done. cards={stats['cards']} batches={stats['batches']} "
        f"nodes={stats['nodes']} edges={stats['edges']} dry_run={config.DRY_RUN}"
    )
