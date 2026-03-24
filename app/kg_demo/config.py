from pathlib import Path

# Data
SOURCE_JSONL = Path("data/cards_collectible.jsonl")
WORK_DIR = Path("data/mvp_kg_demo")
SAMPLED_JSONL = WORK_DIR / "sampled_1000_cards.jsonl"
PROMPTS_JSONL = WORK_DIR / "prompts_50_cards.jsonl"
LLM_OUTPUT_JSONL = WORK_DIR / "llm_outputs.jsonl"
GRAPH_JSON = WORK_DIR / "graph.json"

# Sampling and batching
SAMPLE_SIZE = 100
CHUNK_SIZE = 50
RANDOM_SEED = 42

# LLM
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
GOOGLE_MODEL = "gemini-2.5-flash-lite"
LLM_TEMPERATURE = 0.1
DRY_RUN = False  # Set False to call Google AI Studio.

# Prompt template
PROMPT_TEMPLATE_PATH = Path("app/kg_demo/prompts/kg_entity_extraction_prompt.md")
