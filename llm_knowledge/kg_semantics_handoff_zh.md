# HearthstoneCardGenerator 项目交接说明

更新时间：2026-04-30

这份文档给后续接力的同学和 agent 工具使用。当前项目重点已经从“先训练 LoRA”调整为“先把 Hearthstone 卡牌数据整理成可复现的数据、结构化语义、KG 检索和生成实验闭环”。

## 当前项目目标

最终报告最好聚焦成一个清晰故事：

HearthGen 是一个面向 Hearthstone 原画生成的检索增强系统。项目贡献不是单纯训练最强图像模型，而是把卡牌 JSON 转成结构化语义，再派生 KG 检索，用更相关的真实卡牌原画作为 reference，辅助后续 Stable Diffusion / LoRA / IP-Adapter 类生成流程。

教授反馈里最关键的问题是：不能只有 future work，必须有结果、图、baseline 和指标。所以接下来最重要的是做出：

- KG/semantic retrieval 比简单 baseline 更相关的证据。
- side-by-side qualitative figure。
- 一个 quantitative table。
- 可复现的代码命令。

## 已完成的数据资产

### Blizzard card JSON

保留在 Git 里的小型核心数据：

- `data/cards_all.jsonl`
- `data/cards_collectible.jsonl`
- `data/sample_img/**`

`cards_all.jsonl` 包含 collectible 和 non-collectible/derived cards。KG/structured semantics 应该优先使用全量 `cards_all.jsonl`，因为 Hearthstone 很多卡的完整语义藏在 token、Quest reward、follow-up spell、Titan ability 等衍生卡里。

### Hearthstone 原画数据

本地已有两个 art 目录，但不会提交进 Git：

- `data/hf_hearthstone_art_512/`
- `data/hs_art_512/`

这些目录已在 `.gitignore` 里显式忽略。不要把大量图片提交到 GitHub。

私有 Hugging Face dataset：

- `TraceOnSnow/hearthstone-art-512`
- 约 7889 张 512x512 HearthstoneJSON art-only JPG
- 包含 `metadata.jsonl`

关键对应关系：

```text
data/hf_hearthstone_art_512/metadata.jsonl 里的 dbf_id
==
data/cards_all.jsonl 里的 id
```

已经手动抽查确认：

- `dbf_id=2539` -> `Flame Lance` -> `images/AT_001.jpg`
- `dbf_id=107923` -> `"Health" Drink` -> `images/VAC_951.jpg`
- `dbf_id=115364` -> `Daydreaming Pixie` -> `images/EDR_530.jpg`

有些 art metadata 的 `dbf_id` 在当前 Blizzard API 的 `cards_all.jsonl` 中找不到，这是数据源覆盖范围差异，不是 join 逻辑错误。

## 已完成的代码模块

### KG 模块

目录：

- `app/kg/`
- `scripts/run_kg.py`
- `scripts/fetch_metadata.py`
- `tests/test_kg_pipeline.py`

功能：

- 从 card JSON 读入卡牌。
- 使用 metadata ID-name map 解析 class/type/set/rarity/spellSchool/keyword。
- 支持 Google 和 MiniMax LLM。
- 支持 dry-run。
- 支持 semantic LLM output 转 KG graph。
- 支持 graph visualization。

MiniMax 使用 OpenAI-compatible endpoint：

```text
https://api.minimax.io/v1/chat/completions
```

环境变量：

```bash
MINIMAX_API_KEY=...
```

### Structured Semantics 模块

目录：

- `app/semantics/`
- `scripts/build_semantics.py`
- `scripts/enrich_semantics.py`
- `tests/test_semantics_pipeline.py`

设计原则：

```text
Raw card JSON
-> rule-based structured semantics
-> optional LLM-enriched semantics
-> LoRA captions
-> KG entities/relations
```

第一层不依赖 LLM，必须全量、稳定、可复现。LLM 只做增强，不作为唯一数据来源。

规则版目前会生成：

- card identity：type/class/set/rarity/spell school/minion type/artist
- stats：mana/attack/health/durability
- clean text
- keywords
- actions
- mechanic tags
- visual tags
- parent/child card graph
- root collectible ids
- derivation depth
- expanded semantics
- LoRA caption

目前支持的常见 action 包括：

- `deal_damage`
- `heal`
- `gain_armor`
- `summon`
- `draw`
- `discover`
- `add_to_hand`
- `destroy`
- `freeze`
- `silence`
- `equip`

## 已跑通过的命令

拉 metadata：

```bash
uv run python scripts/fetch_metadata.py
```

构建全量 base semantics：

```bash
python3 scripts/build_semantics.py --out-dir data/semantics
```

当前输出结果：

```text
cards=11589
edges=9990
captions=6165
out_dir=data/semantics
```

其中：

- `cards_semantics_base.jsonl`：全量卡牌语义，包括无图卡和衍生卡。
- `derived_edges.jsonl`：parent/child card edges。
- `lora_captions.jsonl`：只包含有真实 art image 的样本。
- `summary.json`：统计信息。

LLM 增强 dry-run：

```bash
python3 scripts/enrich_semantics.py \
  --semantics data/semantics/cards_semantics_base.jsonl \
  --out-dir data/semantics_enriched_smoke \
  --limit 5 \
  --chunk-size 2 \
  --dry-run \
  --no-resume \
  --force-llm
```

真实 MiniMax 小样本增强：

```bash
MINIMAX_API_KEY=你的key python3 scripts/enrich_semantics.py \
  --semantics data/semantics/cards_semantics_base.jsonl \
  --out-dir data/semantics_enriched \
  --limit 200 \
  --chunk-size 10 \
  --provider minimax
```

测试：

```bash
python3 -m unittest discover -s tests
```

当前通过：

```text
Ran 11 tests
OK
```

## 当前 Git 状态注意事项

目前有多组未提交改动：

- `app/kg/`
- `app/semantics/`
- `scripts/build_semantics.py`
- `scripts/enrich_semantics.py`
- `scripts/fetch_metadata.py`
- `scripts/run_kg.py`
- `scripts/download_hs_art.py`
- `scripts/prepare_hf_art_dataset.py`
- `tests/`
- 以及一些早期 `kg_demo` / README / visualization 修改

不要直接 `git reset --hard`。需要提交时请按功能分 commit：

1. KG pipeline commit
2. semantics pipeline commit
3. art download/HF dataset helper commit
4. docs/handoff commit

## 接下来最应该做的工作

### 任务 1：把 structured semantics 转成 KG graph

当前 KG 还偏向 LLM extraction。更稳的路线是从 `cards_semantics_base.jsonl` 直接派生 KG：

```text
card -> HAS_CLASS -> class
card -> HAS_TYPE -> card_type
card -> HAS_KEYWORD -> keyword
card -> PERFORMS_ACTION -> action
card -> TARGETS -> target
card -> AFFECTS_RESOURCE -> resource
card -> HAS_MECHANIC_TAG -> mechanic
card -> HAS_VISUAL_TAG -> visual
card -> HAS_CHILD_CARD -> card
```

这样 KG 不依赖 LLM，也能覆盖全量卡。

建议新增：

- `app/semantics/to_kg.py`
- `scripts/build_semantic_kg.py`

### 任务 2：做 retrieval baseline 和 KG retrieval

为了回应教授反馈，必须比较 KG retrieval 和简单 baseline。

建议实现：

- `app/retrieval/text_retriever.py`
- `app/retrieval/kg_retriever.py`
- `scripts/run_retrieval_eval.py`

baseline 可以先用 TF-IDF，不要一开始就依赖 CLIP。KG retrieval 用 shared semantic nodes 打分。

推荐打分：

```text
score =
3 * shared_action
+ 2 * shared_keyword
+ 2 * shared_class
+ 1 * shared_resource
+ 1 * shared_target
+ 1 * shared_mechanic_tag
+ 1 * shared_visual_tag
```

### 任务 3：固定 8-10 个 query prompts

示例：

- Warlock spell that deals damage and has Lifesteal
- Paladin card that summons tokens
- Warrior card that gains armor
- Mage spell that freezes enemies
- Deathrattle minion that summons another minion
- Demon Hunter card that attacks multiple enemies
- Priest card that heals friendly characters
- Shaman card using elemental or nature magic

每个 query 输出：

- TF-IDF top-5
- KG top-5
- card names
- image paths
- semantic reasons

### 任务 4：做报告用图和表

最重要的图：

```text
Query | TF-IDF retrieved refs | KG retrieved refs | KG reasons
```

最重要的表：

```text
Method | Class match@5 | Action match@5 | Overall relevance@5
TF-IDF
KG retrieval
```

如果来不及训练 LoRA，至少要把 KG retrieval 的证据做好。否则 KG 会被看成复杂度堆砌。

### 任务 5：生成侧最小闭环

如果图像生成队友能接上，建议最小实验矩阵：

```text
Prompt
Text-only SD
Text + TF-IDF reference
Text + KG reference
LoRA + KG reference, if ready
```

如果 LoRA 来不及，不要强行把 LoRA 写成核心结果。可以写成 dataset/preparation branch，把主要结果放在 KG retrieval improves reference relevance。

## 已知风险

- `cards_all.jsonl` 和 HearthstoneJSON art metadata 覆盖范围不完全一致。
- 当前 action regex 只能覆盖常见文本，复杂条件仍需要 LLM 增强。
- LLM 增强必须控制 schema，否则 KG node label 会变脏。
- 大图像数据不要进 GitHub。
- `data/semantics/` 是生成结果，当前被 `data/*` ignore，不会被 Git 跟踪。
- 最终报告必须有结果图和表，不要只写 pipeline。

## 给 agent 的优先级指令

如果时间很紧，按这个顺序做：

1. 不要改数据下载逻辑。
2. 基于 `data/semantics/cards_semantics_base.jsonl` 做 KG graph。
3. 做 KG retrieval vs TF-IDF retrieval。
4. 输出 `results/retrieval_eval/results.jsonl`、`summary.csv`、`retrieval_grid.png`。
5. 把图和表放进报告。
6. 只有在 retrieval 结果稳定后，再做 LLM enrichment 或 LoRA caption 优化。

