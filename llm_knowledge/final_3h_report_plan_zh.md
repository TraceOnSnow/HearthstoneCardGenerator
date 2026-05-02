# 3 小时最终报告执行计划

时间点：2026-05-01 晚。目标是满足 COMP 646 final report 的硬性要求，不再扩展功能。

最重要参考：

- 教授要求：`llm_knowledge/requirement/COMP 646 — General advice for the final project report.html`
- 模板：`llm_knowledge/template/template.zip`
- MovieGen Figure 1 参考：`llm_knowledge/2410.13720v2.pdf`

## 一句话目标

最终报告不要写成“我们做了很多组件”。要写成一个闭环系统：

> HearthGen converts Hearthstone card data and user requests into structured semantic representations, builds a knowledge graph for mechanically grounded retrieval, and uses the retrieved references to guide Stable Diffusion / LoRA artwork generation.

主贡献不是“训练了最强图像模型”，而是：

> structured semantic KG retrieval provides better, more explainable Hearthstone references than plain text/CLIP retrieval, especially for mechanic-heavy or meme/family-based card design requests.

## 必须满足的老师要求

报告里必须出现：

1. 4 页 PDF，使用模板。
2. GitHub source code link 可打开。
3. Figure 1 是最重要图，必须用真实输入/输出，不要空框图。
4. 至少 6 个 qualitative examples。
5. 至少一个 quantitative table。
6. 有 baseline 对比。
7. 每个外部组件要引用：Stable Diffusion, LoRA, CLIP, DINOv2 if used, MiniMax, HearthstoneJSON/HearthSim.
8. 明确说明哪些是我们做的。
9. Acknowledgement 里说明 AI assistance.
10. 不要放命令行/Jupyter 截图。

## 报告标题

建议标题：

```text
HearthGen: Knowledge-Graph-Augmented Retrieval for Hearthstone Card Art Generation
```

## 4 页结构

### Abstract

4-5 句：

- Hearthstone card generation needs both visual style and game-mechanic consistency.
- We construct structured semantics from Hearthstone card JSON.
- We build a semantic KG and retrieve mechanically relevant references.
- We evaluate retrieval and generation against TF-IDF / CLIP baselines.
- Results show KG references improve semantic relevance and LoRA improves Hearthstone style.

### 1 Introduction

重点回答：

- 普通 text-to-image 模型不知道 Hearthstone card mechanics。
- Hearthstone 卡牌有隐式关系：generated cards, tokens, Rager meme family, Deathrattle, Battlecry, etc.
- 只用 CLIP visual similarity 容易找“看起来像”的图，不一定机制相关。
- 我们提出 structured semantics + KG retrieval + LoRA generation。

要写清楚 contribution：

```text
Our contributions are:
1. A structured semantic representation for Hearthstone cards.
2. A semantic KG and retrieval pipeline for card-design evidence.
3. A generation/evaluation pipeline comparing SD, LoRA, TF-IDF/CLIP references, and KG references.
```

### 2 Method

分 3 段。

#### Structured Semantics

输入：

```text
cards_all.jsonl / cards_collectible.jsonl
```

输出字段：

```text
identity, stats, keywords, actions, mechanic_tags, generated_card_refs, derived_cards, lora_caption
```

规则抽取负责稳定字段：

- class/type/set/rarity
- mana/attack/health
- keywordIds
- simple action regex: deal damage, summon, draw, gain armor, freeze, destroy
- childIds/derived cards

LLM enrichment 只作为增强：

- generated_card_refs
- derived card role
- implicit conditions/triggers
- semantic summary

#### Semantic KG Retrieval

KG nodes:

```text
card, class, card_type, keyword, action, target, resource, mechanic, generated_card_name, related_card_name
```

KG edges:

```text
HAS_CLASS, HAS_KEYWORD, PERFORMS_ACTION, AFFECTS_RESOURCE, GENERATES_CARD, HAS_CHILD_CARD
```

Retrieval:

- Natural language -> structured query
- Query fields match KG nodes
- Return not just one ranked card, but faceted evidence:
  - family/name anchors
  - mechanic matches
  - identity matches
  - diversified shortlist

这里可以用 Iron Rager 例子说明：

```text
Request: DIY a defensive Control Warrior Rager meme card.
KG retrieves Warrior armor cards and Rager-family anchors such as Magma Rager.
LLM designs Iron Rager: 3-mana 5/1, Battlecry: Gain 3 Armor.
```

#### Generation

Models:

- Stable Diffusion v1.5 base
- Hearthstone LoRA adapter
- Reference-conditioned img2img

Reference policy:

- use one primary visual reference
- use other KG evidence as text semantic hints
- high denoise strength if using img2img to avoid “fusion monster” artifacts

### 3 Experiments

Two experiments only.

#### Experiment A: Retrieval

Question:

```text
Does KG retrieval return more relevant card references than TF-IDF / CLIP?
```

Methods:

- TF-IDF baseline
- CLIP nearest-neighbor baseline
- Semantic KG retrieval

Inputs:

- 10 natural-language/structured card-design queries
- include both easy explicit queries and harder implicit/meme/family queries

Outputs:

- top-k retrieved cards
- retrieval grid
- human or proxy relevance scores

#### Experiment B: Generation

Question:

```text
Do LoRA and KG references improve Hearthstone artwork generation?
```

Methods:

- SD text-only
- LoRA text-only
- LoRA + TF-IDF reference
- LoRA + KG reference

Metrics:

- CLIP prompt alignment
- style similarity
- reference similarity
- human review if available

### 4 Results and Analysis

必须放两个表：

Table 1 retrieval:

```text
Method | Relevance@5 / human score | Notes
TF-IDF
CLIP
KG
```

Table 2 generation:

```text
Method | CLIP alignment ↑ | Style similarity ↑ | Reference similarity ↑
SD text-only
LoRA text-only
LoRA + TF-IDF ref
LoRA + KG ref
```

如果数字不完美，不要编。写成：

```text
Automatic metrics are proxy measures and do not fully capture card-design correctness; therefore we also provide qualitative analysis.
```

必须写失败：

- KG retrieval can over-rank repeated mechanic matches.
- We fixed this by returning faceted evidence rather than a single flat list.
- Strong img2img reference can create fusion artifacts.
- Future use should use one primary reference and semantic text hints.

### 5 Conclusion

2-4 句即可。

说：

- We built an end-to-end prototype.
- KG helps retrieve mechanically grounded references.
- LoRA improves style.
- Remaining limitation: evaluation scale and reference-conditioning artifacts.

### Acknowledgement

必须有：

```text
We acknowledge the use of AI coding assistants for implementation support and drafting assistance. All experiments, analysis, and final system design decisions were reviewed by the project team.
```

## Figure 1 设计：模仿 MovieGen，不要空框图

MovieGen Figure 1 的特点：

- 多行能力展示。
- 每行都有真实 prompt/reference/source/output。
- Figure 本身能一眼说明系统能做什么。
- 不是单纯 boxes and arrows。

我们的 Figure 1 应该做成 3 行能力图：

### Row 1: Structured KG Retrieval

左边：

```text
User request:
"Design a defensive Control Warrior Rager meme card..."
```

中间：

```text
Structured query:
class=Warrior
type=Minion
action=gain_armor
related=Magma Rager/Rager
```

右边：

```text
KG evidence:
Magma Rager
Hookfist-3000
Armorsmith
Drywhisker Armorer
```

### Row 2: KG-Augmented Card Design

左边：same request + KG evidence

右边：

```text
Iron Rager
3 Mana Warrior Minion
5/1
Battlecry: Gain 3 Armor.
```

### Row 3: Artwork Generation

Columns:

```text
Prompt
SD text-only
LoRA text-only
LoRA + TF-IDF reference
LoRA + KG reference
```

Use generated images from:

```text
results/final_generation_eval/
results/generation_eval/
results/generation_eval_clip_ref/
```

## Figure 2 设计

Figure 2 是 side-by-side qualitative generation:

Rows: 4-6 prompts.

Columns:

```text
Prompt
SD text-only
LoRA text-only
LoRA + TF-IDF ref
LoRA + KG ref
```

不要放 10 行，太小。选 4 行最清楚的。

候选 prompts:

- warlock_lifesteal_damage
- druid_solar_lunar_generation
- deathrattle_taunt_token
- freeze_enemy_mage_spell
- choose_one_demon_or_destroy
- rager meme if generated image exists

## Figure 3 or Table Only

如果版面不够，不做 Figure 3。用一个小表替代 retrieval qualitative。

可以放：

```text
Query: Defensive Control Warrior Rager meme

KG evidence:
Magma Rager -> statline/meme anchor
Hookfist-3000 -> armor mechanic
Armorsmith -> Warrior armor identity

Designed card:
Iron Rager, 3 mana 5/1, Battlecry: Gain 3 Armor.
```

这个例子很能说明 KG 不只是找图，而是支持 card design semantics。

## 可用素材清单

### Retrieval

```text
results/retrieval_eval/retrieval_grid.html
results/retrieval_eval/tfidf_results.jsonl
results/retrieval_eval/clip_results.jsonl
results/retrieval_eval/kg_results.jsonl
results/retrieval_eval/judging_template.csv
```

### Generation

```text
results/final_generation_eval/
results/generation_eval/
results/generation_eval_clip_ref/
results/generation_text_only_eval/
```

### KG card design

```text
results/kg_card_design_rager_en/parsed_query.json
results/kg_card_design_rager_en/retrieved_cards.jsonl
results/kg_card_design_rager_en/evidence_package.json
results/kg_card_design_rager_en/design.json
results/kg_card_design_rager_en/summary.md
```

## 引用清单

至少需要这些 bib entries:

- Stable Diffusion / Latent Diffusion Models
- LoRA
- CLIP
- DINOv2 if automatic metrics mention it
- Retrieval-Augmented Generation or GraphRAG / KG retrieval
- HearthstoneJSON / HearthSim as data source
- MiniMax M2.7 if used for semantic enrichment / card design

如果时间不够，先用 URL bib entries 也可以，比没有引用强。

## 最后三小时具体时间表

### 0:00-0:30

- 解压模板到 `submission/final_report/`
- 确定 `report.tex`
- 改标题、作者、section skeleton
- 准备 bib

### 0:30-1:15

- 生成 Figure 1 和 Figure 2
- 不要截图命令行
- 可以用 Python/PIL/matplotlib 拼图
- 导出 PDF 或高分辨率 PNG，再嵌入 LaTeX

### 1:15-2:00

- 填 Abstract / Introduction / Method
- 写清楚 exact models:
  - Stable Diffusion v1.5
  - MiniMax-M2.7
  - CLIP ViT-B/32
  - DINOv2 if used

### 2:00-2:30

- 填 Experiments / Results
- 放两个表
- 每个图写一段讨论

### 2:30-2:50

- 写 Related Work / Conclusion / Acknowledgement
- 插 references

### 2:50-3:00

- 编译 PDF
- 检查 4 页
- 检查图可读
- 检查 GitHub link
- 提交

## 最重要的取舍

不要再跑新实验。不要再全量 enrich。不要再调 LoRA。

如果只剩 30 分钟，优先保证：

1. 4 页 PDF 能编译。
2. 有 Figure 1。
3. 有 side-by-side generation Figure 2。
4. 有一个 quantitative table。
5. 有 references 和 GitHub link。

这些比额外提升结果质量更重要。
