# 最终实验设计：LoRA、Reference、KG Retrieval 的证据链

这份文档记录 2026-05-01 讨论后确定的最终实验逻辑。目标不是再扩展功能，而是在报告里清楚证明两件事：

1. LoRA 和 reference image conditioning 对 Hearthstone artwork generation 有必要。
2. KG retrieval 相比简单 CLIP nearest-neighbor baseline 不是复杂度堆砌，而是能带来更相关的参考图和更好的生成结果。

## 总体故事

最终系统可以写成：

```text
User natural language request
-> LLM/rule parser converts it into a structured retrieval query
-> TF-IDF / CLIP / KG retrieve reference Hearthstone artworks
-> SD base / LoRA generate images with or without references
-> qualitative + quantitative evaluation
```

报告里不要只说“我们有 KG”和“我们有 LoRA”。要把它们放进同一个闭环：

- LoRA 解决风格问题：生成图更像 Hearthstone art。
- Reference image 解决具体视觉对齐问题：生成图更贴近检索到的真实卡牌风格和主题。
- KG retrieval 解决参考图质量问题：比 CLIP/TF-IDF 更容易找出机制上真正相关的卡，尤其是衍生卡、生成卡、隐式机制关系。

## 证据链 1：证明 LoRA 和 Reference 有必要

同一批 prompts，做四列生成对比：

```text
Prompt
| SD base, text-only
| SD base, text + reference
| LoRA, text-only
| LoRA, text + reference
```

这回答两个问题：

- LoRA 是否让图更像 Hearthstone 风格？
- Reference image 是否让生成更贴近具体语义、职业、机制、视觉主题？

### 需要的输入

- 一批固定 prompts，建议 6-10 个。
- 每个 prompt 最好对应一个 retrieval query，这样能复用 KG/CLIP 检索结果。
- 每个 prompt 有一个 reference image，后续可以分别用 KG reference 和 CLIP reference 做对比。

### 定性分析

报告里放 side-by-side figure。每一行一个 prompt，每列一种生成方法。

人工解释重点：

- `SD base, text-only` 是否风格不够像 Hearthstone。
- `SD base + reference` 是否主题更接近，但风格仍可能不稳定。
- `LoRA, text-only` 是否 Hearthstone 风格更强，但具体语义可能漂移。
- `LoRA + reference` 是否同时有 Hearthstone 风格和 reference 对齐。

### 定量分析

最稳的是人工评分，模型评分可以作为辅助。

人工评分建议字段：

```text
prompt_alignment: 1-5
hearthstone_style: 1-5
reference_consistency: 1-5
overall_quality: 1-5
notes
```

可以补充的自动指标：

- CLIP text-image similarity：prompt 和生成图是否匹配。
- CLIP/DINO image similarity：生成图和 reference image 是否接近。
- 多模态 LLM judge：让模型看 prompt + image 打分，但这只能作为辅助证据。

## 证据链 2：证明 KG Retrieval 有必要

同一段自然语言需求，先转成同一个 structured query，然后三种 retrieval 方法都用它去找 reference images：

```text
Natural language prompt
-> structured retrieval query
-> TF-IDF baseline
-> CLIP nearest-neighbor baseline
-> KG retrieval
```

这样比较才公平。不是 KG 用结构化 query、CLIP 用另一个 prompt，而是从同一个用户需求出发。

### 为什么不能只用简单 query

如果 query 是：

```text
Warlock spell that deals damage and has Lifesteal
```

那么 TF-IDF、CLIP、KG 可能都能做得还可以，因为关键词非常显式。这种例子证明不了 KG 的价值。

应该重点使用包含隐式关系和衍生卡关系的 query，例如：

```text
Druid card that gives or generates Solar Eclipse and Lunar Eclipse.
```

```text
Deathrattle minion that summons another Taunt minion token.
```

```text
Warlock spell that damages a minion and discounts your next Deathrattle minion if the target dies.
```

这些 query 的关键不是字面相似，而是结构化语义：

- generated_card_names
- generated_roles
- related_card_names
- conditions
- triggers
- mechanic_tags
- child / derived card relations

这正是 KG 应该比 CLIP/TF-IDF 更强的地方。

### 检索结果评估

先评估 retrieval 本身，不急着生成图。

对每个 query 的 top-5 retrieved cards 做人工评分：

```text
class_match: 0/1
action_match: 0/1
keyword_match: 0/1
generated_card_match: 0/1
overall_relevance: 0/1/2
notes
```

最后汇总成表：

```text
Method | Class Match@5 | Action Match@5 | Generated-card Match@5 | Overall Relevance@5 | Hit@5
TF-IDF
CLIP
KG
```

### 生成结果评估

然后把三种 retrieval 方法找出的 reference image 分别送入 generation pipeline：

```text
Same prompt
| LoRA + TF-IDF reference
| LoRA + CLIP reference
| LoRA + KG reference
```

如果时间允许，也可以做：

```text
Same prompt
| SD base + CLIP reference
| SD base + KG reference
| LoRA + CLIP reference
| LoRA + KG reference
```

这能直接回答教授的问题：

> KG retrieval 是否让最终生成结果比简单 CLIP nearest-neighbor retrieval 更好？

## 当前已有能力

当前 repo 已经有：

- structured semantics pipeline
- semantic KG build pipeline
- natural language to structured query parser
- KG retrieval
- TF-IDF retrieval baseline
- CLIP text-to-image nearest-neighbor retrieval baseline
- retrieval judging template
- retrieval grid HTML visualization
- LoRA text-only inference smoke test

当前已经能生成这些本地结果：

```text
results/retrieval_eval/tfidf_results.jsonl
results/retrieval_eval/clip_results.jsonl
results/retrieval_eval/kg_results.jsonl
results/retrieval_eval/judging_template.csv
results/retrieval_eval/retrieval_grid.html
```

## 当前缺口

主要缺口是 generation comparison pipeline。

需要新增：

1. 固定生成 prompts 配置。
2. 一个脚本读取 retrieval results，选 top-1 reference image。
3. 同一 prompt 自动跑四列或多列生成。
4. 输出 side-by-side HTML / contact sheet。
5. 输出 generation judging template。

建议先做最小可跑版本：

```text
configs/generation_prompts.json
scripts/run_generation_comparison.py
scripts/render_generation_grid.py
scripts/make_generation_judging_template.py
```

## 最终报告中应该怎么写

报告里的实验部分建议拆成两个子实验：

### Experiment A: Generation Ablation

目的：证明 LoRA 和 reference image 的价值。

方法：

```text
SD base text-only
SD base + reference
LoRA text-only
LoRA + reference
```

输出：

- side-by-side generation figure
- generation human rating table

### Experiment B: Retrieval Ablation

目的：证明 KG retrieval 的价值。

方法：

```text
TF-IDF baseline
CLIP nearest-neighbor baseline
KG retrieval
```

输出：

- retrieval side-by-side figure
- retrieval relevance rating table
- 如果时间允许，LoRA + different references 的 generation comparison

## 最关键的取舍

如果时间非常紧，优先级是：

1. 完成 retrieval comparison，因为这是教授明确指出的 KG 风险。
2. 完成 LoRA vs SD base 的 text-only 生成对比，证明 LoRA 有用。
3. 完成 LoRA + KG reference vs LoRA + CLIP reference，证明 KG reference 有用。
4. 自动指标最后做，人工评分表和 side-by-side 图优先。

人工评分不是弱点。对于课程项目，它是最稳、最容易解释、最不容易被环境问题卡住的量化证据。
