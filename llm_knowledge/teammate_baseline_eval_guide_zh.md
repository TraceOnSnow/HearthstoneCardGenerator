# 给队友的任务说明：Query 文件、人工评分、Baseline 检索

更新时间：2026-04-30

这份文档是给不熟悉项目代码的队友看的。你可以把它直接交给你的 coding agent，让 agent 按步骤完成。你的任务不是改核心 KG 算法，而是把实验材料和 baseline evaluation 做完整。

## 一句话背景

我们这个项目想做的是：

```text
Hearthstone 卡牌 JSON
-> 结构化语义
-> 知识图谱 KG
-> 检索相关参考卡图
-> 给 LoRA / Stable Diffusion 生成更好的 Hearthstone 风格图
```

现在核心数据已经准备好了。每张卡已经被转成结构化语义，比如：

```text
职业：Warlock
类型：Spell
关键词：Lifesteal
动作：deal_damage
目标：minion
图片：images/VAC_951.jpg
caption：Hearthstone card art, Warlock, Fel spell, Lifesteal...
```

老师要求我们证明一件事：

```text
KG retrieval 比简单 baseline 更会找相关参考图。
```

所以你要做的是：

```text
固定测试 query
-> 跑一个简单 baseline retrieval
-> 生成评分表
-> 整理结果给报告使用
```

## 你不需要做什么

你不需要：

- 训练 LoRA。
- 改 KG schema。
- 改 structured semantics。
- 调 MiniMax / OpenAI / Gemini。
- 改图片下载脚本。
- 做复杂模型。

你只负责实验侧的基础设施。

## 你需要知道的数据文件

核心输入文件：

```text
data/semantics/cards_semantics_base.jsonl
data/semantics/lora_captions.jsonl
data/hf_hearthstone_art_512/metadata.jsonl
```

其中最重要的是：

```text
data/semantics/lora_captions.jsonl
```

它每一行是一张可以训练/检索的真实原画样本，大概长这样：

```json
{
  "card_id": 2539,
  "slug": "2539-flame-lance",
  "name": "Flame Lance",
  "collectible": true,
  "root_collectible_ids": [2539],
  "image": "images/AT_001.jpg",
  "caption": "Hearthstone card art, Mage, Fire spell, deal damage minion..."
}
```

你的 baseline retrieval 可以直接基于这个文件做。

## 任务 1：写固定 query 文件

新增文件：

```text
configs/retrieval_queries.json
```

如果没有 `configs/` 目录，就创建它。

文件格式用 JSON list：

```json
[
  {
    "query_id": "warlock_lifesteal_damage",
    "text": "Warlock spell that deals damage and has Lifesteal",
    "classes": ["Warlock"],
    "card_types": ["Spell"],
    "keywords": ["Lifesteal"],
    "actions": ["deal_damage"],
    "targets": ["minion"],
    "spell_schools": ["Fel"],
    "mechanic_tags": ["lifesteal_damage"],
    "visual_tags": ["fel magic"]
  }
]
```

至少写 8 条，最好 10 条。

推荐 query：

```text
warlock_lifesteal_damage
Warlock spell that deals damage and has Lifesteal

paladin_summon_tokens
Paladin card that summons minions

warrior_gain_armor
Warrior card that gains Armor

mage_freeze_spell
Mage spell that freezes enemies

deathrattle_summon_minion
Deathrattle minion that summons another minion

priest_healing_card
Priest card that restores Health

shaman_elemental_spell
Shaman card with elemental or Nature magic

hunter_beast_synergy
Hunter card that supports Beasts

rogue_weapon_or_attack
Rogue card about weapons or attacking

druid_nature_buff
Druid Nature card that buffs friendly minions
```

注意：

- `query_id` 用 snake_case。
- `text` 是自然语言 query。
- 结构化字段可以不全填，但 `classes/actions/keywords/card_types` 尽量填。
- 不要写太抽象的 query，比如 “cool dark card”，不好评分。

## 任务 2：实现 baseline retrieval

新增脚本：

```text
scripts/run_baseline_retrieval.py
```

目标：

```text
输入 configs/retrieval_queries.json
读取 data/semantics/lora_captions.jsonl
输出 baseline top-k 检索结果
```

先做最简单版本：TF-IDF 或 BM25。

如果你不想加新依赖，直接用 Python 标准库做一个简单词袋 baseline：

1. 把每张卡的 `name + caption` 合成文档。
2. 把 query 的 `text + classes + actions + keywords + visual_tags` 合成查询文本。
3. 分词、小写。
4. 用词重叠或 TF-IDF 分数排序。
5. 每个 query 取 top-5。

输出文件：

```text
results/retrieval_eval/baseline_results.jsonl
```

每行格式：

```json
{
  "query_id": "warlock_lifesteal_damage",
  "method": "tfidf_baseline",
  "rank": 1,
  "card_id": 107923,
  "card_name": "\"Health\" Drink",
  "image": "images/VAC_951.jpg",
  "score": 12.34,
  "query_text": "Warlock spell that deals damage and has Lifesteal",
  "caption": "Hearthstone card art, Warlock, Fel spell, Lifesteal..."
}
```

命令建议：

```bash
python3 scripts/run_baseline_retrieval.py \
  --queries configs/retrieval_queries.json \
  --captions data/semantics/lora_captions.jsonl \
  --out results/retrieval_eval/baseline_results.jsonl \
  --top-k 5
```

## 任务 3：生成人工评分模板

新增脚本：

```text
scripts/make_judging_template.py
```

输入：

```text
results/retrieval_eval/baseline_results.jsonl
```

以后 KG retrieval 做好后，也要支持：

```text
results/retrieval_eval/kg_results.jsonl
```

输出：

```text
results/retrieval_eval/judging_template.csv
```

CSV 字段：

```text
query_id
method
rank
card_id
card_name
image
query_text
class_match
action_match
keyword_match
overall_relevance
comment
```

其中这些字段先留空，人工之后填写：

```text
class_match
action_match
keyword_match
overall_relevance
comment
```

评分规则：

```text
class_match: 0 或 1，职业是否符合 query
action_match: 0 或 1，动作是否符合 query，比如 deal_damage/summon/heal
keyword_match: 0 或 1，关键词是否符合 query，比如 Lifesteal/Taunt/Deathrattle
overall_relevance: 0/1/2，整体相关性
comment: 可选说明
```

整体相关性解释：

```text
0 = 基本不相关
1 = 部分相关
2 = 很相关
```

人工评分不是训练模型，只是为了最终报告有数字表。

## 任务 4：生成 summary 表

新增脚本：

```text
scripts/summarize_judging.py
```

输入：

```text
results/retrieval_eval/judging_template.csv
```

人工填完分后，把它汇总成：

```text
results/retrieval_eval/summary.csv
```

输出表大概这样：

```text
method,class_match_at_5,action_match_at_5,keyword_match_at_5,overall_relevance_at_5
tfidf_baseline,0.52,0.40,0.31,0.86
kg_retrieval,0.78,0.69,0.55,1.34
```

一开始只有 baseline 没关系。等 KG retrieval 做好，再把 KG 的结果合并进去。

## 任务 5：生成简单 HTML 展示

新增脚本：

```text
scripts/render_retrieval_grid.py
```

输入：

```text
results/retrieval_eval/baseline_results.jsonl
results/retrieval_eval/kg_results.jsonl
```

输出：

```text
results/retrieval_eval/retrieval_grid.html
```

展示内容：

```text
每个 query 一块
左边 baseline top-5
右边 KG top-5
每张卡显示 card_name、image、score、reasons
```

注意：HTML 可以先只支持 baseline，KG 做好后再补。不要为了视觉效果花太多时间。

## 推荐文件结构

你最终应该新增这些文件：

```text
configs/retrieval_queries.json
scripts/run_baseline_retrieval.py
scripts/make_judging_template.py
scripts/summarize_judging.py
scripts/render_retrieval_grid.py
```

运行后生成这些文件：

```text
results/retrieval_eval/baseline_results.jsonl
results/retrieval_eval/judging_template.csv
results/retrieval_eval/summary.csv
results/retrieval_eval/retrieval_grid.html
```

`results/` 目录一般不要提交到 Git，除非团队决定把最终报告结果也提交。

## 可以让 coding agent 怎么帮你

你可以直接把下面的话发给你的 coding agent：

```text
Please implement the baseline retrieval/evaluation utilities for this Hearthstone project.

Context:
- Use data/semantics/lora_captions.jsonl as the retrieval corpus.
- Add configs/retrieval_queries.json with 10 fixed structured queries.
- Implement a simple dependency-free TF-IDF or token-overlap baseline retrieval.
- Output results/retrieval_eval/baseline_results.jsonl with query_id, method, rank, card_id, card_name, image, score, query_text, caption.
- Add scripts/make_judging_template.py to create a CSV for human judging.
- Add scripts/summarize_judging.py to aggregate filled judging scores by method.
- Add scripts/render_retrieval_grid.py to render retrieval results into a simple HTML grid.
- Do not modify KG schema, LoRA training scripts, image data, or structured semantics.
- Add small tests if practical, but prioritize runnable scripts and clear output files.
```

如果 agent 问你 baseline 用什么，回答：

```text
Use a simple TF-IDF or token-overlap baseline over card name + caption. No new heavy dependencies.
```

如果 agent 问你 KG 结果在哪里，回答：

```text
KG retrieval is not ready yet. Make scripts work with baseline_results.jsonl now and optionally accept kg_results.jsonl later.
```

## 完成标准

你这部分算完成，需要满足：

```text
python3 scripts/run_baseline_retrieval.py --queries configs/retrieval_queries.json --captions data/semantics/lora_captions.jsonl --out results/retrieval_eval/baseline_results.jsonl --top-k 5
```

能跑通，并生成 baseline results。

然后：

```text
python3 scripts/make_judging_template.py --inputs results/retrieval_eval/baseline_results.jsonl --out results/retrieval_eval/judging_template.csv
```

能生成评分模板。

再然后：

```text
python3 scripts/render_retrieval_grid.py --inputs results/retrieval_eval/baseline_results.jsonl --out results/retrieval_eval/retrieval_grid.html
```

能生成 HTML 展示。

最后人工填完 `judging_template.csv` 后：

```text
python3 scripts/summarize_judging.py --input results/retrieval_eval/judging_template.csv --out results/retrieval_eval/summary.csv
```

能生成 summary。

## 最后提醒

你的目标不是做一个很聪明的 baseline。你的目标是做一个清楚、可复现、能和 KG retrieval 对比的 baseline。

最终报告要回答：

```text
KG retrieval 是否比简单 baseline 更会找相关 Hearthstone reference？
```

你做的 query、baseline、评分表，就是为了回答这个问题。

