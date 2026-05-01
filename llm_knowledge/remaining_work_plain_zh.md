# 剩余工作人话版

更新时间：2026-04-30

这份文档解释项目现在还缺什么，以及为什么要做这些事情。目标是让队友和他们的 coding agent 能直接接力。

## 现在我们已经有什么

目前已经有：

- Hearthstone 卡牌 JSON 数据。
- Hearthstone 原画数据，放在 Hugging Face private dataset：`comp646/hearthstone-art-512`。
- 去掉 Battlegrounds、Mercenaries、英雄皮肤等特殊模式后的结构化语义。
- LoRA 训练用 metadata，已经放在 HF dataset 的 `lora/metadata.jsonl`。
- 接下来会有一个微调好的 LoRA。
- 接下来 KG 侧会有 node/edge 结构。

所以后面重点不是继续堆数据，而是证明系统真的有用。

## 老师到底要看什么

老师的反馈很明确：

1. 要看到生成结果。
2. 要看到同一批 prompt 的 side-by-side 对比。
3. 要证明 KG retrieval 比简单 CLIP nearest-neighbor baseline 更有用。
4. 要有定量结果，不能只有“看起来不错”。
5. 图要清楚，不能截图 notebook，不能字太小。
6. 要有引用，至少 cite LoRA、Stable Diffusion、CLIP、KG/retrieval 相关工作。

## 什么是人工评分

人工评分就是我们自己看检索结果，然后按固定规则打分。

例如 query 是：

```text
Warlock spell that deals damage and has Lifesteal
```

CLIP baseline 找 5 张参考图，KG retrieval 也找 5 张参考图。然后人看每张图对应的卡牌，填表：

```text
职业对不对：0/1
动作对不对：0/1
关键词对不对：0/1
整体相关性：0/1/2
```

最后算平均分。这样报告里可以有一张表：

```text
Method | Class match@5 | Action match@5 | Keyword match@5 | Overall relevance@5
CLIP baseline | ...
KG retrieval | ...
```

这就是定量证据。它不复杂，但能回答老师的问题：KG 到底有没有比 baseline 更好。

## 还要做的核心事情

### 1. 做 KG retrieval

输入一句需求：

```text
Warlock spell that deals damage and has Lifesteal
```

输出 KG 找到的 top-5 参考卡：

```text
"Health" Drink
reason: class=Warlock, keyword=Lifesteal, action=deal_damage, target=minion
```

这一步是 KG 侧最重要的功能。

### 2. 做 CLIP baseline

同样的 query，用 CLIP nearest-neighbor 找 top-5 图片。

它是对照组。没有 baseline，就没法证明 KG 有价值。

### 3. 固定 6-10 个测试 prompt

不要临时挑好看的例子。提前写死一批：

```text
Warlock Lifesteal damage spell
Paladin card that summons minions
Warrior armor gain card
Mage freeze spell
Deathrattle minion that summons another minion
Priest healing card
Shaman elemental or nature spell
Hunter beast synergy card
```

每个 prompt 都跑 CLIP baseline 和 KG retrieval。

### 4. 做 retrieval 对比结果

每个 prompt 输出：

```text
CLIP top-5
KG top-5
KG reasons
```

需要生成：

```text
results/retrieval_eval/results.jsonl
results/retrieval_eval/summary.csv
results/retrieval_eval/retrieval_grid.png
```

### 5. LoRA 队友做生成对比图

至少要有同一批 prompt 的 side-by-side：

```text
text-only
text + CLIP reference
LoRA + KG reference
```

如果时间够，最好加第四列：

```text
LoRA + CLIP reference
```

这张图是最终报告最重要的 qualitative result。

### 6. 做定量表

最低限度做 retrieval 定量表：

```text
CLIP baseline vs KG retrieval
```

可以用人工评分。目标是证明 KG 找到的参考图更符合职业、动作、关键词和整体语义。

如果时间够，再加 generation 指标：

```text
CLIP text-image similarity
style similarity to Hearthstone art
```

但如果时间紧，先把 retrieval relevance 表做好。

### 7. 做最终报告图

至少需要三张图：

```text
Figure 1: system overview
Figure 2: retrieval comparison
Figure 3: generation comparison
```

Figure 1 内容：

```text
Card JSON + Art Dataset
-> Structured Semantics
-> KG
-> KG Retrieval
-> Reference Images / Prompt Augmentation
-> LoRA Generation
```

Figure 2 内容：

```text
Prompt | CLIP refs | KG refs | KG reasons
```

Figure 3 内容：

```text
Prompt | text-only | text+reference | LoRA+reference
```

## 最短总结

现在要证明两件事：

1. LoRA 真的能生成 Hearthstone 风格图。
2. KG 找 reference 比 CLIP baseline 更懂卡牌机制。

所以还缺：

```text
KG retrieval
CLIP baseline
固定 prompts
retrieval 评分表
LoRA 生成对比图
最终报告图表
```

如果时间很紧，优先做 KG retrieval vs CLIP baseline 的表和图。老师已经明确说，如果没有这个证据，KG 分支会显得只是为了复杂而复杂。

