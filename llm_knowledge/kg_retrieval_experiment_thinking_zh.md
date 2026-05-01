# KG Retrieval 实验设计思考

更新时间：2026-04-30

这份文档整理我们刚才关于 KG、LLM parser、CLIP baseline 的讨论，目的是避免最终报告把功劳讲混。

## 核心问题

如果系统流程写成：

```text
自然语言
-> LLM 转 structured query
-> 按关键词检索
```

那别人会质疑：

```text
这到底是 KG 的功劳，还是 LLM 抽关键词的功劳？
```

这个质疑是合理的。所以最终实验不能只展示 `LLM parser + keyword matching`。

## 正确拆法

应该把系统拆成两个不同问题：

```text
问题 1：KG retrieval 本身有没有用？
问题 2：LLM parser 能不能把用户自然语言接到 KG retrieval？
```

报告里的主要实验应该优先回答问题 1。

## KG retrieval 实验怎么做

主实验不要先依赖 LLM parser，而是使用固定的 structured query 文件。

示例：

```json
{
  "query_id": "dark_damage_healing",
  "text": "A dark spell that hurts an enemy but also heals its caster",
  "classes": ["Warlock"],
  "card_types": ["Spell"],
  "keywords": ["Lifesteal"],
  "actions": ["deal_damage"],
  "targets": ["minion"],
  "spell_schools": ["Fel"],
  "mechanic_tags": ["lifesteal_damage"],
  "visual_tags": ["fel magic"]
}
```

这样做的意义是：

```text
在语义需求已经明确的情况下，KG 是否能比 baseline 找到更符合机制的参考图？
```

这样不会把 LLM parser 的错误和 KG retrieval 的错误混在一起。

## LLM parser 的位置

LLM parser 应该作为 user-side demo 或可选模块：

```text
用户自然语言
-> LLM parser
-> structured query
-> KG retrieval
```

它说明系统可以支持自然语言输入，但不是 KG retrieval 的主要证据。

## 为什么 KG 不只是关键词检索

如果 KG 只做：

```text
query action=deal_damage
找所有 action=deal_damage 的卡
```

那确实和关键词检索差不多。

KG 要体现价值，需要用这些图结构信息：

```text
共享多个语义节点
动作-目标-资源组合
父子卡关系 HAS_CHILD_CARD
root collectible 聚合
derived card semantics
可解释 reasons
```

例如 query：

```text
A card whose full effect depends on generated follow-up cards
```

普通文本/CLIP 很难知道一张卡有没有 child cards。KG 可以直接利用：

```text
HAS_CHILD_CARD
HAS_ROOT_COLLECTIBLE
mechanic:has_derived_cards
```

这就是 KG 的优势。

## Baseline 应该怎么定义

至少有两个 baseline 概念，不能混用名字。

文本 baseline：

```text
natural language query
-> TF-IDF/BM25 over card name + text + caption
-> top-k cards
```

CLIP nearest-neighbor baseline：

```text
query text -> CLIP text embedding
art image -> CLIP image embedding
cosine similarity -> top-k images
```

教授反馈里说的 CLIP nearest-neighbor 更接近第二个，也就是多模态检索。

## Query 设计不要太直白

如果 query 是：

```text
Warlock spell that deals damage and has Lifesteal
```

那 CLIP/text baseline 也可能表现不错，因为关键词都明说了。

更好的 query 应该考机制理解、隐含语义和衍生卡关系：

```text
A dark spell that hurts an enemy but also heals its caster
A card that fills the board with small allies
A creature that does something useful after it dies
A defensive card that helps the hero survive by increasing protection
A card whose full effect depends on generated follow-up cards
A card that creates future choices instead of immediate damage
```

目标不是证明 KG 对所有 query 都赢，而是证明在需要机制理解、动作目标理解、衍生卡理解的 query 上，KG 更可靠。

## 报告里的表述

不要写：

```text
KG replaces CLIP.
```

应该写：

```text
CLIP retrieves visually similar references, while KG retrieves mechanically grounded references using structured card semantics, including actions, targets, resources, and derived-card relations.
```

最核心的一句话：

```text
LLM parser 是自然语言入口；KG retrieval 的贡献是用结构化语义图实现机制对齐、衍生卡关系推理和可解释 reference retrieval。
```

