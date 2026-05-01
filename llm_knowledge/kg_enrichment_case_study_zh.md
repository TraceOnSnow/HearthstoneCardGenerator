# KG LLM Enrichment Case Study 与当前实现说明

## 背景

我们现在把项目拆成三层：

1. `structured semantics`：从 Hearthstone raw JSON 生成的结构化语义，服务 LoRA caption 和 KG。
2. `LLM enrichment`：只补规则难以稳定抽到的复杂玩法关系。
3. `semantic KG`：从结构化语义派生的可检索知识图谱。

重要边界：KG 不再使用 `visual_tags` 或 `lora_caption`。这些字段是给图像生成/LoRA 用的，不应该作为知识图谱检索事实，否则 KG 会混入二次视觉推断。

## Case Study 发现的问题

随机检查和手动检查中暴露了这些 bad cases：

- `Extraterrestrial Egg`：文本是 Deathrattle 召唤一个会攻击最低血量敌人的 Beast。规则只能抽到 `summon`，但应该进一步补 token/child 关系、`attack` 行为、`lowest_health_enemy` target。
- `Forensic Duster`：文本是让对手随从下回合费用增加。规则漏掉了 `increase_cost`，应该补 `target=opponent_minions`、`resource=cost`、`duration=next_turn`、`mechanic_tags=cost_disruption/minion_tax`。
- `Sludge Belcher`：规则能抽到 Deathrattle summon，但不知道 child card 是 `Slime` token。LLM 应该用输入里的 child card details 标注 `deathrattle_token/summoned_token`。
- `Sleep Paralysis`：Choose One 卡同时有召唤和摧毁分支，还有限制 `can't attack`。需要 `action_groups` 表达分支，`constraints` 表达限制。
- `Kiri, Chosen of Elune`：会给玩家 Solar Eclipse 和 Lunar Eclipse。规则能看到 childIds，但不知道“生成两张命名法术”这一语义。LLM 应补 `generated_card_refs`。
- `Uluu, the Everdrifter`：每回合在手牌中获得 Choose One choices。规则很难抽到 `while_in_hand` 和 `accumulating_choices`。

## Prompt 已优化的输出字段

LLM enrichment prompt 现在要求输出：

- `actions`：补全或修正动作，例如 `increase_cost`、`attack`、`add_to_hand`。
- `action_groups`：表达 Choose One、条件分支、顺序效果、重复效果。
- `mechanic_tags`：纯 gameplay/search 标签，不能放视觉描述。
- `constraints`：例如 `cannot_attack`、`next_turn_only`、`while_in_hand`、`enemy_only`。
- `generated_card_refs`：这张卡生成、加入手牌、召唤、变形、装备或施放的具体卡。
- `derived_cards`：只允许引用 input 里已有的 child card IDs，用来给 child 关系补 role。
- `related_card_refs`：文本里提到但没有可靠 ID 的命名卡。
- `semantic_summary`：一句话概括玩法语义。
- `generation_hints.visual_tags`：只给 LoRA 用，不进入 KG 检索。

## 当前代码变化

- `app/semantics/prompts/enrich_semantics_prompt.md`：加入 case-study-driven schema 和规则。
- `app/semantics/enrichment.py`：prompt 输入现在会包含 child card 的 name/text/stats/actions，方便 LLM 对齐 token/option/child roles。
- `app/semantic_kg/build.py`：KG 支持 `constraints`、`action_groups`、`generated_card_refs`、`related_card_refs`；不再写入 visual nodes 或 caption。
- `app/semantic_kg/retrieval.py`：检索不再使用 `visual_tags` 或 `caption`，只用 gameplay semantics 和 card text。
- `app/semantic_kg/query_parser.py`：自然语言 query parser 把视觉描述放入 `generation_hints.visual_tags`，不作为 KG 检索字段。
- `scripts/visualize_semantic_kg.py`：可视化节点类型更新，移除 visual node。

## 当前验证

已跑：

```bash
python3 -m unittest discover -s tests
```

结果：17 tests passed。

已重建 KG：

```bash
python3 scripts/build_semantic_kg.py \
  --semantics data/semantics/cards_semantics_base.jsonl \
  --out-dir data/semantic_kg
```

当前 KG 规模：

- cards: 8661
- nodes: 10938
- edges: 95976
- card_index: 8661

已确认主 KG 和新 sample HTML 中没有 `HAS_VISUAL_TAG`、`visual:`、`caption`。

已跑 retrieval smoke：

```bash
python3 scripts/run_kg_retrieval.py \
  --query-text 'Warlock spell that deals damage to a minion and has Lifesteal' \
  --query-id warlock_lifesteal_damage \
  --out results/kg_retrieval/kg_results_lifesteal_clean.jsonl \
  --top-k 5
```

Top results 包括 `"Health" Drink`、`Drain Soul`、`Lesser Amethyst Spellstone`，理由只来自 class/type/keyword/action/target/mechanic/text overlap。

## 接下来怎么用

如果要真正获得 LLM 增强后的 KG：

1. 跑 LLM enrichment，生成 `cards_semantics_enriched.jsonl`。
2. 用 enriched semantics 重建 KG。
3. 用 structured query 或 LLM query parser 跑 retrieval。

建议先跑 200 张重点卡，不要一上来全量：

```bash
python3 scripts/enrich_semantics.py \
  --semantics data/semantics/cards_semantics_base.jsonl \
  --out-dir data/semantics_enriched_200 \
  --limit 200 \
  --chunk-size 8 \
  --provider minimax

python3 scripts/build_semantic_kg.py \
  --semantics data/semantics_enriched_200/cards_semantics_enriched.jsonl \
  --out-dir data/semantic_kg_enriched_200
```

建议 batch size：

- `chunk-size=5`：最稳，适合复杂卡/调 prompt。
- `chunk-size=8`：推荐默认，平衡成本和上下文长度。
- `chunk-size=12-20`：只适合稳定后批量跑，失败时重跑成本更高。

