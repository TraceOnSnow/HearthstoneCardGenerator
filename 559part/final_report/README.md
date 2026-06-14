# COMP 559 Final Report Package

Main PDF:

- `report.pdf`

Main source:

- `report.tex`
- `references.bib`
- `neurips_2020.sty`
- `figures/`

Key local experiment outputs used in the report:

- `../../results/graphml_559/table_node2vec_retrieval.md`
- `../../results/graphml_559/table_link_prediction.md`
- `../../results/diy_retrieval_design_eval_real_llm/table_retrieval_metrics.md`
- `../../results/diy_generation_eval_real/table_generation_metrics.md`

Main 559 numbers:

- Semantic KG retrieval overall@5: `0.8061`
- Node2Vec retrieval overall@5: `0.7002`
- TF-IDF retrieval overall@5: `0.6788`
- CLIP retrieval overall@5: `0.4557`
- Node2Vec held-out edge prediction ROC-AUC: `0.8389`
- Node2Vec held-out edge prediction average precision: `0.8579`

Build command:

```bash
pdflatex -interaction=nonstopmode report.tex
bibtex report
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```
