---
layout: paper
title: "Welcome — how a paper note is formatted"
date: 2026-07-10
tags: [Meta]
summary: "A sample entry showing the front matter, tags, math, and code that every paper note can use."
authors: "jiyoon0630"
paper_url: "https://arxiv.org/abs/1706.03762"
code_url: "https://github.com/jiyoon0630/paper-reivew"
---

This first entry is a template you can copy. Delete it once you've added a few
real papers. Everything below is standard Markdown.

## Front matter

Every note starts with a YAML block. Only `title` and `date` are required:

```yaml
---
layout: paper
title: "Attention Is All You Need"
date: 2026-07-10          # drives the ordering (newest first)
tags: [NLP, Transformer]  # click-to-filter labels on the home page
summary: "One-line description shown in the list."   # optional
authors: "Vaswani et al."                             # optional
venue: "NeurIPS 2017"                                 # optional
paper_url: "https://arxiv.org/abs/1706.03762"         # optional
code_url: "https://github.com/..."                    # optional
---
```

## Math

Inline math like $a^2 + b^2 = c^2$ works, and so do display equations:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

## Code

Fenced code blocks are syntax-highlighted:

```python
def scaled_dot_product_attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
    weights = scores.softmax(dim=-1)
    return weights @ v
```

## The rest

Use normal Markdown for the write-up:

- **Problem** — what the paper tackles.
- **Method** — the key idea.
- **Results** — what actually moved.
- **My take** — why it matters / where it breaks.

> Tip: keep the filename as `YYYY-MM-DD-short-title.md` so the date sorts correctly.
