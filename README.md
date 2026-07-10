# Paper Review Archive

A GitHub Pages site for archiving papers I study. Each paper is a single
Markdown file; the site lists them newest-first and lets you filter by tag.

**Live site:** https://jiyoon0630.github.io/paper-reivew/ *(after the one-time setup below)*

## One-time setup: turn on GitHub Pages

1. Merge this branch into `main` (or open a PR and merge it).
2. On GitHub, go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **GitHub Actions**.
4. Push to `main` (or run the *Build and deploy site* workflow manually).
   The site deploys automatically on every push to `main`.

## Adding a new paper

1. Create a file in `_posts/` named `YYYY-MM-DD-short-title.md`.
2. Copy the front matter from [`templates/paper-template.md`](templates/paper-template.md).
3. Commit and push to `main` — the site rebuilds automatically.

Front matter (only `title` and `date` are required):

```yaml
---
layout: paper
title: "Attention Is All You Need"
date: 2026-07-10
tags: [NLP, Transformer]   # click-to-filter labels
summary: "One-line description."
authors: "Vaswani et al."
venue: "NeurIPS 2017"
paper_url: "https://arxiv.org/abs/1706.03762"
code_url: "https://github.com/..."
---
```

- **Math:** `$...$` inline, `$$...$$` display (rendered with MathJax).
- **Code:** fenced code blocks are syntax-highlighted.

## Running locally (optional)

```bash
bundle install
bundle exec jekyll serve
# open http://localhost:4000/paper-reivew/
```
