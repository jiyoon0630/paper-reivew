# paper-reivew — working notes for Claude

Personal archive of paper-study notes, published as a Jekyll **GitHub Pages**
site at https://jiyoon0630.github.io/paper-reivew/. The owner studies papers
with Claude; the final write-up is Markdown that Claude posts here.

The site UI is **English**, the home page defaults to the **English** paper
list, and papers can be **bilingual (Korean + English)**.

## Default job: post every uploaded Markdown in BOTH languages

When the owner uploads a Markdown paper note, the default task is to **publish
it as two paired posts — one Korean, one English** — unless they say otherwise.

Steps for one uploaded note:

1. Pick a shared key `<ref>` (kebab-case, e.g. `pld-self-improving-vla`) and a
   date `YYYY-MM-DD` (use the note's front-matter `date` if present).
2. Create two files in `_posts/`:
   - Korean: `YYYY-MM-DD-<ref>.md`  with `lang: ko`
   - English: `YYYY-MM-DD-<ref>-en.md` with `lang: en`
   Both carry the **same `ref`** so the KO ⇄ EN toggle links them automatically.
3. If the upload is in only one language, **translate it into the other**.
   Translate faithfully: preserve all math verbatim, translate the text inside
   tables and ASCII diagrams too, and keep the callout (`> ### 💡/⚠️/📌/🔗`)
   structure intact.
4. Fill in front matter (see below). Give each language its own `summary` in
   that language.
5. Build locally to verify, then commit and push (see Deploy). The KO/EN toggle
   and the home language switch update automatically.

A single-language note is fine too (just omit the pair) — the toggle only shows
when a translation with the same `ref` exists.

## Front matter

Only `title` and `date` are required; for bilingual posts also set `lang` and
`ref`.

```yaml
---
layout: paper
lang: en                 # ko | en  (drives the home language switch + toggle)
ref: pld-self-improving-vla   # shared key linking the KO and EN versions
title: "Paper Title"
date: 2025-10-30         # newest-first ordering
venue: "ICLR 2026 · arXiv:2511.00091"   # optional
tags: [VLA, Reinforcement-Learning, Paper-Review]   # click-to-filter labels
summary: "One-line description, in this file's language."   # optional
authors: "First author et al."          # optional
affiliations: "NVIDIA · CMU · ..."      # optional
paper_url: "https://arxiv.org/abs/..."  # optional → 📄 Paper link
code_url: "https://github.com/..."      # optional → 💻 Code link
---
```

- **Math:** `$...$` inline, `$$...$$` display (rendered with MathJax on the live
  site; it won't render in a local sandbox that can't reach the CDN — that's
  expected, verify structure instead).
- **Code fences** are syntax-highlighted (rouge). Unlabeled fences (ASCII
  diagrams) render as plain monospace.
- Callout boxes = a blockquote whose first line is `### <emoji> Title`; they get
  card styling. The opening quote of a note renders as a lead card.

## Local verification (optional but preferred for long notes)

```bash
bundle install   # first time
JEKYLL_ENV=production bundle exec jekyll build
# jekyll exe lives at the rbenv gems path if `bundle exec jekyll` can't find it:
#   bundle exec ruby $(gem which jekyll | sed 's#/lib/jekyll.rb#/exe/jekyll#') build
```
Check `_site/papers/<ref>/` and `_site/papers/<ref>-en/` exist and that the
language toggle + home `English (n) / 한국어 (n)` switch are wired.

## Auto-merge / push policy (the owner delegated this)

The owner has delegated merge authority for this personal repo.

- If you can push directly, **push straight to `main`** (skip the PR). Every
  push to `main` triggers the deploy workflow.
- Keep the dev branch `claude/github-pages-paper-archive-7u14xl` in sync with
  `main` when you push.
- If a session forces you onto a non-`main` branch and the GitHub tools are
  available, open a PR and **merge it yourself** (squash) as soon as it's
  mergeable — don't wait for the owner to click merge.
- After a change reaches `main`, confirm the deploy workflow succeeded; if it
  failed, read the logs and fix it.

## Deploy

GitHub Pages builds via `.github/workflows/pages.yml` (Pages Source: GitHub
Actions, enabled by the workflow's `enablement: true`). The repo is public, so
Pages is free. The `github-pages` deployment environment must allow the branch
that deploys (`main`); if a fresh environment ever restricts branches wrongly,
deleting the `github-pages` environment and re-running fixes it.

## Layout / structure reference

- One paper = one Markdown file in `_posts/`, named `YYYY-MM-DD-<ref>[-en].md`.
- `index.html` — home: language switch + tag filter (combined, English-first).
- `assets/js/filter.js` — combined language + tag filtering.
- `_layouts/paper.html` — paper page; renders authors/affiliations and the
  KO ⇄ EN toggle (shown when a same-`ref` translation exists).
- `assets/css/style.css` — callout cards, tags, language switch/toggle, code,
  dark-mode.
- `templates/paper-template.md` — copy-paste front-matter template.
