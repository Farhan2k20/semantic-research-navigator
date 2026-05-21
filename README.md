# Semantic Research Navigator

A small research-style prototype for turning a messy set of paper notes into a structured evidence map. The artifact demonstrates how an AI-assisted workflow can help a student or researcher compare claims, extract methods, identify limitations, and generate a concise synthesis without losing the source-level trail.

## Why This Exists

Literature reviews often become a pile of PDFs, highlights, and disconnected notes. Semantic Research Navigator explores a better workflow: papers are represented as evidence cards, each card is scored for relevance and confidence, and the system produces a synthesis that keeps claims tied to their supporting sources.

This is a prototype artifact rather than a peer-reviewed publication. It is designed to communicate a research idea clearly, with a working interactive project page that can be shared as a portfolio or application link.

## Core Idea

The prototype models a literature review as four connected layers:

1. **Question** - the research question or topic being explored.
2. **Evidence** - paper-level claims, methods, datasets, and limitations.
3. **Synthesis** - themes that appear across multiple sources.
4. **Gaps** - missing comparisons, weak evidence, or unanswered questions.

## Features

- Interactive project page for exploring the artifact.
- Evidence cards with method, claim, limitation, and confidence fields.
- A lightweight scoring model for ranking paper relevance.
- A synthesis panel that turns selected evidence into a short research summary.
- Clear framing for responsible use: the tool supports literature review, but does not replace source reading.

## Demo

Open `index.html` locally or publish this repository with GitHub Pages.

Suggested GitHub Pages URL after publishing:

`https://Farhan2k20.github.io/semantic-research-navigator/`

## Repository Structure

```text
semantic-research-navigator/
  index.html
  styles.css
  app.js
  README.md
  submission.md
```

## Suggested Submission Blurb

See [`submission.md`](submission.md) for a paste-ready link and description.

## Future Work

- Add real paper metadata import from BibTeX or DOI.
- Use embeddings to cluster related claims.
- Add citation export for generated syntheses.
- Evaluate whether the workflow improves recall and synthesis quality compared with manual note-taking.
