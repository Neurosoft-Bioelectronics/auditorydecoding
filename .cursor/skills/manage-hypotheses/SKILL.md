---
name: manage-hypotheses
description: >-
  Manage experiment hypotheses in the Obsidian vault. Create, update, and link
  hypothesis markdown files following the vault structure. Use when the user
  mentions hypotheses, experiments, research questions, vault, Obsidian, or
  wants to track experimental results.
---

# Manage Hypotheses

This skill manages the Obsidian-compatible experiment tracking vault at `vault/` in the project root.

## Vault Structure

```
vault/
├── .obsidian/                    # Obsidian config
├── _templates/
│   └── hypothesis.md             # Template for new hypotheses
├── experiments/
│   └── <experiment-name>/        # One folder per experiment
│       ├── _overview.md          # Experiment overview (links all RQs)
│       ├── <hypothesis>.md       # One file per hypothesis
│       └── results/              # Generated artifacts (plots, CSVs, HTML)
└── topics/                       # Cross-cutting concept notes
```

## Hypothesis Lifecycle

```
draft --> active --> completed (with outcome)
                 \-> abandoned (with reason)
```

| Status | Meaning |
|--------|---------|
| `draft` | Question formulated, method not yet designed |
| `active` | Experiment running or analysis in progress |
| `completed` | Results obtained, verdict rendered |
| `abandoned` | Dropped (document reason in Interpretation) |

| Outcome | When to use |
|---------|-------------|
| `pending` | No results yet (draft/active) |
| `supported` | Hypothesis confirmed |
| `partially-supported` | Mixed or weak evidence |
| `not-supported` | Hypothesis refuted |
| `strongly-supported` | Overwhelming evidence in favor |

## Creating a New Hypothesis

1. Read the template at `vault/_templates/hypothesis.md`
2. Create the file at `vault/experiments/<experiment>/<id>.md`
3. Fill in the frontmatter — all fields are required:

```yaml
---
id: rq1                              # Short identifier
title: "Descriptive Title"
experiment: "[[<experiment>/_overview]]"
status: draft                         # draft|active|completed|abandoned
outcome: pending                      # pending|supported|partially-supported|not-supported|strongly-supported
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: []                              # Thematic tags for grouping
depends_on: []                        # Wikilinks to prerequisite hypotheses
leads_to: []                          # Wikilinks to follow-up hypotheses
---
```

4. Fill body sections: Question, Hypothesis, Background, Method, Results, Interpretation, Verdict, Artifacts, Links
5. Update the parent `_overview.md` to include the new hypothesis in its table

## Creating a New Experiment

1. Create `vault/experiments/<experiment-name>/`
2. Create `vault/experiments/<experiment-name>/results/` for artifacts
3. Write `_overview.md` with:
   - Frontmatter: `type: experiment`, `status`, `created`, `updated`, `tags` (include `overview`)
   - Body: description, dataset, methodology, code references, RQ table, key conclusions, artifact links
4. Create individual hypothesis files

## Updating a Hypothesis

When updating status or adding results:

1. Update `status` and `outcome` in frontmatter
2. Update `updated` date
3. Fill in Results and Interpretation sections
4. Write the Verdict section
5. Update `leads_to` if new follow-up hypotheses emerged
6. If the hypothesis was abandoned, explain why in Interpretation

## Linking Hypotheses

Use Obsidian `[[wikilinks]]` for graph connectivity:

- **Frontmatter links** (`depends_on`, `leads_to`): structural relationships
- **Inline wikilinks**: ad-hoc references in body text, e.g. "as shown in [[rq6-power-ratio-snr]]"
- **Topic notes** in `vault/topics/`: create when a concept spans multiple experiments

When adding a link A → B via `leads_to`, also add B → A via `depends_on`.

## Adding Results Artifacts

1. Place generated files (CSV, PNG, HTML) in `vault/experiments/<experiment>/results/`
2. Reference from the hypothesis markdown with relative links: `[Label](results/filename.ext)`
3. Do NOT commit large binary files — keep them local or symlink from `outputs/`

## File Naming Conventions

- Hypothesis files: `<short-id>.md` (e.g., `rq1-channel-session-snr.md`)
- Use lowercase, hyphens for spaces, no underscores
- Keep names short but descriptive
- Prefix with `rq<N>-` when part of a numbered sequence

## Example: Quick Hypothesis Creation

To create a new hypothesis "RQ8: Does X improve Y?" in the snr-analysis experiment:

1. Read `vault/_templates/hypothesis.md`
2. Write `vault/experiments/snr-analysis/rq8-does-x-improve-y.md` with filled frontmatter and body
3. Add `"[[rq8-does-x-improve-y]]"` to the `leads_to` of whichever hypothesis motivated it
4. Add a row to the RQ table in `vault/experiments/snr-analysis/_overview.md`
