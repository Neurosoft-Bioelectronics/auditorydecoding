# Experiment Vault

An Obsidian-compatible vault for tracking research hypotheses, experiments, and their relationships.

## Setup

Open the `vault/` directory as an Obsidian vault:

1. Open Obsidian
2. "Open folder as vault" → select the `vault/` directory
3. The `.obsidian/` config is pre-configured with Templates and Graph view

## Structure

```
vault/
├── _templates/hypothesis.md      # Template for new hypotheses
├── experiments/                   # One subfolder per experiment
│   └── <name>/
│       ├── _overview.md           # Experiment summary & RQ index
│       ├── <hypothesis>.md        # Individual hypothesis files
│       └── results/               # Plots, CSVs, HTML reports
└── topics/                        # Cross-cutting concept notes
```

## Hypothesis Lifecycle

Each hypothesis file tracks its status through a lifecycle:

| Status | Description |
|--------|-------------|
| `draft` | Question formulated, method not yet designed |
| `active` | Experiment running or analysis in progress |
| `completed` | Results obtained, verdict rendered |
| `abandoned` | Dropped with documented reason |

Outcomes for completed hypotheses: `supported`, `partially-supported`, `not-supported`, `strongly-supported`.

## Creating a New Hypothesis

Use the Obsidian Templates plugin (Ctrl/Cmd+T) and select `hypothesis` to insert the template into a new file. Fill in the frontmatter and body sections.

Alternatively, use the Cursor agent with the `manage-hypotheses` skill.

## Graph View

Open the Obsidian Graph View (Ctrl/Cmd+G) to visualize hypothesis relationships:

- Nodes are colored by location (experiments = blue, topics = green)
- Edges are created by `[[wikilinks]]` in frontmatter (`depends_on`, `leads_to`) and body text
- Use tags to filter the graph by theme

## Conventions

- File names: lowercase, hyphen-separated (e.g., `rq1-channel-session-snr.md`)
- Always update both sides of a link: if A `leads_to` B, then B `depends_on` A
- Place generated artifacts in `results/` and reference with relative links
- Use `topics/` for concepts shared across experiments
