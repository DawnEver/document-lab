---
name: stale-lanes-and-hold-tags-discarded-by-the-one-day-ruling
description: "User ruling 2026-11-23: origin keeps only pr/ branches updated within one day plus the toolchain task's, and only v* tags; these stale lanes and hold/archive tags were discarded by -s ours merges, then deleted"
metadata:
  type: project
---

User ruling 2026-11-23 (integrator): keep only pr/ branches updated within one day plus the toolchain task's
assigned work; unlanded content is merged or discarded, never archived (no bundle, no archive tag, no copy under
output/); origin tags are versions only. Each ref below was idle more than 24 h and unlanded, so one `-s ours`
merge records its discard (integration keeps its own side), and `branchprune delete` removes it once this lands:

- refs/tags/archive/transformer-glyphs-and-raster
- refs/tags/archive/the-ops-app-may-break-glass
- refs/tags/archive/capture
