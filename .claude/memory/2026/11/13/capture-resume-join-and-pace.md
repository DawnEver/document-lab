---
name: capture-resume-join-and-pace
description: "pr/capture-3: --resume/capture-merge (drift-aligned join), adaptive auto pace, pause, wait-when-lost; lessons from a real resume"
metadata:
  type: project
---

Branch pr/capture-3 (worktree .claude/worktrees/capture-3, local, not pushed; based on pr/capture-2 4c6e9ef): check_name refuses Windows-unnameable outputs (a:b.pdf became a hidden NTFS stream; recovered by copying the stream). capture.join.join + capture-merge + capture --resume: the second's head (grown until unique) is found in the first's tail (searched back, doubling) by stitch.drift, which ranks sideways shifts by column ink and verifies each with locate at the strict tolerance; seam cut mid-overlap so window borders at view edges drop out; a shifted second is redrawn in line. --resume moves the area sideways (and to the earlier width within --width-tolerance 64) at Start. Measured on BS-EN-60034-1: a 13.5 grey-level miss was NOT re-rendering but a 56 px sideways offset of the marked area (0.06 once aligned) -- loosening tolerance to 20 hid it and produced a misaligned join (reverted from backup); JOIN_TOLERANCE is back to 8. Auto: Pace grows reach by 1.25 every --streak 3 placed steps up to --stride-max 0.75, settle gap halves back; lost beyond recovery now waits red until the user brings the view back (--stop-when-lost restores the stop). HUD: Pause/Resume, W x H label, --open. Open: push/PR of pr/capture-3; PR #8 blocked by dev-lab RECURSIVE-DELETE refusal over budget (user will prompt) and an undeclared AGENTS.md merge resolution in 9e9d031.
