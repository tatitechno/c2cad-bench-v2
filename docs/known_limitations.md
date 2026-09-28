# Known Limitations

- C2CAD-Bench evaluates primitive coordinate assembly, not full professional CAD workflows.
- The primitive vocabulary excludes freeform surfaces, sketch constraints, fillets, chamfers, swept profiles, and feature trees.
- The current v1.0 prompt set includes explicit counts, vectors, axes, and formula-like hints in some families. These are released because they are the exact evaluated prompts. Use `data/prompt_audit.csv` to inspect these signals.
- The scorer is deterministic but not a replacement for CAD-kernel validation, manufacturability checks, finite-element analysis, or domain expert review.
- Model rankings can become stale as hosted APIs change.
- The visualizer is qualitative; the score tables are the source of record.
