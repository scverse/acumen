# acumen fit

```{acumen-command} fit
`fit` repeats one epoch at a time (bench, wiki, improve, validate) and stops when `--patience`
epochs pass without validation mean success rising by more than `--min-delta`, when `--max-epochs`
is reached, or when validation reaches 100%. `--epochs N` runs exactly N epochs with no early
stopping. Re-running `acumen fit` resumes from the last finished epoch, and `--epochs` and
`--max-epochs` count the total number of epochs, not how many more to run. Afterwards, the best
version is benched once on `test` and `report.html` is written.

`--ablation` runs one ablation arm in place of a normal fit. One project root holds one run of one
arm, so use a fresh root for each and compare arms on `test`.

- `no-wiki` runs the full fit loop, but the improver never gets a wiki. Instead it gets the parent
  skill, the parent's raw train runs (each task with an `INDEX.md` of pass/fail and load per run),
  and the source.
- `no-optimize` writes `v1` once from the package source alone, with no train benches, no wiki and
  no loop, then benches that `v1` and the no-skill baseline on `test` only. `valid` is skipped and
  the report shows the `test` split.
```
