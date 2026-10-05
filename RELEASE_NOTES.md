# Clean release notes

## Retained

- Clean surrogate training, stochastic gradient-matching signing, victim
  retraining, model definitions, augmentations, datasets, and supporting code.
- Optional baselines and model/dataset branches that share those modules.

## Excluded

- All historical Git objects, branches, and tags.
- Figures, PDFs, images, notebooks (including embedded outputs), experiment
  dumps, archives, duplicate submission sources, editor settings, and private
  cluster/job launchers.
- One-off detection/analysis scripts with embedded paths or experimental data;
  the new detection CLI uses explicit local inputs.

## Portability fixes and additions

- Create output directories before writing run status or arguments.
- Permit zero data-loader workers and CPU metrics/synchronization in small tests.
- Keep `torchrun` rendezvous settings instead of requiring SLURM in every
  distributed invocation. Multi-GPU behavior remains unvalidated here.
- Accept direct or per-run clean checkpoints; sort checkpoint candidates.
- Add ImageFolder indexing and relative image-filename support.
- Fix undefined dropout import, remove two unused registrations referencing a
  missing model block, and remove the unimplemented MNIST parser branch.
- Fix validation-loss quantiles for list-valued metrics and limit top-k to the
  available classes for toy tests.
- Make trusted pickle loading explicit in the entry points.
- Add a reference stage script, binomial detection CLI, offline tests,
  dependency specifications, release hygiene, and upstream license notices.

The original gradient-matching objective and optimization update are retained.
Public example seeds are not the historical experimental keys. Fixes and new
helpers mean this is not a byte-identical archive of the original experiments.
See README for the validation boundaries; no paper-scale results are claimed.
