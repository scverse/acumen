# Getting started

This tutorial builds a skill for [decoupler](https://github.com/scverse/decoupler). decoupler is
an scverse package that infers biological activities, such as pathway, transcription factor and
gene set scores, from omics data. We will set up an `acumen` project, write a few tasks, optimize
a skill on them and ship it into decoupler so that its users can install it.

To keep things short, we use only two tasks.

- **scell** is a code task. The agent runs a pathway analysis on a single-cell dataset and
  reports the top pathways of one cell type.
- **species** is a knowledge task. The agent answers a multiple-choice question about which
  organisms decoupler's databases support.

## Create a project

Create an empty directory and run `acumen init` inside it.

```bash
mkdir example_decoupler && cd example_decoupler
acumen init
```

`acumen init` writes two starter files, `config.yaml` and `tasks.yaml`. Both files have comments
that explain every field, so you only need to fill them in. The project now looks like this.

```text
example_decoupler/
├── config.yaml     # the target package and how to benchmark it
└── tasks.yaml      # the tasks and their answers
```

## Configure the target

`config.yaml` tells `acumen` which package to build a skill for and how to benchmark it. This is
the config for decoupler.

```yaml
# The package to build a skill for, and the ref to check out.
repo: https://github.com/scverse/decoupler
ref: main

# What to install into the target's throwaway venv.
extras: []
# [full] installs all of decoupler's optional dependencies.
# Without it, some tasks fail because a dependency is missing.
dependency_groups: [full]
pip_packages: []
python: "3.12"

# Extra environment variables the agents may keep.
env_passthrough: []

# Benchmark models. A pass is models x tasks x replicates per split.
models:
    - claude-fable-5-1
    - claude-opus-5-5
    - claude-sonnet-5-5
    - claude-haiku-4-5-20251001
    - gpt-6-astra
    - gpt-6.1-sol
    - gpt-6.1-terra
    - gpt-6-luna
# Runs of each model on each task and split.
n_replicates: 3
# Number of agents that run at the same time.
max_concurrency: 6

# Caps for each benchmark agent.
max_turns: 40
max_usd: 3.0

# Model for the agents that write tasks, review, improve and ship.
meta_model: claude-fable-5-1
```

These are the fields that matter most.

- **`repo` and `ref`** point to the `main` branch of decoupler on GitHub. `acumen` clones this
  branch and installs it into a throwaway virtual environment, the *target venv*. Every script and
  every agent runs inside this environment.
- **`dependency_groups`** adds decoupler's `full` dependency group to the target venv. Some tasks
  need these optional dependencies and fail without them.
- **`models` and `n_replicates`** set the size of a benchmark pass. Each model runs each task
  `n_replicates` times on every split that is benched. Here this gives 8 models × 2 tasks × 3
  replicates, which is 48 runs per split.
- **`max_turns` and `max_usd`** limit each benchmark agent. An agent that goes off track fails
  instead of spending a lot of money.
- **`meta_model`** is the model of the meta-agents. These agents do the work around the
  benchmark. They write and review tasks, write the wiki, improve the skill and ship it.
- **`skill_name`** is the name of the skill. If it is not set, it defaults to the repository
  name, in this case `decoupler`.

`acumen` builds the target venv the first time a command needs it and keeps it in
`~/.cache/acumen`. The `--cache` option of each command changes this location. `acumen` does not
pull new commits by itself. When decoupler’s `main` branch changes and you want the new code, pass
`--refresh-target` to the next command, for example `acumen prepare --refresh-target`. `acumen`
then clones the repository again and rebuilds the venv. If `repo` is a local path, `acumen`
rebuilds the venv by itself when the checked-out commit changes, so commit your changes first.
Changing `ref` or any of the install fields also creates a new venv.

## Write the tasks

A task is a prompt together with the answer the agent should give. Every task has three splits,
and they play the same roles as in machine learning.

  | split   | used for                                                      |
  | ------- | ------------------------------------------------------------- |
  | `train` | The improver learns the skill from the results on this split. |
  | `valid` | Each new version is scored here. This picks the best version. |
  | `test`  | The best version is benched here once, at the very end.       |

The three splits test the same functionality, but each one changes something. A split can use
other input data, other parameters, another postprocessing step or a different question. This
way the agent cannot memorize a single dataset or configuration. A skill only scores well on
`valid` and `test` if it teaches the agent how to use the package.

Keep these rules in mind when you write tasks.

- `acumen` compares answers by **exact string match** after `strip()`. Ask for a short answer
  and describe its format, such as the separator, the order and the case.
- **Do not name the package** in the prompt. `acumen` already tells the agent that decoupler is
  installed and that it should use it. The prompt only describes the analysis.
- **Cover what matters to you.** A task is any functionality of the package that agents should
  perform correctly. `acumen` only optimizes the skill for the functionalities that have a task,
  so agents may keep failing at the ones that are left out.

```{important}
Writing the tasks is the most important step in `acumen`. If the tasks are not meaningful, the
benchmark is meaningless too, and training wastes time and money.

We strongly recommend writing the tasks by hand. Someone who knows the package knows what it can
realistically do, where it breaks and which corner cases matter. `acumen tasks` can generate a
first draft of tasks, but we do not recommend it, because the generated tasks are less relevant
than tasks written with domain knowledge. The tasks in this tutorial were written by hand.
```

### A code task

```yaml
- id: scell
  data: [pbmc3k.h5ad]
  train:
      data: [progeny.csv]
      prompt: >
          Using the pbmc3k data, find out the 3 most active PROGENy pathways
          in CD4 T cells compared to the other cell types.

          Report only the pathway names sorted by score (descending),
          separated by semicolons without spaces.
      answer: MAPK;Estrogen;TGFb
  valid:
      data: [progeny.csv]
      prompt: >
          Find the 3 most deactivated PROGENy pathways in Megakaryocytes
          compared to the other cell types in the pbmc3k dataset.

          Report only the pathway names sorted by score (ascending),
          separated by semicolons without spaces.
      answer: Trail;JAK-STAT;Estrogen
  test:
      data: [hallmark.csv]
      prompt: >
          I am working with the pbmc3k dataset. I want to find the 5 most
          active hallmarks pathways in B cells compared to other cell types.

          Report only the pathway names sorted by score (descending),
          separated by semicolons without spaces.
      answer: "KRAS_SIGNALING_DN;ALLOGRAFT_REJECTION;\
          INTERFERON_GAMMA_RESPONSE;E2F_TARGETS;SPERMATOGENESIS"
```

The three splits ask for a different cell type, a different direction (most active or most
deactivated) and a different number of pathways. The `test` split also uses a different gene set
resource. All three need the same decoupler workflow.

The `data` field lists files from the prepared data cache, which the
[next section](#prepare-the-data) builds. `acumen` links these files into the working directory
of each run, so agents do not download anything while they are benchmarked. The downloads happen
once, when the data is prepared. This makes the benchmark faster and avoids network errors during
the runs, so agents can focus on using the package.

A `data` field at the task level applies to every split. A `data` field inside a split **adds**
files to that list. In this example, every split gets `pbmc3k.h5ad`. The `train` and `valid`
splits also get the PROGENy network `progeny.csv`, and the `test` split gets the hallmark gene
sets `hallmark.csv`.

### A knowledge task

```yaml
- id: species
  needs_script: false
  train:
      prompt: >
          I have a guinea pig dataset, can I use the databases provided in
          decoupler?

          a) Yes, all of them using decoupler's orthology conversion.

          b) Yes, but only the ones containing guinea pig gene symbols.

          c) No, decoupler can only be used with human data.

          d) Yes, but decoupler does not directly provide conversion from
          guinea pig to human.

          Pick the right answer and reply only the lowercase letter.
      answer: d
  valid:
      prompt: >
          I have a pig dataset, can I use the databases provided in
          decoupler?

          a) Yes, all of them using decoupler's orthology conversion.

          b) Yes, but only the ones containing pig gene symbols.

          c) No, decoupler can only be used with human data.

          d) Yes, but decoupler does not directly provide conversion from
          pig to human.

          Pick the right answer and reply only the lowercase letter.
      answer: a
  test:
      prompt: >
          I have a zebrafish dataset, could I use the databases in
          decoupler?

          a) Yes, but only the ones containing zebrafish symbols.

          b) Yes, all of them using orthology conversion as implemented in
          decoupler.

          c) Yes, but decoupler does not provide conversion from zebrafish
          to human.

          d) No, only human data is possible.

          Pick the right answer and reply only the lowercase letter.
      answer: b
```

Not every task needs code. This task checks whether the agent knows which organisms decoupler’s
orthology conversion supports. The multiple-choice format keeps the answer to a single letter. The
options change order between splits, so the skill cannot simply learn a letter. The field
`needs_script: false` tells `acumen check` that this task has no reproducer.

Both tasks go under a top-level `tasks` key.

<!-- panache-ignore-format-start -->
:::{dropdown} The complete `tasks.yaml`

```yaml
tasks:
    - id: scell
      data: [pbmc3k.h5ad]
      train:
          data: [progeny.csv]
          prompt: >
              Using the pbmc3k data, find out the 3 most active PROGENy pathways
              in CD4 T cells compared to the other cell types.

              Report only the pathway names sorted by score (descending),
              separated by semicolons without spaces.
          answer: MAPK;Estrogen;TGFb
      valid:
          data: [progeny.csv]
          prompt: >
              Find the 3 most deactivated PROGENy pathways in Megakaryocytes
              compared to the other cell types in the pbmc3k dataset.

              Report only the pathway names sorted by score (ascending),
              separated by semicolons without spaces.
          answer: Trail;JAK-STAT;Estrogen
      test:
          data: [hallmark.csv]
          prompt: >
              I am working with the pbmc3k dataset. I want to find the 5 most
              active hallmarks pathways in B cells compared to other cell types.

              Report only the pathway names sorted by score (descending),
              separated by semicolons without spaces.
          answer: "KRAS_SIGNALING_DN;ALLOGRAFT_REJECTION;\
              INTERFERON_GAMMA_RESPONSE;E2F_TARGETS;SPERMATOGENESIS"

    - id: species
      needs_script: false
      train:
          prompt: >
              I have a guinea pig dataset, can I use the databases provided in
              decoupler?

              a) Yes, all of them using decoupler's orthology conversion.

              b) Yes, but only the ones containing guinea pig gene symbols.

              c) No, decoupler can only be used with human data.

              d) Yes, but decoupler does not directly provide conversion from
              guinea pig to human.

              Pick the right answer and reply only the lowercase letter.
          answer: d
      valid:
          prompt: >
              I have a pig dataset, can I use the databases provided in
              decoupler?

              a) Yes, all of them using decoupler's orthology conversion.

              b) Yes, but only the ones containing pig gene symbols.

              c) No, decoupler can only be used with human data.

              d) Yes, but decoupler does not directly provide conversion from
              pig to human.

              Pick the right answer and reply only the lowercase letter.
          answer: a
      test:
          prompt: >
              I have a zebrafish dataset, could I use the databases in
              decoupler?

              a) Yes, but only the ones containing zebrafish symbols.

              b) Yes, all of them using orthology conversion as implemented in
              decoupler.

              c) Yes, but decoupler does not provide conversion from zebrafish
              to human.

              d) No, only human data is possible.

              Pick the right answer and reply only the lowercase letter.
          answer: b
```

:::
<!-- panache-ignore-format-end -->

## Prepare the data

Preparing data is optional. It is useful when tasks start from datasets or resources that would
otherwise be downloaded in every run. If handling downloads is part of what a task should test,
do not prepare data for that task.

The `scell` task needs a dataset and two prior knowledge resources, so we build them once. Each
file in `prepare/` is a short Python script that writes one or more files into the data cache.

```python
# prepare/pbmc3k.py
import decoupler as dc

dc.ds.pbmc3k().write_h5ad("pbmc3k.h5ad")
```

```python
# prepare/progeny.py
import decoupler as dc

net = dc.op.progeny(organism="human")
net.to_csv("progeny.csv", index=False)
```

```python
# prepare/hallmark.py
import decoupler as dc

net = dc.op.hallmark(organism="human")
net.to_csv("hallmark.csv", index=False)
```

Then run `acumen prepare`.

```bash
acumen prepare
```

`acumen` runs each script in the target venv, inside the data cache directory (`cache/` by
default). This means the scripts can use decoupler to fetch the data. `acumen` skips a script that
already succeeded, unless the script changed or you pass `--force`. Use `--only NAME` to run a
single script. `acumen` keeps a record of each build in `.acumen-prepared/` and makes the cache
read-only, so no run can change the data of another run.

After this step, the project looks like this.

```text
example_decoupler/
├── config.yaml
├── tasks.yaml
├── prepare/                 # new: the scripts you wrote
│   ├── hallmark.py
│   ├── pbmc3k.py
│   └── progeny.py
└── cache/                   # new: the prepared data
    ├── .acumen-prepared/    # one build record per script
    ├── hallmark.csv
    ├── pbmc3k.h5ad
    └── progeny.csv
```

When a benchmark run starts, `acumen` links the files listed under `data` into the working
directory of the run. The agent then finds `pbmc3k.h5ad` and `progeny.csv` next to it.
`acumen prepare` also warns you if `tasks.yaml` lists a file that no script produced.

## Check the tasks

Checking the tasks is optional, but we strongly recommend it before you start training. A task
with a wrong answer is worse than no task, because agents that solve it correctly are graded as
wrong.

To check a code task, give each of its splits a **reproducer** at `tasks/<id>-<split>.py`. The
reproducer redoes the analysis and writes its answer to `answer.md`, just like an agent does in a
benchmark run. This is the reproducer for the `train` split of `scell`.

```python
# tasks/scell-train.py
import anndata as ad
import decoupler as dc
import pandas as pd

# Read the dataset and the PROGENy network
adata = ad.read_h5ad("pbmc3k.h5ad")
net = pd.read_csv("progeny.csv")

# Infer pathway activities
dc.mt.ulm(data=adata, net=net)

# Rank pathways per cell type against the rest
score = dc.pp.get_obsm(adata, key="score_ulm")
df = dc.tl.rankby_group(
    adata=score,
    groupby="celltype",
    reference="rest",
    method="t-test_overestim_var",
)

# Write the top 3 pathways for CD4 T cells
top = df[df.group == "CD4 T cells"].sort_values("stat", ascending=False)
with open("answer.md", "w") as f:
    f.write(";".join(top.head(3)["name"]))
```

The `valid` and `test` reproducers only change the cell type, the sort direction, the number of
pathways and the network.

:::{dropdown} `tasks/scell-valid.py`
```python
import anndata as ad
import decoupler as dc
import pandas as pd

# Read the dataset and the PROGENy network
adata = ad.read_h5ad("pbmc3k.h5ad")
net = pd.read_csv("progeny.csv")

# Infer pathway activities
dc.mt.ulm(data=adata, net=net)

# Rank pathways per cell type against the rest
score = dc.pp.get_obsm(adata, key="score_ulm")
df = dc.tl.rankby_group(
    adata=score,
    groupby="celltype",
    reference="rest",
    method="t-test_overestim_var",
)

# Write the top 3 pathways for Megakaryocytes
top = df[df.group == "Megakaryocytes"].sort_values("stat", ascending=True)
with open("answer.md", "w") as f:
    f.write(";".join(top.head(3)["name"]))
```
:::

:::{dropdown} `tasks/scell-test.py`
```python
import anndata as ad
import decoupler as dc
import pandas as pd

# Read the dataset and the hallmark gene sets
adata = ad.read_h5ad("pbmc3k.h5ad")
net = pd.read_csv("hallmark.csv")

# Infer pathway activities
dc.mt.ulm(data=adata, net=net)

# Rank pathways per cell type against the rest
score = dc.pp.get_obsm(adata, key="score_ulm")
df = dc.tl.rankby_group(
    adata=score,
    groupby="celltype",
    reference="rest",
    method="t-test_overestim_var",
)

# Write the top 5 pathways for B cells
top = df[df.group == "B cells"].sort_values("stat", ascending=False)
with open("answer.md", "w") as f:
    f.write(";".join(top.head(5)["name"]))
```
:::

The `species` task has `needs_script: false`, so it has no reproducers. Now run `acumen check`.

```bash
acumen check
```

`acumen check` works in two phases.

1. **Reproduce.** `acumen` runs each reproducer in the target venv, with the same `data` files a
   benchmark run gets. It then compares the answer in `answer.md` with the answer in
   `tasks.yaml`. This phase is deterministic and free. The `species` task is reported as
   `skipped`.
2. **Review.** An agent reads the prompts, the answers and the reproducers of each task and
   looks for inconsistencies between them. These are usually human errors made while writing the
   tasks by hand. For example, a prompt may ask for pathways in ascending order while the recorded
   answer lists them in descending order. No agent could ever solve such a task. The review flags
   every split with an inconsistency and suggests a fix.

Run `acumen check --no-review` to skip the second phase. It starts no agent and costs nothing, so
you can run it after every change to `tasks.yaml`. Use `--task ID` and `--split SPLIT` to check
only part of the tasks.

With the reproducers, the project looks like this.

```text
example_decoupler/
├── config.yaml
├── tasks.yaml
├── prepare/
├── cache/
└── tasks/              # new: one reproducer per task and split
    ├── scell-test.py
    ├── scell-train.py
    └── scell-valid.py
```

## Optimize the skill

Now train the skill with `acumen fit`.

```bash
acumen fit
```

```{warning}
This step can take a long time and spend a lot of tokens. Every benchmark pass runs each model
on each task, `n_replicates` times, so its cost grows with the number of models, replicates and
tasks. In this project one pass is 48 runs per split, and every epoch runs at least two passes.
Try a smaller setup first, for example with fewer models or with `--replicates 1`.
```

`acumen fit` trains the skill in the same way you would train a model, one **epoch** at a time.
Each epoch has four steps.

1. **Bench.** `acumen` benchmarks the current skill on `train`. The first epoch has no skill yet,
   so it benchmarks agents without a skill instead. It does this on both `train` and `valid`, so
   that every later version has a baseline to compare against.
2. **Wiki.** By default, `acumen` writes down what happened on each task in `wiki/`. It records
   what the agents tried, what worked and what made the passing runs different from the failing
   ones. This step is optional, as explained [below](#with-or-without-the-wiki).
3. **Improve.** A meta-agent writes the first skill from the results of the `train` runs, or
   rewrites the current one. The versions are saved as `skills/v1`, `skills/v2` and so on. This
   agent only ever sees the `train` tasks. It never sees the `valid` or `test` tasks, so their
   scores stay a fair measure of the skill.
4. **Validate.** `acumen` benchmarks the new version on `valid`.

After each epoch, `acumen fit` prints a progress line and adds a row to the training curve in
`training.csv`. Training stops when one of these things happens.

- The validation success rate does not improve for `--patience` epochs (2 by default). An epoch
  only counts as an improvement if it beats the last best score by more than `--min-delta`
  (0.05 by default, which is 5 percentage points).
- Training reaches `--max-epochs` (10 by default).
- The validation success rate reaches 100%.

Use `--epochs N` to run exactly N epochs without early stopping. When training stops,
`acumen fit` takes the version with the best validation score and benchmarks it once on `test`,
together with the baseline. It then writes the report, `report.html`. You can regenerate the
report at any time with `acumen report`.

You can stop and resume `acumen fit` at any time. When you run it again, it continues from the
last finished epoch. Note that `--epochs` and `--max-epochs` count all epochs, not only the new
ones.

### Example output

The output below shows what a fit on this project looks like.

```console
$ acumen fit
fit: target 10 epoch(s) total (early stopping with patience 2, min-delta 5pp)

Epoch 1/10
bench baseline 100%|██████████████| 96/96  [7m48s]  success 48% (mean)  $29.86 total
          wiki 100%|██████████████| 2/2  [3m05s]  $4.12 total
  inferring skill → v1 done  (6m40s, $5.62, 58 turns)
   bench valid 100%|██████████████| 48/48  [4m02s]  success 71% (mean)  $14.90 total
Epoch 1/10 |█░░░░░░░░░░░░░| v1  train 46%  valid 71%  $0.31/run  (best v1, patience 0/2)

Epoch 2/10
...
Epoch 2/10 |███░░░░░░░░░░░| v2  train 67%  valid 83%  $0.27/run  (best v2, patience 0/2)

Epoch 3/10
...
Epoch 3/10 |████░░░░░░░░░░| v3  train 77%  valid 79%  $0.28/run  (best v2, patience 1/2)

Epoch 4/10
...
Epoch 4/10 |██████░░░░░░░░| v4  train 79%  valid 81%  $0.27/run  (best v2, patience 2/2)

early stop: validation did not improve by more than 5pp over v2 (83%) in 2 epoch(s); best is v2 (83%).
 held-out test 100%|██████████████| 96/96  [7m31s]  success 64% (mean)  $27.40 total

fit complete: best version v2 (valid 83%).
training curve → /home/me/example_decoupler/training.csv
report → /home/me/example_decoupler/report.html
```

In this example, `v2` scores best on `valid`. The next two versions do not beat it by more than 5
percentage points, so training stops after epoch 4. `acumen fit` then benchmarks `v2` and
the baseline on `test`.

After training, the project looks like this.

```text
example_decoupler/
├── config.yaml
├── tasks.yaml
├── prepare/
├── cache/
├── tasks/
├── runs/                # new: every graded benchmark run
│   ├── noskill/         # the baseline, on train, valid and test
│   │   └── train/
│   │       └── claude-opus-5-5/
│   │           └── scell/
│   │               └── rep_1/   # answer.md, transcript, result
│   ├── skill_v1/        # each version on train and valid,
│   ├── ...              # and the best one also on test
│   └── skill_v4/
├── wiki/                # new: what the improver learns from (optional)
├── skills/              # new: one directory per version
│   ├── v1/              # SKILL.md and references/
│   ├── ...
│   └── v4/
├── training.csv         # new: the training curve, one row per epoch
└── report.html          # new: the report
```

Each step of an epoch is also a command of its own (`acumen bench`, `acumen wiki` and
`acumen improve`), and `acumen epoch` runs one full epoch. Use `--feedback` to give the improver your own notes. The
[API](api/index.md) lists every option.

### With or without the wiki

`acumen fit` writes the wiki by default, but the wiki is optional. To train without it, run the
following command.

```bash
acumen fit --ablation no-wiki
```

This runs the same loop without the wiki step. The improver learns directly from the current skill
and the raw results of the `train` runs. In our experiments both options perform about the same,
so you can use either one. Run it in a new project directory, because each directory holds a
single fit.

## Ship the skill

The version to ship is the one that `acumen fit` selected, which is the version with the best
validation score. The **Held-out test** section of `report.html` names this version and shows the
command to ship it. In the example above, this is `v2`.

```{warning}
Do not simply ship the version of the last epoch. Later versions can score lower than earlier
ones. In the example above, `v4` is the newest version, but `v2` is the best one.
```

Ship the selected version with `acumen ship`.

```bash
acumen ship --skill v2
```

`acumen ship` starts an agent that adds a `decoupler-install-skills` command to decoupler. This
command installs the skill, which is `SKILL.md` and its `references/` folder. The agent works out
how the package is built. It then builds the package in a fresh environment, installs it and
checks that the skill is included. Our `repo` is a GitHub URL, so the agent opens a pull request
with the change. If `repo` is a local path, the agent writes the change directly into it.

Once the change is released, decoupler users can install the skill for the agent they use.

```bash
decoupler-install-skills --agent claude
```

The `--agent` option also accepts `codex`, `agents` and `claude-science`. Use `--dest` to install
the skill into any other directory.
