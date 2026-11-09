# API

`acumen` is used through its command line. Each page below shows the same text as `acumen <command> --help`.

## Workflow

The commands of a typical project, in the order you run them. See [Getting started](../getting_started.md) for a walkthrough.

### Set up

Describe the target package, write the tasks, build their data and verify their answers.

```{acumen-commands}
init
tasks
prepare
check
```

### Optimize

Train the skill on the tasks.

```{acumen-commands}
fit
```

### Ship

Make the best version installable for the package's users.

```{acumen-commands}
ship
```

## Individual stages

`acumen fit` is built from the commands below, and each one also runs on its own.

### Epoch

`acumen fit` runs epochs back to back until validation stops improving.

```{acumen-commands}
epoch
```

### Inside an epoch

`acumen epoch` benches the current skill on `train`, records what happened in the wiki, improves the skill, and benches the new version on `valid`.

```{acumen-commands}
bench
wiki
improve
```

### Report

`acumen fit` writes the report once training ends.

```{acumen-commands}
report
```

## Utilities

```{acumen-commands}
prices
```

```{toctree}
:hidden: true

init
tasks
prepare
check
fit
ship
epoch
bench
wiki
improve
report
prices
```
