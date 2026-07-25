# MIRA

The formal MIRA implementation is self-contained under `MIRA/`. It uses the
sibling official `ATENA-A-EDA/benchmark` checkout and writes formal runs to:

```text
results/MIRA/{schema}{dataset}/seed{seed}/
```

Run from `ATENA_dataset`:

```bash
python MIRA/run.py \
  --schema cyber \
  --dataset_number 1 \
  --workers 28 \
  --seed 0 \
  --steps 1000000 \
  --avp 0
```

AVP is disabled by default. Only the exact value `--avp 1` enables it. See
`MIRA/README.md` for the complete method contract, batch command, outputs, and
verification steps.
