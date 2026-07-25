# MIRA

The formal MIRA implementation is self-contained under `MIRA/`. It uses the
sibling official `ATENA-A-EDA/benchmark` checkout and writes formal runs to:

```text
results/MIRA/{schema}{dataset}/seed{seed}/
```

AVP is not included in the public release and is fixed to `0`.

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

See `MIRA/README.md` for the method contract, batch command, and outputs.
