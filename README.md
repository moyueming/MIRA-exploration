# MIRA Exploration

This repository is the minimal reproduction package for MIRA across SDSS
Galaxy target discovery, UCI Covertype target discovery, and the official
ATENA A-EDA benchmark. Generated results, checkpoints, plots, tests, and paper
sources are intentionally excluded.

## Methods

| Task | MIRA | Comparison methods |
| --- | --- | --- |
| Galaxy | MIRA; no-extrinsic ablation via `--w_ext 0` | Random, Pure A3C, DORA, Greedy EDA, ATENA-style, ATENA-style + extrinsic reward |
| Covertype | `mira`; `mira_no_ext` | Random, Pure A3C, DORA, Greedy EDA, ATENA-style, ATENA-style + extrinsic reward |
| Official ATENA | MIRA with AVP fixed off | Official ATENA, Random, Greedy, DORA, Pure A3C |

## Layout

```text
app/pipelines/                  Galaxy data and set operators
app/data/sdss/                 Galaxy metadata and split dataset archive
rl/A3C_2_actors/               Galaxy MIRA implementation
rl/targets/                     Fixed Galaxy targets for seeds 1-3
baselines/                      Galaxy baselines and shared Greedy policy
covertype-exploration/          Covertype environment, methods, and downloader
ATENA_dataset/MIRA/             MIRA for the official ATENA benchmark
ATENA_dataset/atena_baselines/  ATENA comparison methods
ATENA_dataset/ATENA-A-EDA/      Pinned official benchmark submodule
```

Clone with the official ATENA dependency:

```bash
git clone --recurse-submodules https://github.com/moyueming/MIRA-exploration.git
cd MIRA-exploration
```

For an existing clone, run `git submodule update --init --recursive`.

## Galaxy

Install with Python 3.8 or 3.9:

```bash
python -m pip install -r requirements.txt
```

The SDSS CSV is distributed as six archive parts. Expected SHA-256 values:

```text
galaxies.tar.gz.aa  b8733cb51e54108687888be55eb700eb1402790958492cf351d5a2747813a689
galaxies.tar.gz.ab  b55600e3aa09d6ef1154a096d97111a96010663063cd349da4d29606ad78a9f1
galaxies.tar.gz.ac  fa381d4240a99b6f92c6aec7da5b93129365f6af9e29e59b9fc90ea8fb4c6a8b
galaxies.tar.gz.ad  fb7baa46fe8943685cb3c859ed78c4aa580eac7e25941a53067ab90e58a3df41
galaxies.tar.gz.ae  c92bad1646e3ed8610af781cee83271b2282551615c4a2cc89623fa8d857083e
galaxies.tar.gz.af  28b409606489aa1c85f4fc6250701323b6697ad91930be16c7b98c9017d8f012
```

Linux/macOS reconstruction:

```bash
cd app/data/sdss
cat galaxies.tar.gz.a{a,b,c,d,e,f} > galaxies.tar.gz
tar -xzf galaxies.tar.gz
cd ../../..
```

Windows PowerShell reconstruction:

```powershell
cd app\data\sdss
cmd /c copy /b galaxies.tar.gz.aa+galaxies.tar.gz.ab+galaxies.tar.gz.ac+galaxies.tar.gz.ad+galaxies.tar.gz.ae+galaxies.tar.gz.af galaxies.tar.gz
tar -xzf galaxies.tar.gz
cd ..\..\..
```

Expected reconstructed hashes:

```text
galaxies.tar.gz  ee8fdbf5b5a96a8536889b61d4b6d8df9ad85619e8141e1529c3c3fa4a7613e8
galaxies.csv     4a8af7c032745039e43fd7dd3abf2720da84a021e3f81c6e67f2e0d970f2e072
```

Example runs:

```bash
python RL-launcher.py --mode scattered --target_set fixed_seed_1 --workers 12 --name galaxy_mira_seed1
python RL-launcher.py --mode scattered --target_set fixed_seed_1 --workers 12 --w_ext 0 --name galaxy_mira_no_ext_seed1
python RL-launcher-pure-a3c.py --mode scattered --target_set fixed_seed_1 --workers 12 --name galaxy_pure_a3c_seed1
python RL-launcher-paper-a3c.py --mode scattered --target_set fixed_seed_1 --workers 12 --name galaxy_dora_seed1
python RL-launcher-atena-style.py --mode scattered --target_set fixed_seed_1 --workers 12 --name galaxy_atena_style_seed1
python RL-launcher-atena.py --mode scattered --target_set fixed_seed_1 --workers 12 --name galaxy_atena_ext_seed1
python RL-launcher-greedy-eda.py --mode scattered --target_set fixed_seed_1 --seed 1 --workers 12 --episodes 1000 --steps 250 --output_prefix outputs/GreedyEDA/seed1
python RL-launcher-random-baseline.py --mode scattered --target_set fixed_seed_1 --seed 1 --episodes 1000 --steps 250
```

## Covertype

The raw CSV is downloaded locally from the official UCI archive:

```bash
cd covertype-exploration
python -m pip install -r requirements.txt
python scripts/download_covertype.py
```

Source: `https://archive.ics.uci.edu/static/public/31/covertype.zip`

Expected `covertype.csv` SHA-256:
`a07902ee1c9d3231c6655f23e6f75a6797d0ba26a2359f533c2c0e65d05c9bd4`.

Example runs:

```bash
python run_full_a3c.py --baseline mira --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix mira_seed1
python run_full_a3c.py --baseline pure_a3c --target_set fixed_seed_1 --seed 1 --workers 12 --output_prefix pure_a3c_seed1
```

See `covertype-exploration/README.md` for all method and preprocessing options.

## Official ATENA

The official repository is pinned as a submodule at commit
`8428c48011dbf2f7f04f3ffded917038e4670657`. It supplies the four Cyber and
four Flights datasets, simulator, references, and evaluation metrics. Netflix
is not part of the paper protocol.

```bash
cd ATENA_dataset
bash scripts/setup_venv.sh
source .venv/bin/activate
python MIRA/run.py --schema cyber --dataset_number 1 --workers 28 --seed 0 --steps 1000000 --avp 0
python run_atena_baselines.py --method pure_a3c --schema cyber --dataset_number 1 --workers 28 --seed 0 --steps 1000000
```

The released MIRA parser rejects AVP values other than `0`; the AVP module and
AVP-enabled variant are not included.

## Generated Files

Training writes to `outputs/`, `saved_models/`, Covertype `preprocessed/`, and
`ATENA_dataset/results/`. These paths are ignored and must not be committed.
