# SmokeAgent

**An LLM agent that turns a written smoke-control brief into a validated FDS study.**
Research prototype, built for French performance-based fire-safety studies (*ingénierie du désenfumage*), evaluated against NIST's own validation experiments.

> Status (2026-10-06): on the 10 "basic" FireBench cases, the agent reproduces FDS within the FDS Validation Guide's own uncertainty on **10/10** cases, from a French natural-language brief, with no human edits to the input files. See [FireBench](#firebench) for exactly what that does and does not mean.

SmokeAgent is a fork of [Foam-Agent](https://github.com/csml-rpi/Foam-Agent) (MIT, Ling Yue et al., RPI) retargeted from OpenFOAM to [FDS 6.11.1](https://github.com/firemodels/fds) (NIST, public domain). The agent graph, retrieval corpus layout and LLM service are Foam-Agent's; everything under `src/fds/`, `firebench/`, `configs/` and `app/` is new. See [NOTICE](NOTICE).

## What it does

```
French/English brief ──► planner (LLM)  ──► ScenarioSpec (pydantic, units m/s/kW/°C)
                                                │
                    deterministic writer ◄──────┘        .fds (mesh, fire ramp, walls, vents, probes, slices)
                                                │
                          3-layer validation ───┤        structural · engineering (D*/dx, HRRPUA, make-up air) · FDS T_END=0
                                                │
                           FDS 6.11.1 (MPI) ────┤        progress.json, HRR check (plateau median + starvation)
                                                │
                     reviewer (LLM) ◄── errors ─┘        RFC 6902 JSON patch on the ScenarioSpec, never on the .fds
                                                │
                   criteria (no LLM) ───────────┤        ASET vs RSET per escape path, thresholds YAML with sources
                                                │
                              report ───────────┘        Smokeview slices, curves, French template
```

Design rules that fell out of the benchmark:

- **The LLM never writes FDS namelists.** It fills a typed `ScenarioSpec`; a pure function writes the `.fds`. Every geometry rule that FDS enforces silently (grid snapping, mesh interfaces, surface precedence) lives in that function once, with a unit test.
- **Repairs are JSON patches on the spec**, so a fix is reviewable and cannot corrupt the input file.
- **Acceptance criteria contain no LLM.** Visibility, temperature, CO, layer height thresholds are a YAML file with a source line per value (`configs/thresholds_fr_default.yaml`); ASET/RSET is computed from the FDS device output.
- **Materials, not INERT.** Walls default to 13 mm plasterboard with thermal properties from NBSIR 88-3752; INERT (isothermal cold walls) under-predicted hot-layer temperatures by 20–60 % on the NBS multi-room tests while getting layer heights right.

## FireBench

FireBench asks one question: *given only the natural-language brief, can the agent reproduce what FDS itself can do on an experiment NIST uses to validate FDS?*

- **Cases** are generated from NIST's own `FDS_validation_dataplot_inputs.csv`: 171 building-smoke cases (NBS multi-room, Steckler, LLNL enclosure, UL/NIST vents, Vettori flat/sloped ceilings, …). Each `case.yaml` lists the experimental columns, the FDS device columns, the metric (max/min/mean, time windows) exactly as NIST's dataplot does.
- **The brief** is a French paragraph drafted by the LLM from the NIST input and reviewed, describing geometry, fire, openings and escape path in words and numbers.
- **The agent run** gets the brief plus NIST's reference `&DEVC` lines (so the same quantities are measured at the same points) and nothing else about the official input file.
- **Four gates:** (1) FDS reads the input and starts; (2) the prescribed fire happened — plateau median of HRR within 10 %, fire not starved for more than half the plateau; (3) each reference quantity is consistent with the experiment within the **FDS Validation Guide 6.11.1, Table 16.1** band for that quantity, |ln(M/E) − ln δ| ≤ 2·√(σ_E² + σ_M²), at least 2/3 of references per case; (4) the ASET/RSET criteria recompute bit-identically from the bundle.
- **Ceiling:** NIST's own input files are run and judged by the same gates. A point FDS itself misses is not held against the agent (secondary rule, ≤ 1.25 × the official error); in the current results no point needed it.

### Results, run tag `2026-10-01`

| case | experiment | agent | NIST input | agent refs in band | repair loops | agent wall time |
| --- | --- | --- | --- | --- | --- | --- |
| nbs_100a | NBS multi-room, 110 kW, open door | PASS | PASS | 8/8 | 0 | 8.9 h |
| nbs_100o | NBS multi-room, 0.1 m vent only | PASS | PASS | 8/8 | 2 | 13 h |
| nbs_100z | NBS multi-room, 3 rooms | PASS | PASS | 8/8 | 0 | 8.1 h |
| steckler_010 | Steckler door flow | PASS | PASS | 1/1 | 1 | 19 h* |
| steckler_011 | Steckler door flow | PASS | PASS | 1/1 | 0 | 0.6 h |
| llnl_03 | LLNL enclosure, 400 kW, HVAC | PASS | PASS | 2/2 | 0 | 0.8 h |
| llnl_04 | LLNL enclosure, 300 kW | PASS | PASS | 2/2 | 0 | 1.0 h |
| ul_nist_vents_test_2 | UL/NIST vents, 2.1 MW | PASS | PASS | 6/6 | 0 | 3.5 h |
| vettori_smooth_open_fast | Vettori flat ceiling | PASS | PASS | 6/6 | 0 | 1.2 h |
| vettori_obstructed_wall_fast | Vettori obstructed ceiling | PASS | PASS | 3/4 | 0 | 1.9 h |

\* first-pass run on the 0.05 m grid NIST uses, judged on its last 60 s; the brief now specifies 0.1 m (steckler_011: 0.6 h). All runs on one laptop (Ryzen 7840HS, no GPU, WSL2), 2–3 MPI ranks per case, overnight. Full `result.json` files, the agent's generated `.fds` inputs and the failed-attempt evidence are in `firebench/results/2026-10-01/`.

**Read this before quoting the numbers.**

- 10 cases out of 171 generated; the rest have not been run (laptop budget: 1–13 h per case).
- The retrieval corpus (`database/fds-6.11`) is built from the NIST validation inputs, **including the benchmark cases**. The planner retrieves "similar validation cases" as examples, so for these 10 cases it can see the official input's style. A held-out variant that excludes the case's own directory from retrieval is the next step; until then the result says "the agent can reproduce FDS from a brief *with* the NIST corpus at hand", not "from scratch".
- Grids are coarser than NIST's on some cases (0.15 m on UL vents, 0.1 m on Steckler vs 0.05 m) because of laptop time; gate 3 compares against the official input run with its own grid.
- The French acceptance thresholds have not yet been reviewed by a practising *ingénieur sécurité incendie*. They are a starting point with sources, not an opinion on French regulation.

### What the benchmark caught that FDS did not

Every failure below produced a clean FDS run with no warning that mattered; only the comparison with experiment exposed it. Fixes are deterministic writer rules, each with a unit test (`tests/unit/test_writer.py`).

| symptom | cause | fix |
| --- | --- | --- |
| HRR = 0 on NBS/UL | 0.1 m burner slab on a 0.2 m grid: FDS collapsed the slab, the FIRE face vanished | burner snapped to the grid, one cell thick, HRRPUA rescaled to keep total HRR |
| fire room at 20 °C after 300 s, corridor never heated | `&HOLE` cut through a wall whose face lay on a mesh interface: FDS left a zero-thickness plate on the neighbour mesh and the door was sealed | obstructions minus holes written as box pieces (jambs + sill), no `&HOLE` |
| layer heights right, temperatures 20–60 % low | all surfaces INERT (isothermal) | material catalogue (gypsum, concrete, ceramic fibre, calcium silicate, fire brick, steel) as building default, per obstruction and per mesh block |
| pedestal hid half the burner face | LLM-placed pedestal snapped onto the burner cells | burner OBST written last (last listed surface wins) |
| HRR "peak" 219 kW vs 110 kW prescribed on NBS 100O | near-closed room: unburnt methane burnt off in a 348 kW burst at 860 s, then the fire died — NIST's own input does the same | HRR gate = plateau median + fraction of plateau starved, not the peak |
| Steckler comparison window unreachable | NIST uses `TIME_SHRINK_FACTOR` and compares 1500–1800 s | per-case steady-state window (last 60 s) |

## Running it

Linux or WSL2 (Ubuntu 24.04 tested), Python 3.12, FDS 6.11.1 on `PATH`, 16 GB RAM is enough.

```bash
# FDS 6.11.1: https://github.com/firemodels/fds/releases  (installs under ~/FDS/FDS6; source its FDS6VARS.sh)
git clone https://github.com/xiao98/smoke-agent && cd smoke-agent
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[all]"

export FOAMAGENT_SOLVER_TARGET=fds-6.11
export FOAMAGENT_MODEL_PROVIDER=anthropic FOAMAGENT_MODEL_VERSION=claude-opus-5-5   # any LangChain provider works
export ANTHROPIC_API_KEY=...                                                          # or ANTHROPIC_BASE_URL for a relay

PYTHONPATH=src python -m pytest tests/unit -q                 # 24 tests, no FDS needed except one T_END=0 smoke test
python src/main.py --prompt_path examples/brief_nbs_100a_fr.txt --output_dir runs/demo   # French brief -> .fds -> FDS -> criteria -> report
./start_ui.sh                                                 # Streamlit demo on :8501 (4 pages, French)
```

FireBench:

```bash
python firebench/build_cases.py --fds_repo ~/work/fds --exp_repo ~/work/exp      # 171 case.yaml from NIST's csv
python firebench/run.py --mode official --only nbs_100a                           # NIST input, judged (the ceiling)
python firebench/run.py --mode agent    --only nbs_100a                           # brief -> SmokeAgent -> judged
python firebench/report.py --run_tag 2026-10-01                                   # markdown table
```

The retrieval corpus for FDS (`database/fds-6.11`, 339 NIST validation inputs + 4 FAISS indices, Qwen3-Embedding-0.6B) is included; `database/script/build_fds_corpus.py` rebuilds it.

## Layout

```
src/fds/            new: spec.py writer.py validate.py runner.py outparse.py criteria.py agent.py namelist.py target.py
src/nodes/, src/*   Foam-Agent graph; one `if is_fds:` branch per node delegates to src/fds/agent.py
firebench/          build_cases.py judge.py run.py report.py write_requirements.py, cases/, results/
configs/            thresholds_fr_default.yaml (with sources), nist_model_uncertainty.yaml (Validation Guide Table 16.1)
app/                Streamlit demo
database/fds-6.11/  retrieval corpus (NIST validation inputs) and FAISS indices
tests/unit/         writer / validate / criteria / judge / patch tests
```

## Roadmap

1. Held-out FireBench (exclude the case's own directory from retrieval); run the remaining cases on a workstation.
2. Report module: Smokeview figures, curves and a French DOCX following the four regulatory questions.
3. Local-model line (Devstral / Qwen) for offices that cannot send briefs to an API.
4. Threshold review with practising fire-safety engineers; Docker image.

## Licensing and attribution

- Foam-Agent: MIT, Copyright (c) 2025 Ling Yue — the LangGraph agent, LLM service, retrieval layout and OpenFOAM code in this repository. The original README is kept as `README.foam-agent.md`.
- SmokeAgent additions (`src/fds/`, `firebench/`, `configs/`, `app/`, `database/fds-6.11/`, this README): MIT, Copyright (c) 2026 XIAO Hao.
- FDS and Smokeview: NIST, public domain. NIST validation inputs and experimental data (`firemodels/fds`, `firemodels/exp`): public domain. Wall material properties: NBSIR 88-3752 as transcribed in the NIST FDS validation inputs. Uncertainty statistics: FDS Validation Guide 6.11.1, Table 16.1.

If you work on performance-based smoke-control studies in France and want to try this on a real brief, open an issue.
