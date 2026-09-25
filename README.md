# social_filters

Analysis code for two-photon calcium-imaging experiments investigating how
zebrafish dorsal-thalamic neurons respond to the spatial and temporal features
of bout-like visual motion. The repository also contains stimulus-generation
and stimulus-timing tools used by these analyses.

## Repository structure

- `src/`: reusable loading, analysis, timing, and plotting code.
- `notebooks/calcium/`: preprocessing, shared, and experiment-specific analyses.
- `notebooks/stimuli/`: stimulus inspection and reporting notebooks.
- `scripts/stimuli/generation/`: stimulus-generation programs.
- `scripts/stimuli/playback/`: stimulus presentation wrappers.
- `configs/stimuli/`: stimulus and experiment configuration files.
- `docs/`: reproducibility and project documentation.
- `archive/notebooks/calcium/`: historical notebooks that are not maintained.

## Local data

Data are not included in this repository. In the lab workspace, source imaging
data are under `Lab/Data/2p`, source stimulus files are under
`Lab/Data/stimuli`, and the organized experiment inventory and derived review
outputs are under `Lab/Results/Paper/2p_derived`. See that folder's `README.md`
for its current experiment and fish inventory.

## Environment

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate social_filters
nbstripout --install
```

## Status

This repository is being prepared for reproducible publication analyses. The
environment and end-to-end analysis entry points are still being standardized.

Notebook outputs are removed from commits with `nbstripout`; its Git filter
must be installed separately in each clone with the command above.
