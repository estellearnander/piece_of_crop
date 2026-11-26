## Interactive prototype (Streamlit)

The `prototype/` folder contains a three-tab, local Streamlit demonstrator for the Piece of Crop experiments: paired RGB vs 13-band classification, model comparison and an illustrative crop-insurer severity matrix.

From the repository root (after installing `environment.yml` and `python -m pip install -e .`):

```bash
python -m pip install -r prototype/requirements.txt
python prototype/preflight.py
python -m streamlit run prototype/app.py
```

See [`prototype/README.md`](prototype/README.md) for complete setup, file requirements, verification and limitations. **Model checkpoints and the full EuroSAT dataset are not included in Git**. Live inference requires matching local checkpoint files and EuroSAT TIFFs; the comparison and insurance tabs can use verified saved test predictions when model weights are not available. Values in the insurer cost matrix are hypothetical severity points, not real financial costs.
