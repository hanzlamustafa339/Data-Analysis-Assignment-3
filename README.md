# Assignment 3 — Hybrid Biomedical Image Analysis

**7PAM2032 Data Analysis with AI** · modality: synthetic DAPI-like fluorescence
microscopy of cell nuclei (256×256, 80 train / 20 val / 12 test, exact ground
truth).

```
raw image → segmentation → quantitative region features
          → structured JSON record → short narrative
```

## Quick start

**Option A — Colab (recommended; this is the version that runs the real LLMs).**

Open `notebook/assignment3_hybrid_pipeline.ipynb`, set *Runtime → Change runtime
type → T4 GPU*, then *Run all*. The first three cells install Ollama, start the
server and pull `llama3.2-vision`, `llama3.2`, `qwen2.5:3b` and `moondream`
(~12 GB, 10–20 min on a fresh session). Everything after that is self-contained:
the dataset is downloaded from GitHub inside the notebook.

**Option B — local scripts.**

```bash
pip install torch scikit-image pandas matplotlib imageio ollama
python src/task1_eda_vlm.py      # EDA figures + VLM prompt experiments
python src/task2_classical.py    # Otsu + regionprops + numbers-first LLM
python src/task3_unet.py BCE     # train one loss variant (~3 min each)
python src/task3_unet.py Dice
python src/task3_unet.py BCE+Dice
python src/task3_unet.py finalise # ablation table + all Task 3 figures
python src/task4_hybrid.py       # hybrid pipeline + robustness trace
```

The scripts expect the dataset unpacked at `data/` (i.e. `data/train/images/…`).

> **Vision model.** `llama3.2-vision` fails to load on the current Ollama build
> (`unknown model architecture`). Following the module announcement permitting
> alternatives, the notebook probes candidates and uses `qwen2.5vl:7b`. All
> committed outputs come from that model; the text steps use `llama3.2`.
>
> **On the LLM steps in `src/`.** `src/llm.py` talks to a local Ollama server if
> one is reachable and otherwise falls back to a labelled deterministic stub, so
> parsing and auditing can be tested without a GPU. The committed outputs in
> `outputs/` are all from the real models via the notebook.

## Layout

```
notebook/assignment3_hybrid_pipeline.ipynb   the submission notebook (41 cells)
src/core.py            data loading, Otsu+regionprops, U-Net, losses, metrics
src/llm.py             every prompt, the Ollama client, parsers, offline stub
src/task1_eda_vlm.py   Task 1 — preparation, EDA, direct VLM description
src/task2_classical.py Task 2 — classical features, numbers-first description
src/task3_unet.py      Task 3 — training, loss ablation, evaluation, figures
src/task4_hybrid.py    Task 4 — hybrid pipeline, quality gate, robustness
report/report_draft.md the report draft
outputs/figures/       fig1–fig7
outputs/metrics/       EDA, Otsu accuracy, ablation, per-image scores, robustness
outputs/records/       task4_records.csv ← the deliverable, plus raw LLM outputs
models/                trained weights (unet_main.pth = BCE+Dice)
```

## Headline results

| | Dice | IoU |
|---|---|---|
| Otsu (no training) | **0.9679** | **0.9378** |
| U-Net, BCE | 0.9548 | 0.9137 |
| U-Net, BCE+Dice (main) | 0.9414 | 0.8895 |
| U-Net, Dice | 0.9381 | 0.8835 |

*Validation split; U-Net at 128×128, Otsu at 256×256. The main model scores
0.9394 / 0.8858 when its prediction is resized to 256 for a matched comparison,
and 0.9409 Dice on the unseen test split.*

Three findings worth knowing before reading the report:

1. **Otsu beats the U-Net**, on all 20 validation images. Foreground/background
   separability is Cohen's d = 5.58 — near-perfect bimodality is Otsu's best
   case, and 80 images at 20 epochs cannot improve on it.
2. **Both fail at counting.** On dense fields Otsu undercounts by ~34 nuclei and
   the U-Net by ~35, because neither attempts instance separation and Dice is
   blind to whether two touching nuclei form one component or two.
3. **Blur breaks the pipeline; low contrast does not.** Dice 0.95 → 0.61 under
   blur (caught by the quality gate on the dense image), versus no measurable
   change under low contrast, which per-image normalisation inverts.
4. **The audit column only protects copied fields.** `n_objects` matched the
   measurement 12/12, but `density_class` — a judgement the model makes rather
   than copies — labelled a 42-object field `sparse` and a 14-object field
   `dense`.

## Extensions implemented

- **Robustness** — blur and low contrast traced through all five stages
  (`outputs/metrics/robustness_trace.csv`, `fig7`).
- **Loss ablation** — BCE vs Dice vs BCE+Dice under a pinned seed
  (`outputs/metrics/unet_loss_ablation.csv`, `fig4`).
- **Vision-model comparison** — `llama3.2-vision` vs `moondream` on the Task 1
  description step, scored on schema compliance (notebook §9).
- *Not implemented:* MedSAM (foundation-model substitution).

## Provenance

The U-Net, losses, metrics and `segment_image` contract are taken from
`LAB_CNN_unet_segmentation_SOLUTIONS.ipynb` (Unit 11); the Otsu/morphology/
`regionprops` chain from `Lab_ClassicalImageProcessing.ipynb` (Unit 10); the
feature-to-text summariser, `JSON:`/`NARRATIVE:` parsing, quality gate and audit
column from `lab5_hybrid_unet_llm_colab_SOLUTIONS.ipynb` (Unit 12); the prompt
patterns and `parse_model_json` from `lab3_multimodal_biomedical_colab.ipynb`
(Unit 3) and `lab4.ipynb` (Unit 4).
