# RVC — final configuration (Chapter 7 / Appendix A)

Human-readable summary of the fixed RVC configuration used for the final
evaluation.

| Item | Value |
| --- | --- |
| Engine | RVC v2 |
| Repository | upstream repository @ commit `81eed5e8f68b6bed1789f682fe78cdd324495afc` |
| Final model | `RVC_Audit8_e800_s52800.pth` (binary excluded; SHA-256 in `evaluation/private_material_manifest.csv`) |
| Model type | output sample rate 40000 Hz, HuBERT base 768-dim features, trained with RMVPE F0 |
| Training | 800 epochs / 52,800 steps, 513 slices / ~31.59 min prepared speech from four raw target-speaker recordings (158.98 min) |
| F0 during training | RMVPE |
| F0 during final inference | Harvest (patched into upstream; see `patches/rvc_harvest_f0.patch`) |
| Pitch shift | +2 semitones (fixed, not scenario-adapted) |
| FAISS index | ADDED index `added_IVF2416_Flat_nprobe_1_RVC_Audit8_v2.index` (IVF2416, nprobe 1) |
| Index rate | 0.75 |
| Protect | 0.33 |
| RMS mix rate | 1.0 |
| Resample SR | 0 (native 40000) |
| Output | WAV 16-bit PCM, 40000 Hz mono |
| Inference driver | `scripts/sts/rvc_inference.py` (one process, model + index loaded once) |

Note: the FAISS index (`added_..._v2.index`) contains speaker-specific retrieval
material and is NOT exported; only its SHA-256 is recorded in the `evaluation/private_material_manifest.csv`.

To apply the Harvest patch on a fresh clone at the above commit:
    git apply patches/rvc_harvest_f0.patch

The final six inference sources are the three executive-condition takes and
three distressed-customer-condition takes (see `evaluation/private_material_manifest.csv`).