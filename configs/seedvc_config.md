# Seed-VC — final configuration (Chapter 7 / Appendix A)

Human-readable summary of the fixed Seed-VC configuration used for the final
evaluation (zero-shot voice conversion).

| Item | Value |
| --- | --- |
| Engine | Seed-VC v2 (CFM-DiT + AR with codebooks, VoiceConversionWrapper) |
| Repository | upstream repository @ commit `51383efd921027683c89e5348211d93ff12ac2a8` |
| Architecture | `configs/v2/vc_wrapper.yaml` (mel 80-band / 22050 Hz / hop 256) |
| Checkpoints | `checkpoints/models--Plachta--Seed-VC/snapshots/257283f9f41585055e8f858fba4fd044e5caed6e/v2/{ar_base,cfm_small}.pth` (not exported; upstream snapshot identifier recorded in `evaluation/private_material_manifest.csv`) |
| Final model / config | CFM100 (cfm_small); enabled config `1.0_100_0.7` |
| CFM steps | 100 |
| CFG rate | 1.0 |
| top_p | 0.7 |
| temperature | 0.7 |
| repetition_penalty | 1.5 |
| AR-EOS min_tokens | `int(expected_tokens * 0.97)` (patched into `modules/v2/ar.py`; see `patches/seedvc_min_tokens_ar_eos.patch`) |
| Vocoder | BigVGAN-v2 `nvidia/bigvgan_v2_22khz_80band_256x` (v2, 22.05 kHz) |
| Sample rate | 22050 Hz (16-bit PCM mono output) |
| Target reference | `seedvc_target_reference` (23.33 s, mono 48 kHz; SHA-256 in `evaluation/private_material_manifest.csv`) |
| Inference driver | `scripts/sts/seedvc_inference.py` (one process, model loaded once) |


To apply the AR-EOS patch on a fresh clone at the above commit:
    git apply patches/seedvc_min_tokens_ar_eos.patch

The six inference sources are the same as RVC (see `evaluation/private_material_manifest.csv`).