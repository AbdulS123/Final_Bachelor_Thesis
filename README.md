# Voice-Cloning-Based Social Engineering

This repository accompanies the Bachelor's thesis **“Voice-Cloning-Based Social Engineering”** and contains the supporting artifacts for the exploratory German-language technical validation in Chapter 7.

The evaluation used three local systems:

- **XTTS-v2** for text-to-speech (TTS)
- **RVC v2** for speech-to-speech voice conversion (STS)
- **Seed-VC** for speech-to-speech voice conversion (STS)

The repository documents the bounded configurations that were tested. It is not a general benchmark of voice-cloning technology and does not measure deception success.

## Contents

- `configs/` — final XTTS-v2, RVC v2, and Seed-VC configurations
- `scripts/` — final generation/conversion and evaluation code; the XTTS folder includes the small local runtime subset required by its published generation script
- `patches/` — the local RVC and Seed-VC modifications used by the evaluated pipelines
- `evaluation/` — the two German scenario texts, privacy-preserving hashes/metadata for non-public material, and blinded-inspection materials
- `results/` — machine-readable final results, ASR transcripts, failure records, timing records, and input-level sensitivity results

## Evaluation design

XTTS-v2 generated each of two fixed German scenario utterances three times, producing six outputs. RVC and Seed-VC converted the same six human source performances: three executive-condition takes and three distressed-customer-condition takes.

The technical evaluation considered Word Error Rate (WER), Character Error Rate (CER), CAMPPlus cosine similarity, Real-Time Factor (RTF), objective output-failure indicators, and a structured single-author perceptual inspection. The public blinded-inspection material consists of the scoring instructions, blank scoring template, and blinding key.

## Software revisions

- Coqui TTS 0.22.0 / XTTS-v2
- RVC v2: `81eed5e8f68b6bed1789f682fe78cdd324495afc`
- Seed-VC: `51383efd921027683c89e5348211d93ff12ac2a8`

The two local implementation changes used in the final RVC and Seed-VC pipelines are provided in `patches/`. Complete upstream repositories and third-party model checkpoints are not redistributed.

## Privacy

Genuine target-speaker recordings, human source recordings, identifiable generated speech, speaker embeddings, speaker-specific RVC weights, and FAISS speaker indices are not published. Non-public material is represented where available by anonymous identifiers, metadata, and SHA-256 hashes in `evaluation/private_material_manifest.csv`.

Because the private speech and speaker-specific artifacts are excluded, this repository supports inspection of the final configurations, code, and reported machine-readable results rather than byte-for-byte reproduction of the original audio outputs. The scripts require the corresponding upstream software and suitable private/replacement speech material for execution.

## Running the published code

The scripts use environment variables instead of machine-specific absolute paths. The main variables are `XTTS_OUTPUT_DIR`, `XTTS_PRIVATE_MODEL_DIR`, `XTTS_PRIVATE_REFERENCE_DIR`, `CAMPPLUS_CHECKPOINT`, `RVC_ROOT`, `RVC_MODEL_DIR`, `RVC_INDEX_ROOT`, `RVC_RMVPE_ROOT`, `RVC_INDEX_FILE`, `SEEDVC_REPO`, `SEEDVC_TARGET_REFERENCE`, `SCENARIO_WS`, `STS_PRIVATE_TARGET_EMBEDDINGS_JSON`, `SOURCE_AUDIO_DIRECTORY`, `SOURCE_AUDIO_DIRECTORY_NORMALIZED`, and `INPUT_LEVEL_WS`.

No private audio, speaker embedding, model weight, or index is included in this repository.
