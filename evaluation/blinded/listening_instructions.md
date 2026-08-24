# Blinded perceptual inspection — Chapter 7 scenario evaluation

This is a STRUCTURED SINGLE-AUTHOR PERCEPTUAL INSPECTION, not a listener study.
One evaluator (the author) inspects each file and records ratings. There is no
sample-size statistics, no listener pool, and no group-level claim.

## Materials used during the inspection
- A private 18-file listening set with neutral IDs `P01.wav` … `P18.wav`.
- The six private original human scenario recordings for delivery calibration.
- Private genuine target-speaker reference audio for resemblance calibration.
- `listening_sheet_template.csv` — the public blank scoring template.
- `blinding_key.csv` — the public mapping from blind IDs to system/scenario/take.

The audio itself is intentionally not published in this repository.

## Scoring scales (enter 1–5 per file in the scoring sheet)

### TARGET-SPEAKER RESEMBLANCE
- 1 = clearly does not resemble the target speaker
- 2 = weak resemblance
- 3 = some recognizable target characteristics
- 4 = clear resemblance
- 5 = strong resemblance to the target

### CONTENT / INTELLIGIBILITY
- 1 = major content or intelligibility failure
- 2 = several important problems
- 3 = understandable but noticeable errors
- 4 = almost completely clear/correct
- 5 = completely clear and correct

### SCENARIO DELIVERY
The delivery that the scenario requires:
- Scenario 1 (executive): firm, impatient, authoritative and time-sensitive delivery.
- Scenario 2 (customer): distressed, overwhelmed and urgent request for assistance.
- 1 = intended delivery absent
- 2 = weak
- 3 = partially conveyed
- 4 = clearly conveyed
- 5 = strongly conveyed

### AUDIO NATURALNESS
- 1 = severe synthetic/metallic/unstable artifacts
- 2 = strong artifacts
- 3 = noticeable artifacts
- 4 = minor artifacts
- 5 = little or no obvious artifact

`short_observation`: one sentence of free text per file (what you noticed), or leave
empty.

## What you must NOT assess
- deception success
- likelihood of compliance
- whether an employee would believe the caller
- real/fake detection accuracy

None of these are part of this inspection, and none of them follow from the four
rated dimensions.

## Procedure
1. Listen to the private source and target references once to calibrate delivery and target voice.
2. Score `P01` … `P12` in ascending order (and `P13` … `P18` if present), one file at
   a time; you may replay a file as often as you like, but score it before moving on.
3. Record all four ratings plus the observation for every file. Do not leave a file
   unrated.
4. Only after scoring, compare your sheet against `blinding_key.csv` if you wish to
   structure the inspection by system afterwards.

The inspection used the final audio outputs without additional normalization, editing, or enhancement.