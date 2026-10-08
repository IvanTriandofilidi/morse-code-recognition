# Evaluation and provenance

## Historical experiments

Snapshot `fd58dbbd1dac0c57ab422a7b092d00ac2ca9b898` contains an exploratory notebook using 30,000 labeled rows and a planned 80/10/10 split. The first saved ten-epoch segment ends at training/validation CTC losses 3.239343 / 3.170117. A second segment loads `epoch60.pt` and ends at 0.771444 / 0.720047. The intermediate training and checkpoint are absent.

`reports/historical-training.json` is extracted from saved output text. Its plotting script does not estimate missing epochs. Loss values are not CER or leaderboard scores. No verified full-data CER is claimed for the refactored package.

## Corrections

| Notebook behavior | Package behavior |
|---|---|
| Approximate length `frames // 33` | Exact convolution/pooling lengths |
| Repeats collapsed after removing blanks | Adjacent path repeats collapsed before blank removal |
| Shared augmented dataset for all splits | Training-only masking |
| Dropped final validation/prediction batch | All rows retained |
| Test metadata truncated to 4,983 rows | Every supplied manifest row processed |
| Test CTC loss passed raw logits | Training CTC receives log-softmax; evaluation uses edit metrics |
| Impossible CTC alignments silently zeroed | Explicit target-length/repeat feasibility check |
| Audio/labels paired by row order | Explicit ID/path manifest |

The default frontend now uses `power_to_db`; historical amplitude conversion is available as an explicit compatibility mode. Original state dictionaries do not directly load because module names and checkpoint formats changed. Weight-key conversion and prediction parity must be checked before claiming checkpoint compatibility.

## Metrics and boundaries

Corpus CER is total Levenshtein edits divided by total reference characters, including spaces. Mean utterance CER averages individual ratios. Exact match is the fraction of fully correct transcripts. CER may exceed 1 when insertions dominate. Decoding stops at each example's valid output length.

Evaluation reports how many input audio hashes appeared during training or validation selection. Zero exact-file overlap does not prove that source sessions or messages are independent. Keep the final holdout out of fitting and model selection.

Synthetic signals and smoke tests demonstrate execution, not real-world radio accuracy. The repository does not include production weights, calibrated confidence, beam search, or an English output alphabet. Full-data training and a frozen independent evaluation remain the next experimental steps.
