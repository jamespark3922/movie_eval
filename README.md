# LSMDC - Evaluation Code

Based on evaluation used for dense video captioning: [densevid_eval](https://github.com/ranjaykrishna/densevid_eval).

## Setup
Run `./get_stanford_models.sh` in `coco-caption` to set up SPICE evaluation.

Download the csv files from the LSMDC website to use them as reference.

## Usage
```
python evaluate.py -s YOUR_SUBMISSION_FILE.JSON -o RESULT.JSON -r LSMDC16_annos_test_someone.csv --verbose
```
Your submission file should follow the format used in LSMDC submission [server](https://competitions.codalab.org/competitions/20669).

```bash
[
  {
    "video_id": int,
    "caption": str,
  },
...
]
```
