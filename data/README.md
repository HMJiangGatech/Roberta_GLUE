# Glue DATA

```bash
python download_glue_data.py
./download_WSC.sh
./preprocess_GLUE_task.sh glue_data ALL
```

WSC for glue:
```bash
# requirement
pip install spacy
pip install scaremoses
python -m spacy download en_core_web_lg
# download data
./download_WSC.sh
```

# Super Glue Data

```bash
./download_superGlue_data.sh
python preprocess_superGLUE_data.py --tasks <task_name>
```

# ANLI

``` bash
python download_glue_data.py --tasks ANLI
./clean_ANLI.sh glue_data
```
Combined data:
``` bash
python download_glue_data.py --tasks ANLI,MNLI,SNLI
./preprocess_NLI.sh glue_data "MNLI SNLI"
```
