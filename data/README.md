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
python -m spacy download en_core_web_lg
# download data
./download_WSC.sh
```

# Super Glue Data

```bash
./download_superGlue_data.sh
python preprocess_superGLUE_data.py --tasks <task_name>
```
