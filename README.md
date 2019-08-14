## Base

| Model              | MNLI    | QNLI | QQP       | RTE  | SST-2| MRPC      | CoLA | STS-B|
|---|---|---|---|---|---|---|---|---|
|`roberta.base` dev  |         | 92.6 | 91.8,89.0 |      | 96.0 |           | 59.8 |      |
| forget        dev  |         | 92.6 |           |      | 96.0 |           | 60.8 |      |




## Reported Results

##### Results on GLUE tasks (dev set, single model, single-task finetuning)

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`roberta.base` | 87.6 | 92.8 | 91.9 | 78.7 | 94.8 | 90.2 | 63.6 | 91.2
`roberta.large` | 90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4
`roberta.large.mnli` | 90.2 | - | - | - | - | - | - | -




## MTVAT
## Base

| Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B|
|---|---|---|---|---|---|---|---|---|
|`roberta.base`  | 87.6, ? | 92.8 | 91.9 | 78.7 | 94.8 | 90.2      | 63.6 | 91.2|
|ours            |         | 92.8 |      | 82.6 | 95.8 | 90.2,84.5 |      |     |
|`roberta.large` | 90.2    | 94.7 | 92.2 | 86.6 | 96.4 | 90.9      | 68.0 | 92.4|


## Large

|method          | MNLI-match | MNLI-mismatch | QNLI | QQP   | RTE   | SST-2    | MRPC         | CoLA          | STS-B  |
|----------------|------------|---------------|------|-------|-------|----------|--------------|---------------|--------|
|baseline (dev)  |0.9024      |0.9021         |0.9478|0.9239 |0.9132 |0.9643    |0.9038,0.8539 |0.8636,0.6660  |0.019   |
|baseline (test) |            |               |      |       |       |          |0.8876,0.8391 |               |        |
|ours            |            |               |      |       |0.9167 |          |0.9255,0.8819 |0.8740,0.6919  |        |
|ours            |            |               |      |       |       |0.9704(t) |0.8986,0.8453 |               |        |
