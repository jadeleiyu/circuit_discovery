# DiscoGP Circuit Discovery


## Installation guide

```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Supporting Models

| Model      | Status | model_download_name      | Note                  |
| ---        | ---    | ---                      |                       |
| gpt2-small | -[x]   | gpt2                     |                       |
| gpt2-xl    | -[x]   | gpt2-xl                  |                       |
| llama-2-7b | -[?]   | meta-llama/Llama-2-7b-hf | Need 80GB+ GPU Memory |
| opt-1.3b   | -[x]   | facebook/opt-1.3b        | 80G: batch_size 32    |
| opt-2.7b   | -[x]   | facebook/opt-2.7b        | 80G: edge only bs 4   |
