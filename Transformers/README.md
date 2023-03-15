# HuggingFace Access Tokens

Before running the scripts, please generate your access tokens from HuggingFace and create an access_tokens.json file with the schema:
```json
{
    "upload_token": "...",
    "download_token": "..."
}
```

# Datasets

As the datasets are too big to store in GitHub, they are stored in HuggingFace. When you run the script to load the datasets for the first time, they will be downloaded locally from HuggingFace and cached in your local system at `~/.cache/huggingface/datasets`

The links to the datasets are provided below:
- yhavinga/ccmatrix : https://huggingface.co/datasets/yhavinga/ccmatrix
- opus100 : https://huggingface.co/datasets/opus100
- 
