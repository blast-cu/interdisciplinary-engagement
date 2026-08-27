# ollama-prompt

# Ollama Setup on CURC

## Links
- ### [Ollama Website](https://ollama.com/)
- ### [Ollama GitHub](https://github.com/ollama/ollama)
- ### [Ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- ### [Ollama Python Library](https://github.com/ollama/ollama-python)

## Setup

1. In your `projects` directory, create a new directory for Ollama:
    ```bash
    mkdir -p /projects/$USER/ollama
    cd /projects/$USER/ollama
    ```
2. Note the latest Ollama version on [GitHub](https://github.com/ollama/ollama/releases/latest). Download the linux distribution and unzip it:
    ```bash
    export ollama_ver="v0.3.14"
    curl -L https://github.com/ollama/ollama/releases/download/${ollama_ver}/ollama-linux-amd64.tgz -o ollama
    tar -xzvf ollama
   ```
3. Make the binary executable:
    ```bash
    chmod +x ./bin/ollama
    ```
4. To test, start up an interactive job on CURC:
    ```bash
    sinteractive --account=<account_name> --partition=<partition_name> --qos=<qos_name> --time=01:00:00 --ntasks=16 --gres=gpu:1 --mem=20G
    ```
   Note that the `--ntasks` and `--mem` values are just placeholders and not optimized. Adjust them as needed. Make sure you allocate enough GPUs such that you have sufficient VRAM to run your model of choice. <br/><br/>
5. Add the following to your `~/.bashrc` file. Update the paths as needed.
    ```bash
    export PATH="$PATH:/projects/$USER/ollama/bin"
    export OLLAMA_TMPDIR="/scratch/alpine/$USER/ollama_temp"
    export OLLAMA_MODELS="/scratch/alpine/$USER/ollama/models"
    export OLLAMA_HOST="0.0.0.0:9999"
    export OLLAMA_NUM_PARALLEL=1
    export OLLAMA_MAX_LOADED_MODELS=1
    ```
6. Source your `~/.bashrc` file:
    ```bash
    source ~/.bashrc
    ```
7. Make the directories, in case they don't exist:
    ```bash
    mkdir -p $OLLAMA_TMPDIR
    mkdir -p $OLLAMA_MODELS
    ```
8. For testing purposes, start the Ollama server in the background:
    ```bash
    nohup ollama serve > log.txt 2>&1 &
    ```
9. Run and chat with Llama 3.1 in the terminal:
    ```bash
    ollama run llama3.1:8b-instruct-fp16
    ```
10. If this is working, you're good to move on and use this repo as follows!


# quick start

Update `configs/default.yaml` and `prompts/default.json` for your task. Create a `.env` file in the repository and add:

```bash
export DATA_PATH="path/to/data"
```

Navigate to `ollama_utils.py` and fill in any `#TODO` comments. Upadate `runner.sh` and then use it to run the script on CURC!