# Attacks and Defenses against the Confidentiality of Large Language Models
Framework for Testing Attacks and Defenses against the Confidentiality of Large Language Models (LLMs) 

## Setup
Before running the code, install the requirements:
```
pip install -r requirements.txt
```
Create both a ```key.txt``` file containing your OpenAI API key as well as a ```hf_token.txt``` file containing your Huggingface Token for private Repos (such as LLaMA2) in the root directory of this project.

## Usage
```
python run.py [-h] [-a | --attacks [ATTACK1, ATTACK2, ..]] [-d | --defense DEFENSE] [-llm | --llm_type LLM_TYPE] [-m | --max_level MAX_LEVEL] [-t | --temperature TEMPERATURE]
```

## Example Usage
```python
python run.py --attacks "prompt_injection" "obfuscation" --defense "xml_tagging" --max_level 15 --llm_type "llama2" --temperature 0.7
```

## Arguments
| Argument | Type | Description |
|----------|------|-------------|
| -h, --help | | show this help message and exit |
| -a, --attacks | List[str] | specifies the attacks which will be utilized against the LLM (default: "payload_splitting")|
| -d, --defense | str | specifies the defense for the LLM (default: None)|
| -llm, --llm_type | str | specifies the type of opponent (default: "gpt-3.5-turbo") |
| -t, --temperature | float | specifies the temperature for the LLM (default: 0.0) to control the randomness |

## Supported Large Language Models (Chat-Only)
| Model | Paramter Specifier | Link |
|-------|------|-----|
| GPT-3.5-Turbo | gpt-3.5-turbo/gpt-3.5-turbo-0301/gpt-3.5-turbo-0613 | [Link](https://platform.openai.com/docs/models/gpt-3-5)|
| GPT-4 | gpt-4/gpt-4-0613 | [Link](https://platform.openai.com/docs/models/gpt-4)|
| LLaMA2 | llama2 | [Link](https://huggingface.co/meta-llama) |

## Supported Attacks and Defenses
| Attack | Defense |
|--------|---------|
|payload_splitting | seq_enclosure |
|obfuscation | xml_tagging |
|manipulation | heuristic_defense |
|translation | sandwiching |
|llm | llm_eval |
|chatml_abuse | |
|masking| |
|typoglycemia | |