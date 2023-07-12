# Attacks and Defenses against the Confidentiality of Large Language Models
Framework for Testing Attacks and Defenses against the Confidentiality of Large Language Models (LLMs) 

## Setup
Before running the code, please install the requirements:
```
pip install -r requirements.txt
```
and also create a ```key.txt``` file containing your OpenAI API key in the root directory of the project.

## Usage
```
python main.py [-h] [-a | --attacks [ATTACK1, ATTACK2, ..]] [-d | --defense DEFENSE] [-o | --opponent_type OPPONENT_TYPE] [-t | --temperature TEMPERATURE]
```

## Example Usage
```python
python main.py --attacks "prompt_injection" "obfuscation" --defense "xml_tagging"
```

## Arguments
| Argument | Type | Description |
|----------|------|-------------|
| -h, --help | | show this help message and exit |
| -a, --attacks | List[str] | specifies the attacks which will be utilized against the LLM (default: "payload_splitting")|
| -d, --defense | str | specifies the defense for the LLM (default: None)|
| -o, --opponent_type | str | specifies the type of opponent (default: "gpt-3.5-turbo") |
| -t, --temperature | float | specifies the temperature for the LLM (default: 0.0) to control the randomness |


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