# Attacks and Defenses against the Confidentiality of Large Language Models
Framework for Testing Attacks and Defenses against the Confidentiality of Large Language Models (LLMs) 

## Usage:
```
python main.py [-h] [-a | --attacks [ATTACK1, ATTACK2, ..]] [-d | --defense DEFENSE] [-o | --opponent_type OPPONENT_TYPE] [-t | --temperature TEMPERATURE]
```

## Example Usage:
```python
python main.py --attacks "prompt_injection" "obfuscation" --defense "sanitization"
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
|prompt_injection | sanitization |
|obfuscation | prompt_based |
|indirect | advs_training |
|manipulation | sandboxing |
|llm |  |
| translation | |