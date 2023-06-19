# Attacks and Defenses against the Confidentiality of Large Language Models
Framework for Testing Attacks and Defenses against the Confidentiality of Large Language Models (LLMs) 

## Usage:
```
python main.py [-h] [-a | --attacks [ATTACK1, ATTACK2, ..]] [-d | --defense DEFENSE]
```

## Example Usage:
```python
python main.py --attacks "prompt_injection" "obfuscation" --defense "sanitization"
```

## Arguments
| Argument | Type | Description |
|----------|------|-------------|
| -a, --attacks | List[str] | specifies the attacks which will be utilized against the LLM |
| -d, --defense | str | specifies the defense for the LLM |

## Supported Attacks and Defenses
| Attack | Defense |
|--------|---------|
|prompt_injection | sanitization |
|obfuscation | prompt_based |
|indirect | advs_training |
|manipulation | sandboxing |
|llm |  |