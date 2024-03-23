# Whispers in the Machine: Confidentiality in LLM-integrated Systems
This is the code repository accompanying our paper [Whispers in the Machine: Confidentiality in LLM-integrated Systems](https://arxiv.org/abs/2402.06922).
> Large Language Models (LLMs) are increasingly integrated with external tools. While these integrations can significantly improve the functionality of LLMs, they also create a new attack surface where confidential data may be disclosed between different components. Specifically, malicious tools can exploit vulnerabilities in the LLM itself to manipulate the model and compromise the data of other services, raising the question of how private data can be protected in the context of LLM integrations.<br><br> In this work, we provide a systematic way of evaluating confidentiality in LLM-integrated systems. For this, we formalize a "secret key" game that can capture the ability of a model to conceal private information. This enables us to compare the vulnerability of a model against confidentiality attacks and also the effectiveness of different defense strategies. In this framework, we evaluate eight previously published attacks and four defenses. We find that current defenses lack generalization across attack strategies. Building on this analysis, we propose a method for robustness fine-tuning, inspired by adversarial training. <br>This approach is effective in lowering the success rate of attackers and in improving the system's resilience against unknown attacks. 

If you want to cite our work, please use the [this](#citation) BibTeX entry.

### This framework was developed to study the confidentiality of Large Language Models (LLMs). The framework contains several features:
- A set of attacks against LLMs, where the LLM is not allowed to leak a secret key -> [jump](#attacks-and-defenses)
- A set of defenses against the aforementioned attacks -> [jump](#attacks-and-defenses)
- Creating enhanced system prompts to safely instruct an LLM to keep a secret key safe -> [jump](#generate-system-prompt-datasets)
- Finetuning and Prefix-Tuning LLMs to harden them against these attacks using the datasets -> [jump](#finetuning-peft-and-prefix-tuning)

>[!WARNING]
><b>GPU utilization is based on the [accelerate](https://huggingface.co/docs/accelerate/index) library and is therefore currently not supported for Windows machines</b>

## Setup
Before running the code, install the requirements:
```
python -m pip install --upgrade -r requirements.txt
```
Create both a ```key.txt``` file containing your OpenAI API key as well as a ```hf_token.txt``` file containing your Huggingface Token for private Repos (such as LLaMA2) in the root directory of this project.

Sometimes it can be necessary to login to your Huggingface account via the CLI:
```
git config --global credential.helper store
huggingface-cli login
```

### Distributed Training
All scripts are able to work on multiple GPUs/CPUs using the [accelerate](https://huggingface.co/docs/accelerate/index) library. To do so, run:
```
accelerate config
```
to configure the distributed training capabilities of your system and start the scripts with:
```
accelerate launch [parameters] <script.py> [script parameters]
```

# Attacks and Defenses
## Usage
```
python attack.py [-h] [-a | --attacks [ATTACK1, ATTACK2, ..]] [-d | --defense DEFENSE] [-llm | --llm_type LLM_TYPE] [-m | --iterations ITERATIONS] [-t | --temperature TEMPERATURE]
```

## Example Usage
```python
python attack.py --attacks "payload_splitting" "obfuscation" --defense "xml_tagging" --iterations 15 --llm_type "llama2-7b" --temperature 0.7
```

## Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```-h, --help``` | - | - | show this help message and exit |
| ```-a, --attacks``` | <b>List[str]</b> | ```payload_splitting``` | specifies the attacks which will be utilized against the LLM |
| ```-d, --defense``` | <b>str</b> | ```None``` | specifies the defense for the LLM |
| ```-llm, --llm_type``` | <b>str</b> | ```gpt-3.5-turbo``` | specifies the type of opponent |
| ```-le, --llm_guessing``` | <b>bool</b> | ```False``` | specifies whether a second LLM is used to guess the secret key off the normal response or not|
| ```-t, --temperature``` | <b>float</b> | ```0.0``` | specifies the temperature for the LLM to control the randomness |
| ```-cp, --create_prompt_dataset``` | <b>bool</b> | ```False``` | specifies whether a new dataset of enhanced system prompts should be created |
| ```-cr, --create_response_dataset``` | <b>bool</b> | ```False``` | specifies whether a new dataset of secret leaking responses should be created |
| ```-i, --iterations``` | <b>int</b> | ```10``` | specifies the number of iterations for the attack |
| ```-n, --name_suffix``` | <b>str</b> | ```""``` | Specifies a name suffix to load custom models. Since argument parameter strings aren't allowed to start with '-' symbols, the first '-' will be added by the parser automatically |

The naming conventions for the models are as follows:
```python
<model_name>-<param_count>-<robustness>-<attack_suffix>-<custom_suffix>
```
e.g.:
```python
llama2-7b-robust-prompt_injection-0613
```
If you want to run the attacks against a prefix-tuned model with a custom suffix (e.g., ```1000epochs```), you would have to specify the arguments a follows:
```python
... --model_name llama2-7b-prefix --name_suffix 1000epochs ...
```


## Supported Large Language Models
| Model | Parameter Specifier | Link | Compute Instance |
|-------|------|-----|-----|
| GPT-3.5-Turbo | ```gpt-3.5``` / ```gpt-3.5-turbo``` | [Link](https://platform.openai.com/docs/models/gpt-3-5)| OpenAI API |
| GPT-4-Turbo | ```gpt-4``` / ```gpt-4-turbo``` | [Link](https://platform.openai.com/docs/models/gpt-4)| OpenAI API |
| LLaMA2 (chat) | ```llama2-7b``` / ```llama2-13b``` / ```llama2-70b``` | [Link](https://huggingface.co/meta-llama) | Local Inference |
| LLaMA2 (base) | ```llama2-7b-base``` / ```llama2-13b-base``` / ```llama2-70b-base``` | [Link](https://huggingface.co/meta-llama) | Local Inference |
| LLaMA2 (chat) Finetuned | ```llama2-7b-finetuned``` / ```llama2-13b-finetuned``` / ```llama2-70b-finetuned``` | [Link](https://huggingface.co/meta-llama) | Local Inference |
| LLaMA2 (chat) hardened | ```llama2-7b-robust``` / ```llama2-13b-robust``` / ```llama2-70b-robust```|  [Link](https://huggingface.co/meta-llama) | Local Inference |
| Vicuna | ```vicuna-7b``` / ```vicuna-13b``` / ```vicuna-33b``` | [Link](https://huggingface.co/lmsys/vicuna-33b-v1.3) | Local Inference |
| StableBeluga (2) | ```beluga-7b``` / ```beluga-13b``` / ```beluga2-70b```| [Link](https://huggingface.co/stabilityai/StableBeluga2) | Local Inference |
| Orca 2 | ```orca2-7b``` / ```orca2-13b``` / ```orca2-70b``` | [Link](https://huggingface.co/microsoft/Orca-2-7b) | Local Inference |
| Gemma | ```gemma-2b``` / ```gemma-7b```| [Link](https://huggingface.co/google/gemma-7b-it) | Local Inference |

(Finetuned or robust/hardened LLaMA models first have to be generated using the ```finetuning.py``` script, see below)

## Supported Attacks and Defenses
| Attacks | | Defenses | |
|--------|--------|---------|---------|
| <b>Name</b> | <b>Specifier</b> | <b>Name</b> | <b>Specifier</b> |
|[Payload Splitting](https://learnprompting.org/docs/prompt_hacking/offensive_measures/payload_splitting) | ```payload_splitting``` | [Random Sequence Enclosure](https://learnprompting.org/docs/prompt_hacking/defensive_measures/random_sequence) | ```seq_enclosure``` |
|[Obfuscation](https://learnprompting.org/docs/prompt_hacking/offensive_measures/obfuscation) | ```obfuscation``` |[XML Tagging](https://learnprompting.org/docs/prompt_hacking/defensive_measures/xml_tagging) | ```xml_tagging``` |
|[Jailbreak](https://learnprompting.org/docs/prompt_hacking/jailbreaking) | ```jailbreak``` |[Heuristic/Filtering Defense](https://learnprompting.org/docs/prompt_hacking/defensive_measures/filtering) | ```heuristic_defense``` |
|Translation | ```translation``` |[Sandwich Defense](https://learnprompting.org/docs/prompt_hacking/defensive_measures/sandwich_defense) | ```sandwiching``` |
|[ChatML Abuse](https://www.robustintelligence.com/blog-posts/prompt-injection-attack-on-gpt-4) | ```chatml_abuse``` | [LLM Evaluation](https://learnprompting.org/docs/prompt_hacking/defensive_measures/llm_eval) | ```llm_eval``` |
|[Masking](https://learnprompting.org/docs/prompt_hacking/offensive_measures/obfuscation) | ```masking``` | [Perplexity Detection](https://arxiv.org/abs/2308.14132) | ```ppl_detection```
|[Typoglycemia](https://twitter.com/lauriewired/status/1682825103594205186?s=20) | ```typoglycemia``` | |
|[Adversarial Suffix](https://llm-attacks.org/) | ```advs_suffix``` | |
|Base Chat | ```base_chat``` | |

The ```base_chat``` attack consists of normal questions to test of the model spills it's context and confidential information even without a real attack.


---
# Finetuning (PEFT and Prefix-Tuning)
This section covers the possible LLaMA2 finetuning options.
PEFT is based on [this](https://github.com/huggingface/peft) paper and Prefix-Tuning is based on [this](https://arxiv.org/abs/2101.00190) paper.

### Setup
Additionally to the above setup run
```bash
accelerate config
```
to configure the distributed training capabilities of your system. And
```bash
wandb login
```
with your WandB API key to enable logging of the finetuning process.

---
## Parameter Efficient Finetuning to harden LLMs against attacks or create enhanced system prompts
The first finetuning option is on a dataset consisting of system prompts to safely instruct an LLM to keep a secret key safe. The second finetuning option (using the ```--train_robust``` option) is using system prompts and adversarial prompts to harden the model against prompt injection attacks.

### Usage
```python
python finetuning.py [-h] [-llm | --llm_type LLM_NAME] [-i | --iterations ITERATIONS] [-a | --attacks ATTACKS_LIST] [-n | --name_suffix NAME_SUFFIX]
```

### Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```-h, --help``` | - | - | Show this help message and exit |
| ```-llm, --llm_type``` | <b>str</b> | ```llama2-7b``` |Specifies the type of llm to finetune |
| ```-i, --iterations``` | <b>int</b> | ```10000``` | Specifies the number of iterations for the finetuning |
| ```-advs, --advs_train``` | <b>bool</b> | ```False``` | Utilizes the adversarial training to harden the finetuned LLM |
| ```-a, --attacks``` | <b>List[str]</b> | ```payload_splitting``` | Specifies the attacks which will be used to harden the llm during finetuning. Only has an effect if ```--train_robust``` is set to True. For supported attacks see the previous section |
| ```-n, --name_suffix``` | <b>str</b> | ```""``` | Specifies a suffix for the finetuned model name |


### Supported Large Language Models
| Model | Parameter Specifier | Link | Compute Instance |
|-------|------|-----|-----|
| LLaMA2 (chat) | ```llama2-7b``` / ```llama2-13b``` / ```llama2-70b``` | [Link](https://huggingface.co/meta-llama) | Local Inference |


---
## Prefix-Tuning to harden LLMs against attacks
This is a new and more efficient approach to finetune LLMs. The prefix-tuning script uses huggingfaces accelerator for distributed training, so make sure to run ```accelerate config``` before running the script.

### Usage
```python
accelerate launch prefix_tuning.py [-h] [-llm | --llm_type LLM_NAME] [-i | --epochs EPOCHS] [-bs | --batch_size BATCH_SIZE] [-lr | --learning_rate LEARNING_RATE] [-ml | --max_seq_length MAX_SEQ_LENGTH] [-pl | --prefix_length PREFIX_LENGTH] [-a | --attacks ATTACKS_LIST] [-n | --name_suffix NAME_SUFFIX]
```

### Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```-h, --help``` | - | - | Show this help message and exit |
| ```-llm, --llm_type``` | <b>str</b> | ```llama2-7b``` |Specifies the type of llm to prefix tune |
| ```-e, --epochs``` | <b>int</b> | ```10``` | Specifies the number of epochs for the prefix tuning |
| ```-bs, --batch_size``` | <b>int</b> | ```2``` | Specifies the batch size for the prefix tuning |
| ```-lr, --learning_rate``` | <b>float</b> | ```0.0001``` | Specifies the learning rate for the prefix tuning |
| ```-ml, --max_length``` | <b>int</b> | ```1024``` | Specifies the max. sequence length in tokens |
| ```-pl, --prefix_length``` | <b>int</b> | ```10``` | Specifies the number of virtual tokens to train as the tuned prefixes |
| ```-a, --attacks``` | <b>List[str]</b> | ```payload_splitting``` | Specifies the attacks which will be used to harden the llm during finetuning. Only has an effect if ```--train_robust``` is set to True. For supported attacks see the previous section |
| ```-n, --name_suffix``` | <b>str</b> | ```""``` | Specifies a suffix for the finetuned model name |


### Supported Large Language Models
| Model | Parameter Specifier | Link | Compute Instance |
|-------|------|-----|-----|
| LLaMA2 (chat) | ```llama2-7b``` / ```llama2-13b``` / ```llama2-70b``` | [Link](https://huggingface.co/meta-llama) | Local Inference |


# Generate System Prompt Datasets
Simply run the ```generate_dataset.py``` script to create new system prompts as a json file using LLMs.

### Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```-h, --help``` | - | - | Show this help message and exit |
| ```-llm, --llm_type``` | <b>str</b> | ```llama2-7b``` |Specifies the type of llm to prefix tune |
| ```-n, --name_suffix``` | <b>str</b> | ```""``` | Specifies a suffix for the model name if you want to use a custom model |
| ```-ds, --dataset_size``` | <b>int</b> | ```1000``` | Size of the resulting system prompt dataset |

# Citation
If you want to cite our work, please use the following BibTeX entry:
```bibtex
@article{evertz-24-whispers,
	title    =  {{Whispers in the Machine: Confidentiality in LLM-integrated Systems}}, 
	author   =  {Jonathan Evertz and Merlin Chlosta and Lea Schönherr and Thorsten Eisenhofer},
	year     =  {2024},
	journal  =  {Computing Research Repository (CoRR)}
}
``````