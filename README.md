# Whispers in the Machine: Confidentiality in LLM-integrated Systems
This is the code repository accompanying our paper [Whispers in the Machine: Confidentiality in LLM-integrated Systems](https://arxiv.org/abs/2402.06922).
> Large Language Models (LLMs) are increasingly integrated with external tools. While these integrations can significantly improve the functionality of LLMs, they also create a new attack surface where confidential data may be disclosed between different components. Specifically, malicious tools can exploit vulnerabilities in the LLM itself to manipulate the model and compromise the data of other services, raising the question of how private data can be protected in the context of LLM integrations.<br><br> In this work, we provide a systematic way of evaluating confidentiality in LLM-integrated systems. For this, we formalize a "secret key" game that can capture the ability of a model to conceal private information. This enables us to compare the vulnerability of a model against confidentiality attacks and also the effectiveness of different defense strategies. In this framework, we evaluate eight previously published attacks and four defenses. We find that current defenses lack generalization across attack strategies. Building on this analysis, we propose a method for robustness fine-tuning, inspired by adversarial training. <br>This approach is effective in lowering the success rate of attackers and in improving the system's resilience against unknown attacks. 

If you want to cite our work, please use the [this](#citation) BibTeX entry.

### This framework was developed to study the confidentiality of Large Language Models (LLMs) in integrated systems. The framework contains several features:
- A set of attacks against LLMs, where the LLM is not allowed to leak a secret key -> [jump](#attacks-and-defenses)
- A set of defenses against the aforementioned attacks -> [jump](#attacks-and-defenses)
- The possibility to test the LLM's confidentiality in dummy tool-using scenarios as well as with the mentioned attacks and defenses -> [jump](#attacks-and-defenses)
- Testing LLMs in real-world tool-scenarios using LangChains Google Drive and Google Mail integrations -> [jump](#real-world-tool-scenarios)
- Creating enhanced system prompts to safely instruct an LLM to keep a secret key safe -> [jump](#generate-system-prompt-datasets)

>[!WARNING]
><b>Hardware aceleration is only fully supported for CUDA machines running Linux. MPS on MacOS should somewhat work but Windows with CUDA could face some issues.</b>

## Setup
Before running the code, install the requirements:
```
python -m pip install --upgrade -r requirements.txt
```
If you want to use models hosted by OpenAI or Huggingface, create both a ```key.txt``` file containing your OpenAI API key as well as a ```hf_token.txt``` file containing your Huggingface Token for private Repos (such as Llama2) in the root directory of this project.

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

## Example Usage
```python
python attack.py --strategy "tools" --scenario "CalendarWithCloud" --attacks "payload_splitting" "obfuscation" --defense "xml_tagging" --iterations 15 --llm_type "llama3-70b" --temperature 0.7 --device cuda --prompt_format "react"
```
Would run the attacks ```payload_splitting``` and ```obfuscation``` against the LLM ```llama3-70b``` in the scenario ```CalendarWithCloud``` using the defense ```xml_tagging``` for 15 iterations with a temperature of 0.7 on a cuda device using the react prompt format in a tool-integrated system.

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
| ```-s, --strategy``` | <b>str</b> | ```None``` | Specifies the strategy for the attack (whether to use normal attacks or ```tools``` attacks) |
| ```-sc, --scenario``` | <b>str</b> | ```all``` | Specifies the scenario for the tool based attacks |
| ```-dx, --device``` | <b>str</b> | ```cpu```| Specifies the device which is used for running the script (cpu, cuda, or mps)
| ```-pf, --prompt_format``` | <b>str</b> | ```react``` | Specifies whether react or tool-finetuned prompt format is used for agents. (react or tool-finetuned) |
| ```-ds, --disable_safeguards``` | <b>bool</b> | ```False``` | Disables system prompt safeguards for tool strategy |
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
| GPT-4 (o1, o1-mini, turbo)| ```gpt-4o``` / ```gpt-4o-mini``` / ```gpt-4-turbo``` | [Link](https://platform.openai.com/docs/models/gpt-4)| OpenAI API |
| LLaMA 2 | ```llama2-7b``` / ```llama2-13b``` / ```llama2-70b``` | [Link](https://huggingface.co/meta-llama) | Local Inference |
| LLaMA 2 hardened | ```llama2-7b-robust``` / ```llama2-13b-robust``` / ```llama2-70b-robust```|  [Link](https://huggingface.co/meta-llama) | Local Inference |
| Llama 3.1 | ```llama3-8b``` / ```llama3-70b``` | [Link](https://huggingface.co/meta-llama) | Local Inference (first: ```ollama pull llama3.1/llama3.1:70b/llama3.1:405b```) |
| Qwen 2.5 | ```qwen2.5-72b``` | [Link](https://qwenlm.github.io/blog/qwen2.5/) | Local Inference (first: ```ollama pull qwen2.5:72b```) |
| Llama 3.2 | ```llama3-1b```/ ```llama3-3b```| [Link](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) | Local Inference (first: ```ollama pull llama3.2/llama3.2:1b```) |
| Reflection Llama | ```reflection-llama```| [Link](https://huggingface.co/mattshumer/ref_70_e3) | Local Inference (first: ```ollama pull reflection```) |
| Vicuna | ```vicuna-7b``` / ```vicuna-13b``` / ```vicuna-33b``` | [Link](https://huggingface.co/lmsys/vicuna-33b-v1.3) | Local Inference |
| StableBeluga (2) | ```beluga-7b``` / ```beluga-13b``` / ```beluga2-70b```| [Link](https://huggingface.co/stabilityai/StableBeluga2) | Local Inference |
| Orca 2 | ```orca2-7b``` / ```orca2-13b``` / ```orca2-70b``` | [Link](https://huggingface.co/microsoft/Orca-2-7b) | Local Inference |
| Gemma | ```gemma-2b``` / ```gemma-7b```| [Link](https://huggingface.co/google/gemma-7b-it) | Local Inference |
| Gemma 2 | ```gemma2-9b``` / ```gemma2-27b```| [Link](https://huggingface.co/blog/gemma2) | Local Inference (first: ```ollama pull gemma2/gemma2:27b```) |
| Phi 3 | ```phi3-3b``` / ```phi3-14b``` | [Link](https://ollama.com/library/phi3) | Local Inference (first: ```ollama pull phi3:mini/phi3:medium```)|

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
|[Typoglycemia](https://twitter.com/lauriewired/status/1682825103594205186?s=20) | ```typoglycemia``` | [PromptGuard](https://huggingface.co/meta-llama/Prompt-Guard-86M)| ```prompt_guard``` |
|[Adversarial Suffix](https://llm-attacks.org/) | ```advs_suffix``` | |
|[Prefix Injection](https://arxiv.org/pdf/2311.16119) | ```prefix_injection``` | |
|[Refusal Suppression](https://arxiv.org/pdf/2311.16119) | ```refusal_suppression``` | |
|[Context Ignoring](https://arxiv.org/pdf/2311.16119) | ```context_ignoring``` | |
|[Context Termination](https://arxiv.org/pdf/2311.16119) | ```context_termination``` | |
|[Context Switching Separators](https://arxiv.org/pdf/2311.16119) | ```context_switching_separators``` | |
|[Few-Shot](https://arxiv.org/pdf/2311.16119) | ```few_shot``` | |
|[Cognitive Hacking](https://arxiv.org/pdf/2311.16119) | ```cognitive_hacking``` | |
|Base Chat | ```base_chat``` | |

The ```base_chat``` attack consists of normal questions to test of the model spills it's context and confidential information even without a real attack.


---
# Finetuning
This section covers the possible LLaMA finetuning options.
We use PEFT, which is based on [this](https://github.com/huggingface/peft) paper.

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
| ```-llm, --llm_type``` | <b>str</b> | ```llama3-8b``` |Specifies the type of llm to finetune |
| ```-i, --iterations``` | <b>int</b> | ```10000``` | Specifies the number of iterations for the finetuning |
| ```-advs, --advs_train``` | <b>bool</b> | ```False``` | Utilizes the adversarial training to harden the finetuned LLM |
| ```-a, --attacks``` | <b>List[str]</b> | ```payload_splitting``` | Specifies the attacks which will be used to harden the llm during finetuning. Only has an effect if ```--train_robust``` is set to True. For supported attacks see the previous section |
| ```-n, --name_suffix``` | <b>str</b> | ```""``` | Specifies a suffix for the finetuned model name |


### Supported Large Language Models
Currently only the LLaMA models are supported (```llama2-7/13/70b``` / ```llama3-8/70b```).


# Generate System Prompt Datasets
Simply run the ```generate_dataset.py``` script to create new system prompts as a json file using LLMs.

### Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```-h, --help``` | - | - | Show this help message and exit |
| ```-llm, --llm_type``` | <b>str</b> | ```llama3-70b``` |Specifies the LLM used to generate the system prompt dataset |
| ```-n, --name_suffix``` | <b>str</b> | ```""``` | Specifies a suffix for the model name if you want to use a custom model |
| ```-ds, --dataset_size``` | <b>int</b> | ```1000``` | Size of the resulting system prompt dataset |


# Real-World Tool Scenarios
To test the confidentiality of LLMs in real-world tool scenarios, we provide the possibility to test LLMs in Google Drive and Google Mail integrations. To do so, run the ```/various_scripts/llm_mail_test.py```script with your Google API credentials.

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
