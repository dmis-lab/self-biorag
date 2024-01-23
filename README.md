# Self-BioRAG: Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models

This is a repository for [Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models]()
 by Minbyul Jeong, Jiwoong Sohn, Mujeen Sung, and Jaewoo Kang.

[7B Model](https://huggingface.co/selfbiorag/selfbiorag_7b) | [13B Model](https://huggingface.co/selfbiorag/selfbiorag_13b) | [Paper]() | [Training data]() | [Summary]() | [Critic Model](https://huggingface.co/selfbiorag/selfbiorag_7b_critic)

**Self-BioRAG** is a framework reliable for biomedical and clinical text instructions that specializes in self-reflection to retrieve, criticize, and generate explanations while preserving generation quality and reasoning ability.

The retrieval-augmented generation (RAG) framework performs document searches unconditionally, regardless of whether a query requires document retrieval. However, **Self-BioRAG** framework use domain-specific instruction-tuned language model to predict whether retrieval is necessary for given query. If the query doesn't require any retrieval of knowledge (factual contents), it directly predicts the answer. If the query necessitates retrieval knowledge, **Self-BioRAG** utilizes the domain-specific retriever (MedCPT, in our case) to retrieve relevant documents. After retriever the top-k evidence, the language model selects the most pertinent evidence for the query. Ultimately, our language model is employed to select the best evidence and generate the answer based on the selected evidence and encoded knowledge.

![](figures/intro_figure.png)

## Updates
* \[**Jan 26, 2024**\] **Self-BioRAG** has been released.

## Content
1. [Installation](#installation)
2. [Quick Usage](#quick-usage)
3. [Overall Workflow](#overall-workflow)
4. [Datasets](#datasets)
5. [Retriever](#retriever)
6. [Critic LM](#critic-lm)
7. [Generator LM](#generator-lm)
8. [Inference](#inference)
9. [FAQ](#faq)
10. [Citation](#citation)
11. [Contact Information](#contact-information)

## Installation
Please create a conda environment by running the command below.

```
conda env create -f selfbiorag.yaml
conda activate selfbiorag
```

If you try to install Python libraries through requirements by running the command below.
```
conda create -n selfbiorag python=3.10
conda activate selfbiorag
pip install -r requirements.txt
```

## Quick Usage
You can download Self-BioRAG from HuggingFace hub.
For inference, we recommend using [vllm](https://vllm.readthedocs.io/en/latest/) to speed up inference.
```py
from vllm import LLM, SamplingParams

model = LLM("selfbiorag/selfbiorag_7b", download_dir=your_download_path_to_load, dtype="half")
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

query_1 = "Classify the given radiology report according to which part of the body it is related to (e.g., chest, abdomen, brain, etc). The intervertebral discs at L4-L5 and L5-S1 are showing signs of degeneration with slight bulging impinging on the adjacent nerve root"
query_2 = "Summarize the key points about the role of BRCA1 and BRCA2 gene mutation in increased risk for breast cancer."
queries = [query_1, query_2]

preds = model.generate([query for query in queries], sampling_params)
for pred in preds:
    print ("Model prediction: ", pred.outputs[0].text)
```

Output
```
Model prediction: 
Model prediction: 
```



## Overall Workflow
Overview of our **Self-BioRAG** process:data construction, training, and inference of Critic LM and Generator LM.
We construct 120k bioemdical instruction sets using two off-the-shelf instruction sets [Mol-Instruction](https://github.com/zjunlp/Mol-Instructions) and [MedInstruct](https://github.com/XZhang97666/AlpaCare/tree/master) and one self-generated biomedical instruction set. 
![](figures/example_figure.png)

## Datasets
Download our overall data: [Instruction-Sets](http://nlp.dmis.korea.edu/projects/selfbiorag-jeong-et-al-2024/data/instruction.tar.gz), [Retriever](http://nlp.dmis.korea.edu/projects/selfbiorag-jeong-et-al-2024/data/retriever.tar.gz), [Critic-LM](http://nlp.dmis.korea.edu/projects/selfbiorag-jeong-et-al-2024/data/critic.tar.gz), [Generator-LM](http://nlp.dmis.korea.edu/projects/selfbiorag-jeong-et-al-2024/data/generator.tar.gz).
You will need ~10.5GB for all process of datasets.

```
mkdir data
cd data
wget http://nlp.dmis.korea.edu/projects/selfbiorag-jeong-et-al-2024/data/instruction.tar.gz
wget http://nlp.dmis.korea.edu/projects/selfbiorag-jeong-et-al-2024/data/retriever.tar.gz
wget http://nlp.dmis.korea.edu/projects/selfbiorag-jeong-et-al-2024/data/critic.tar.gz
wget http://nlp.dmis.korea.edu/projects/selfbiorag-jeong-et-al-2024/data/generator.tar.gz
```

```
tar -zxvf instruction.tar.gz
tar -zxvf retriever.tar.gz
tar -zxvf critic.tar.gz
tar -zxvf generator.tar.gz
```

## Retriever
retriever

## Critic LM
The process of data creation for critic language model is composed of 6 steps as follows.

1. Create Retrieval Data for Three Reflective tokens: Retreival, Relevance, and Supportive
```
cd data_creation/critic/gpt4_reward
$ export DATA_PATH=path_to_your_data_folder
python create_retrieval_data.py --input_files $DATA_PATH/instruction/all_biomedical_instruction.json --output_file $DATA_PATH/retriever/medcpt_top10_evidence_createret.json --multiple_sent --initial_retrieval_file $DATA_PATH/retriever/medcpt_top10_evidence.json
```

2. Make Utility Token Data
```
python chatgpt_utility.py --input_file_name $DATA_PATH/critic/critic_5k_utility.json --model_name gpt-4 --output_file_name $DATA_PATH/critic/critic_utility_chatgpt.json
```

3. Make Retrieval Token Data
```
python chatgpt_need_retrieval.py --input_files $DATA_PATH/critic/critic_5k_retrieval.json --output_file_name $DATA_PATH/critic/critic_retrieval_chatgpt.json --model_name gpt-4 --multi_retrieval --three_way
```

4. Make Relevance Token Data
```
python chatgpt_relevance.py --input_files $DATA_PATH/critic/critic_5k_retrieval.json --output_file_name $DATA_PATH/critic/critic_relevance_chatgpt.json --model_name gpt-4 --multi_retrieval
```

5. Make Supportive Token Data
```
python chatgpt_groundness.py --input_files $DATA_PATH/critic/critic_5k_retrieval.json --output_file_name $DATA_PATH/critic/critic_groundness_chatgpt.json --model_name gpt-4 --multi_retrieval
```

6. Combine GPT-4 API Calls Reward
```
python combine_chat_gpt_reward.py --ut_file $DATA_PATH/critic/critic_utility_chatgpt.json --multi_ret_file $DATA_PATH/critic/critic_retrieval_chatgpt.json --multi_ground_file $DATA_PATH/critic/critic_groundness_chatgpt.json --rel_file $DATA_PATH/critic/critic_relevance_chatgpt.json --output_file_name $DATA_PATH/critic/bio_critic_train
```

After constructing the train dataset, we train the Critic LM as follows.

* Training
```
cd ..
mkdir critic_lm
torchrun --nproc_per_node=8 --master_port 2569 train_special_tokens.py --model_name_or_path selfrag/self_rag_critic --data_path $DATA_PATH/critic/bio_critic_train.json --bf16 True --output_dir critic_lm/selfbiorag_7b_critic/ --num_train_epochs 3 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy no --save_strategy steps --save_steps 300 --save_total_limit 50 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.01 --lr_scheduler_type cosine --logging_steps 10 --fsdp "full_shard auto_wrap"
```

## Generator LM
* Data Construction

* Training

## Inference
inference 

## FAQ
FAQ

## Citation
Citation

## Contact Information
For help or issues using **Self-BioRAG**, please submit a GitHub issue. Please contact Minbyul Jeong (`minbyuljeong (at) korea.ac.kr`) for communication related to **Self-BioRAG**.



