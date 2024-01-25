import openai
import pandas as pd
import argparse
import json
from collections import Counter
from tqdm import tqdm
import backoff
from openai.error import APIError, Timeout, APIConnectionError
import jsonlines
import random

import io
openai.api_key_path = "your_path_to_use_chatgpt_api"

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


PROMPT_DICT = {
    "context": (
        "Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n\n"
        "The detailed criterion is as follows:\n"
        "5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
        "4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
        "3: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
        "2: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
        "1: The response is barely on-topic or completely irrelevant.\n"
        "##\n Instruction: Who is the current prime minister of the UK as of 2023?\n"
        "Output: Boris Johnson was the prime minister of the UK from 2019 - 2022.\n"
        "Perceived utility: 2\n"
        "Explanation: While the output provides a factually correct statement about the UK prime minister from 2019 to 2022, this instruction asks who the prime minister is as of 2023, so it doesn't answer the instruction. Therefore, the utility is 2.\n\n"
        "##\nInstruction: Given a description of a travel destination, recommend 10 tourist attractions with detailed explanations of each. The travel destination is Tokyo, Japan.\n"
        "Output: 'Tokyo is a vibrant city full of exciting tourist attractions. Some must-see sights include the Tokyo Skytree, Tokyo Disneyland, Sensoji Temple, Meiji Shrine, Tsukiji Fish Market, Harajuku, and Shinjuku Gyoen.\n"
        "Perceived utility: 3\n"
        "Explanation: This output doesn't provide descriptions of each attraction and the number of the attractions is also less than 10. While this output partially answers the instructions, it doesn't match the instructions strictly. \n\n"
        "##\nInstruction: {instruction}\n"
        "Output:{output}\n"
    ),
    "bio_context": (
        "Given an instruction and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this score perceived utility.\n\n"
        "The detailed criterion is as follows:\n"
        "5: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.\n"
        "4: The response mostly fulfills the need in the query, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, or improving coherence. \n"
        "3: The response is acceptable, but some major additions or improvements are needed to satisfy users' needs.\n"
        "2: The response still addresses the main request, but it is not complete or not relevant to the query.\n"
        "1: The response is barely on-topic or completely irrelevant.\n"
        "##\n Instruction: Generate a description of the structure and mechanism of action of the enzyme nicotinamide adenine dinucleotide phosphate (NADPH) oxidase.\n"
        "Output: NADPH oxidase is a membrane-bound enzyme that plays an important role in the production of reactive oxygen species. It is composed of six distinct subunits and is found in both the cytosol and the membrane. The subunits are composed of two copies each of the cytosolic subunits: p47 phox, p67 phox, p40 phox, and two copies of the membrane-associated subunits: gp91phox and p22phox. NADPH oxidase functions by donating electrons to the active site of the enzyme, thus allowing it to reduce oxygen molecules to form superoxide anion (O2−). The end product of this reaction is the reactive oxygen species, which is used to fight infection or for other body functions. NADPH oxidase is regulated by various signals such as calcium, nicotinamide adenine dinucleotide phosphate (NADP), and calcium binding proteins. In addition, it is inhibited by certain compounds, such as apocynin, that prevent its activation. NADPH oxidase plays an important role in the body's defenses against infection, and it is thus essential for many biological processes.\n"
        "Perceived utility: 5\n"
        "Explanation: This response provides a descriptive explanation of the structure and function of NADPH oxidase, giving full and comprehensive information. Thus, this is deemed a high utility response satisfying the user's information needs.\n\n"
        "##\n Instruction: Was the enhancement of Lck activity immediate?\n"
        "Output: No, the enhancement of Lck activity did not have an immediate relevance to TCR-mediated signal transduction, as judged from the lack of detectable de novo phosphorylation of zeta-chain and the absence of functional responses of T cells.\n"
        "Perceived utility: 3\n"
        "Explanation: The response robustly addresses the aspect of Lck activity enhancement not being immediate. However, the details provided thereafter are quite dense and technical, and without proper context, they may not be easy to interpret for all readers. Simplifying this information or providing some additional context could improve the perceived utility.\n\n"
        "##\n Instruction: Alcohol intake during pregnancy causes, all, except?\n"
        "Output: Brachycephaly\n"
        "Perceived utility: 5\n"
        "Explanation: This is a concise and correct answer to the posed question, focusing on the exclusion of one potential effect of alcohol intake during pregnancy. The response specifically points out that brachycephaly is not caused by alcohol intake during pregnancy, thus correctly answering the query. There's not much room for more elaborate discussion in this case, making this answer complete and highly useful for the user's information needs in this context.\n\n"
        "##\nInstruction: {instruction}\n"
        "Output:{output}\n"
    )
}


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def postprocess(results):
    raw_output = results["choices"][0]["message"]["content"]
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:")[1]
        score_string = raw_output.split("\nExplanation:")[0]
        score = None
        for i in range(1, 6):
            if str(i) in score_string:
                score = int(i)
        if score is None:
            return "", explanation
        else:
            return score, explanation
    else:
        return "", ""

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
    
def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_name', type=str, nargs='+')
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()
    
    ## use at contriever msmarco setting
    # examples = []
    # mol_instruction = []
    # with open(args.input_file_name + "mol_instruction_qa.json", 'r') as fp:
    #     lines = fp.readlines()
    #     for line in lines:
    #         inst = json.loads(line)
    #         mol_instruction.append(inst)

    # examples.extend(mol_instruction)
    # examples.extend(jload(args.input_file_name + "self_instruct_biomedical.json", "r"))
    # examples.extend(jload(args.input_file_name + "MedInstruct-52k.json", "r"))
    # for item in examples:
    #     # self-BIORAG
    #     if item["input"] in item["instruction"]:
    #         continue
    #     else:
    #         item["instruction"] = item["instruction"] + " " + item["input"]
        
    #     # self-RAG
    #     # if len(item["input"]) > 1:
    #     #     item["instruction"] = item["instruction"] + " " + item["input"]

    ## 11.22 use at medcpt settings
    examples = []
    for input_file in args.input_file_name:
        if input_file.endswith(".json") or input_file.endswith(".json_tmp"):
            examples += json.load(open(input_file))
        else:
            examples += load_jsonlines(input_file)

    result_list = []
    if args.n is not None:
        examples = random.sample(examples, k=args.n)

    for idx, example in tqdm(enumerate(examples)):
        if example["input"] not in example["instruction"]:
            example["instruction"] = example["instruction"] + "\n" + example["input"]
        
        try:
            results = completions_with_backoff(
                model=args.model_name,
                messages=[
                    {"role": "user",
                        "content": PROMPT_DICT["bio_context"].format_map(example)},
                ],
                request_timeout=60,
                max_tokens=200,
            )
            score, explanation = postprocess(results)
            result_list.append({"input": example, "score": score, "explanation": explanation,
                               "raw_output": results["choices"][0]["message"]["content"]})
            if idx % 20 == 0:
                print("Input: {}".format(example["instruction"]))
                print("Output: {}".format(example["output"]))
                print("Score: {0} ({1})".format(score, explanation))

        except (APIError, Timeout, APIConnectionError):
            results = "ERROR: API error outputs"
        if idx % 100 == 0:
            with open(args.output_file_name + "_tmp", "w") as outfile:
                json.dump(result_list, outfile)

    with open(args.output_file_name, "w") as outfile:
        json.dump(result_list, outfile)


if __name__ == "__main__":
    main()
