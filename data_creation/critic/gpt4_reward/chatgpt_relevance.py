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

openai.api_key_path = "your_path_to_use_chatgpt_api"
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


KNOWLEDGE_INSTRUCTIONS = {"nq": "Please answer the following questions using the shortest possible response. For example, if the question asks 'What is the capital of France?'', you can simply reply with 'Paris'.",
                          "fever": "Determine whether the following statement is true or false.",
                          "wow": "You have been provided with a chat history between two agents, separated by new lines. Generate a response that is informative and engaging based on the latest message."}

PROMPT_DICT = {
    "multi": (
        "You'll be provided with an instruction, along with evidence and possibly some preceding sentences. "
        "When there are preceding sentences, your focus should be on the sentence that comes after them. "
        "Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
        "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
        "###\nInstruction: Given four answer options, A, B, C, and D, choose the best answer.\n\n"
        "Input: Earth rotating causes\n"
        "A: the cycling of AM and PM\nB: the creation of volcanic eruptions\nC: the cycling of the tides\nD: the creation of gravity\n\n"
        "Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.\n\n"
        "Rating: [Relevant]\n"
        "Explanation: The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A.\n\n"
        "###\nInstruction: age to run for us house of representatives\n\n"
        "Evidence: The Constitution sets three qualifications for service in the U.S. Senate: age (at least thirty years of age); U.S. citizenship (at least nine years); and residency in the state a senator represents at the time of election.\n\n"
        "Rating: [Irrelevant]\n"
        "Explanation: The evidence only discusses the ages to run for the US Senate, not for the House of Representatives.\n\n"
        "###\nInstruction: {instruction}\n\n"
        "Evidence: {evidence}\n\n"
        "Rating:"
    ),
    "multi_no_preceding": (
        "You'll be provided with an instruction, along with evidence and possibly some preceding sentences. "
        "When there are preceding sentences, your focus should be on the sentence that comes after them. "
        "Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
        "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
        "###\nInstruction: Given four answer options, A, B, C, and D, choose the best answer.\n\n"
        "Input: Earth rotating causes\n"
        "A: the cycling of AM and PM\nB: the creation of volcanic eruptions\nC: the cycling of the tides\nD: the creation of gravity\n\n"
        "Evidence: Rotation causes the day-night cycle which also creates a corresponding cycle of temperature and humidity creates a corresponding cycle of temperature and humidity. Sea level rises and falls twice a day as the earth rotates.\n\n"
        "Rating: [Relevant]\n"
        "Explanation: The evidence explicitly mentions that the rotation causes a day-night cycle, as described in the answer option A.\n\n"
        "###\nInstruction: Describe a leader or a politician whom you admire. \n\n"
        "Preceding sentences: Leaders and politicians have the power to shape the course of history and impact the lives of countless individuals. Among the myriad of notable figures, Nelson Mandela stands as an exemplary leader whose indomitable spirit, unwavering commitment to justice, and remarkable ability to unite a divided nation have made him an admired and revered personality on a global scale. "
        "Evidence: Barack Obama was one of the most influential people of the world and the man with a difference. He has served as the President of the United States of America. He was the 44th President of America. He was elected in the year 2009 to the office of the President. He was the first-ever African-American President of America.\n\n"
        "Rating: [Irrelevant]\n"
        "Explanation: While the evidence discuss Barack Obama, who is known as an influential political leader, the preceding sentences describe Nelson Mandela, so this evidence doesn't provide useful information to generate an helpful continuation.\n\n"
        "###\nInstruction: {instruction}\n\n"
        "Evidence: {evidence}\n\n"
        "Rating:"
    ),
    "bio_multi": (
        "You'll be provided with an instruction, along with evidence and possibly some preceding sentences. "
        "When there are preceding sentences, your focus should be on the sentence that comes after them. "
        "Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
        "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
        "###\nInstruction: Based on the provided genetic information, classify whether the mutation described predisposes carriers to breast cancer.\nThe provided dataset is a genetic report showing a mutation in the BRCA1 gene.\n"
        "Evidence: TBRCA mutations: is everything said? BACKGROUND: Mutations in the BRCA1 and BRCA2 genes constitute a risk factor for breast cancer development. BRCA mutation research has been an active field since the discovery of the genes, and new mutations in both genes are constantly described and classified according to several systems. AIM: We intend to provide an overview of the current state of BRCA1 and BRCA2 mutation description and classification. We wanted to know whether there was a trend towards a more frequently described mutation type and what the proportion of pathogenic mutations was. RESULTS: We found that, although new mutations are described each year as reflected in current database records, very few of them are reported in papers. Classification systems are highly heterogeneous and a consensus among them\n"
        "Rating: [Relevant]\n"
        "Explanation: The evidence mentions BRCA1/2 mutation probability models and their potential role in predicting breast cancer risk, directly relevant to the instruction about classifying the predisposition to breast cancer based on a mutation.\n\n"
        "###\nInstruction: {instruction}\n"
        "Evidence: {evidence}\n"
        "Rating:"
    ),
    "bio_multi_no_preceding": (
        "You'll be provided with an instruction, along with evidence and possibly some preceding sentences. "
        "When there are preceding sentences, your focus should be on the sentence that comes after them. "
        "Your job is to determine if the evidence is relevant to the initial instruction and the preceding context, and provides useful information to complete the task described in the instruction. "
        "If the evidence meets this requirement, respond with [Relevant]; otherwise, generate [Irrelevant].\n\n"
        "###\nInstruction: Based on the provided genetic information, classify whether the mutation described predisposes carriers to breast cancer.\nThe provided dataset is a genetic report showing a mutation in the BRCA1 gene.\n"
        "Input: The provided dataset is a genetic report showing a mutation in the BRCA1 gene.\n"
        "Evidence: TBRCA mutations: is everything said? BACKGROUND: Mutations in the BRCA1 and BRCA2 genes constitute a risk factor for breast cancer development. BRCA mutation research has been an active field since the discovery of the genes, and new mutations in both genes are constantly described and classified according to several systems. AIM: We intend to provide an overview of the current state of BRCA1 and BRCA2 mutation description and classification. We wanted to know whether there was a trend towards a more frequently described mutation type and what the proportion of pathogenic mutations was. RESULTS: We found that, although new mutations are described each year as reflected in current database records, very few of them are reported in papers. Classification systems are highly heterogeneous and a consensus among them\n"
        "Rating: [Relevant]\n"
        "Explanation: The evidence mentions BRCA1/2 mutation probability models and their potential role in predicting breast cancer risk, directly relevant to the instruction about classifying the predisposition to breast cancer based on a mutation.\n\n"
        "###\nInstruction: {instruction}\n"
        "Evidence: {evidence}\n"
        "Rating:"
    ),
}


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def postprocess(results):
    raw_output = results["choices"][0]["message"]["content"]
    print(raw_output)
    if "\nExplanation:" in raw_output:
        explanation = raw_output.split("\nExplanation:")[1]
        if explanation[0] == " ":
            explanation = explanation[1:]
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


def process_input(example, multi_retrieval=False):
    if multi_retrieval is False:
        return PROMPT_DICT["context"].format_map(example)
    else:
        if "sent_idx" not in example or example["sent_idx"] == 0 or len(example["preceding_sentences"]) == 0:
            return PROMPT_DICT["bio_multi_no_preceding"].format_map(example)
        else:
            return PROMPT_DICT["bio_multi"].format_map(example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='+')
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--multi_retrieval', action="store_true")
    parser.add_argument('--model_name', type=str, default="gpt-4")
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--use_bm25', action="store_true")
    args = parser.parse_args()

    examples = []
    for input_file in args.input_files:
        if input_file.endswith(".json") or input_file.endswith(".json_tmp"):
            examples += json.load(open(input_file))
        else:
            examples += load_jsonlines(input_file)
    
    if args.use_bm25:
        for example in examples:
            example['evidence'] = example['bm25_evidence']

    result_list = []
    if args.n is not None and len(examples) > args.n:
        examples = random.sample(examples, k=args.n)

    task_types = Counter([item["dataset_name"]
                         for item in examples if "dataset_name" in item])

    print(Counter(task_types))

    for idx, example in tqdm(enumerate(examples)):
        if "output" not in example and "answers" in example:
            example["output"] = example["answers"][0] if type(
                example["answers"]) is list else example["answers"]
        if "target_output" not in example and "output" in example:
            example["target_output"] = example["output"]
        if "instruction" not in example and "question" in example:
            data_type = example["q_id"].split("_")[0]
            if data_type in KNOWLEDGE_INSTRUCTIONS:
                example["instruction"] = KNOWLEDGE_INSTRUCTIONS[data_type] + \
                    example["question"]
            else:
                example["instruction"] = example["question"]
        if "As a language model, I cannot" in example["output"]:
            continue
        input = process_input(example, multi_retrieval=args.multi_retrieval)
        try:
            results = completions_with_backoff(
                model=args.model_name,
                messages=[
                    {"role": "user",
                        "content": input},
                ],
                request_timeout=60,
                max_tokens=200,
            )
            score, explanation = postprocess(results)
            result_list.append({"input": example, "score": score, "explanation": score,
                               "raw_output": results["choices"][0]["message"]["content"]})
            if idx % 20 == 0:
                print("Input: {}".format(example["instruction"]))
                print("Output: {}".format(example["output"]))
                print("Evidence: {}".format(example["evidence"]))
                print("Score: {0} ({1})".format(score, explanation))

        except (APIError, Timeout, APIConnectionError):
            results = "ERROR: API error outputs"
        if idx % 20 == 0:
            print("saved output at {}".format(args.output_file_name + "_tmp"))
            with open(args.output_file_name + "_tmp", "w") as outfile:
                json.dump(result_list, outfile)

    with open(args.output_file_name, "w") as outfile:
        json.dump(result_list, outfile)


if __name__ == "__main__":
    main()
