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

from passage_retrieval import Retriever
from transformers import AutoTokenizer

openai.api_key_path = "your_path_to_use_chatgpt_api"

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


PROMPT_DICT = {
    "context": (
        "Given an instruction, please make a judgment on whether finding some external documents from the web (e.g., Wikipedia) helps to generate a better response. Please answer [Yes] or [No] and write an explanation.\n\n"
        "##\nInstruction: Give three tips for staying healthy.\n"
        "Need retrieval?: [Yes]\n"
        "Explanation: There might be some online sources listing three tips for staying healthy or some reliable sources to explain the effects of different behaviors on health. So retrieving documents is helpful to improve the response to this query.\n\n"
        "##\nInstruction: Describe a time when you had to make a difficult decision.\n"
        "Need retrieval?: [No]\n"
        "Explanation: This instruction is asking about some personal experience and thus it does not require one to find some external documents.\n\n"
        "##\nInstruction: Write a short story in third person narration about a protagonist who has to make an important career decision.\n"
        "Need retrieval?: [No]\n"
        "Explanation: This instruction asks us to write a short story, which does not require external evidence to verify.\n\n"
        "##\nInstruction: What is the capital of France?\n"
        "Need retrieval?: [Yes]\n"
        "Explanation: While the instruction simply asks us to answer the capital of France, which is a widely known fact, retrieving web documents for this question can still help.\n\n"
        "##\n Instruction: Find the area of a circle given its radius. Radius = 4\n"
        "Need retrieval?: [No]\n"
        "Explanation: This is a math question and although we may be able to find some documents describing a formula, it is unlikely to find a document exactly mentioning the answer.\n\n"
        "##\nInstruction: Arrange the words in the given sentence to form a grammatically correct sentence. quickly the brown fox jumped\n"
        "Need retrieval?: [No]\n"
        "Explanation: This task doesn't require any external evidence, as it is a simple grammatical question.\n\n"
        "##\nInstruction: Explain the process of cellular respiration in plants."
        "Need retrieval?: [Yes]\n"
        "Explanation: This instruction asks for a detailed description of a scientific concept, and is highly likely that we can find a reliable and useful document to support the response.\n\n"
        "##\nInstruction:{instruction}\n"
        "Need retrieval?: "
    ),
    "multi_retrieval": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn’t require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"
        "##\nInstruction: Explain the use of word embeddings in Natural Language Processing.\n"
        "Preceding sentences: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.\n"
        "Evidence: Word embedding\nWord embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension.\n"
        "Output: Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.\n"
        "Rating: [Retrieval]\n"
        "Explanation: The output discusses the applications of word embeddings, while the evidence only discusses the definitions of word embeddings and how it works. Therefore, we need to retrieve other evidence to verify whether the output is actually correct or not.\n"
        "###\nInstruction: {instruction}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}\n"
        "Rating: "),
    "multi_retrieval_no_preceding": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn’t require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"
        "##\nInstruction: Explain the use of word embeddings in Natural Language Processing.\n"
        "Preceding sentences: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured.\n"
        "Evidence: Word embedding\nWord embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension.\n"
        "Output: Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.\n"
        "Rating: [Retrieval]\n"
        "Explanation: The output discusses the applications of word embeddings, while the evidence only discusses the definitions of word embeddings and how it works. Therefore, we need to retrieve other evidence to verify whether the output is actually correct or not.\n"
        "###\nInstruction: {instruction}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}\n"
        "Rating: "
    ),
    "multi_retrieval_three_way": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:\n"
        "- If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence]. \n"
        "- If the sentence doesn't require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with  [No Retrieval]. \n"
        "If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments. \n\n"
        "##\nInstruction: Explain the use of word embeddings in Natural Language Processing.\n"
        "Preceding sentences: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured. \n"
        "Evidence:\nWord embedding\nWord embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension. \n"
        "Output: Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.\n"
        "Rating: [Retrieval]\n"
        "Explanation: The output discusses the applications of word embeddings, while the evidence only discusses the definitions of word embeddings and how it works. Therefore, we need to retrieve other evidence to verify whether the output is correct or not.\n"
        "###\nInstruction: {instruction}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}\n"
        "Rating: "
    ),
    "multi_retrieval_three_way_no_preceding": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:\n"
        "- If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence]. \n"
        "- If the sentence doesn't require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with  [No Retrieval]. \n"
        "- If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments. \n\n"
        "##\nInstruction: Explain the use of word embeddings in Natural Language Processing.\n"
        "Preceding sentences: Word embeddings are one of the most powerful tools available for Natural Language Processing (NLP). They are mathematical representations of words or phrases in a vector space, allowing similarities between words and the context in which they are used to be measured. \n"
        "Evidence:\nWord embedding\nWord embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with one dimension per word to a continuous vector space with a much lower dimension. \n"
        "Output: Word embeddings are useful for tasks such as sentiment analysis, text classification, predicting the next word in a sequence, and understanding synonyms and analogies.\n"
        "Rating: [Retrieval]\n"
        "Explanation: The output discusses the applications of word embeddings, while the evidence only discusses the definitions of word embeddings and how it works. Therefore, we need to retrieve other evidence to verify whether the output is correct or not.\n"
        "###\nInstruction: {instruction}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}\n"
        "Rating: "
    ),
    "bio_multi_retrieval": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. If the output sentence can be verified solely with the evidence or doesn’t require any verification, respond with [No Retrieval]. If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments.\n\n"
        "##\nInstruction: What is the mechanism of 72T4Cl activation under physiological conditions?\n"
        "Evidence: Timing mechanism and effective activation energy concerned with aging and lifespan in the long-lived and thermosensory mutants of Caenorhabditis elegans.. The lifespans of many poikilothermic animals, including the nematode Caenorhabditis elegans, depend significantly on environmental temperature. Using long-living, thermosensory mutants of C. elegans, we tested whether the temperature dependency of the mean lifespan is compatible with the Arrhenius equation, which typically represents one of the chemical reaction rate theories. The temperature dependency of C. elegans was the Arrhenius type or normal, but daf-2(e1370) mutants were quite different from the others. However, taking into account the effect of the thermal denaturation of DAF-2 with the temperature, we showed that our analyzed results are compatible with previous ones. We investigated the timing mechanism of one parameter (the onset of biodemographic aging (t(0))) in the lifespan equation by applying the RNAi feeding method to daf-2 mutants in order to suppress daf-16 activity at different times during the life cycle. In summary, we further deepened the biological role of two elements, t(0) and z (the inverse of the aging rate), in the lifespan equation and mean lifespan formulated by our diffusion model z(2) = 4Dt(0), where z is composed of t(0) and D (the diffusion constant).\n"
        "Output: The mechanism of 72T4Cl activation under physiological conditions is not known.\n"
        "Rating: [No Retrieval]\n"
        "Explanation: The output sentence does not provide any factual information that needs to be verified, it is stating a fact that the mechanism is not known. The evidence does not mention 72T4Cl at all, supporting the claim that the mechanism of its activation is not known.\n\n"
        "##\nInstruction: Pigment providing colour to stool is \n"
        "Evidence: The importance of stool tests in diagnosis and follow-up of gastrointestinal disorders in children. Stool samples should be evaluated macroscopically in terms of color, consistency, quantity, form, odor, and presence of mucus. The presence of a small amount of mucus in stool is normal. However, the presence of copious mucus or bloody mucus is abnormal. The normal color is tawny due to the presence of bilirubin and bile. In infants, the stool may be green, its consistency may be watery or pasty. Stool color varies greatly depending on diet. Clay-colored or putty colored stool is observed in biliary obstructions. If more than 100 mL blood is lost from the upper gastrointestinal system, black, tarry stool is observed. Besides bleeding, black-colored stool may also be observed due to iron or bismuth treatment. Red-colored stool is observed in lower gastrointestinal tract bleeding.\n"
        "Output: Stercobilinogen. The characteristic brown colour of feces is due to the presence of pigment stercobilin (derived from stercobilinogen).\n\nSo, actually it is stercobilin (not stercobilinogen) which is responsible for the brown color of feces. Stercobilin is derived from stercobilinogen and therefore stercobilinogen is the best answer amongst the options provided.\n"
        "Rating: [Retrieval]\n"
        "Explanation: The provided evidence does not mention stercobilin or stercobilinogen providing the color to stool, although it does mention bilirubin and bile contribute to the color. Therefore, additional evidence would be necessary to verify this output sentence.\n\n"
        "###\nInstruction: {instruction}\n"
        "Preceding sentences: {preceding_sentences}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}\n"
        "Rating: "),
    "bio_multi_retrieval_three_way_no_preceding": (
        "You will be provided with an instruction, evidence, output sentence, and preceding sentences (optional). If the preceding sentence is given, the output should be the sentence that follows those preceding sentences.  Your task is to determine whether the information in the output sentence can be fully verified by the evidence or if it requires further external verification. There are three cases:\n"
        "- If the output sentence can be verified solely with the evidence, then respond with [Continue to Use Evidence]. \n"
        "- If the sentence doesn't require any factual verification (e.g., a subjective sentence or a sentence about common sense), then respond with  [No Retrieval]. \n"
        "If additional information is needed to verify the output sentence, respond with [Retrieval]. Please provide explanations for your judgments. \n\n"
        "##\nInstruction: What is the mechanism of 72T4Cl activation under physiological conditions?\n"
        "Evidence: Timing mechanism and effective activation energy concerned with aging and lifespan in the long-lived and thermosensory mutants of Caenorhabditis elegans.. The lifespans of many poikilothermic animals, including the nematode Caenorhabditis elegans, depend significantly on environmental temperature. Using long-living, thermosensory mutants of C. elegans, we tested whether the temperature dependency of the mean lifespan is compatible with the Arrhenius equation, which typically represents one of the chemical reaction rate theories. The temperature dependency of C. elegans was the Arrhenius type or normal, but daf-2(e1370) mutants were quite different from the others. However, taking into account the effect of the thermal denaturation of DAF-2 with the temperature, we showed that our analyzed results are compatible with previous ones. We investigated the timing mechanism of one parameter (the onset of biodemographic aging (t(0))) in the lifespan equation by applying the RNAi feeding method to daf-2 mutants in order to suppress daf-16 activity at different times during the life cycle. In summary, we further deepened the biological role of two elements, t(0) and z (the inverse of the aging rate), in the lifespan equation and mean lifespan formulated by our diffusion model z(2) = 4Dt(0), where z is composed of t(0) and D (the diffusion constant).\n"
        "Output: The mechanism of 72T4Cl activation under physiological conditions is not known.\n"
        "Rating: [No Retrieval]\n"
        "Explanation: The output sentence does not provide any factual information that needs to be verified, it is stating a fact that the mechanism is not known. The evidence does not mention 72T4Cl at all, supporting the claim that the mechanism of its activation is not known.\n\n"
        "##\nInstruction: Pigment providing colour to stool is \n"
        "Evidence: The importance of stool tests in diagnosis and follow-up of gastrointestinal disorders in children. Stool samples should be evaluated macroscopically in terms of color, consistency, quantity, form, odor, and presence of mucus. The presence of a small amount of mucus in stool is normal. However, the presence of copious mucus or bloody mucus is abnormal. The normal color is tawny due to the presence of bilirubin and bile. In infants, the stool may be green, its consistency may be watery or pasty. Stool color varies greatly depending on diet. Clay-colored or putty colored stool is observed in biliary obstructions. If more than 100 mL blood is lost from the upper gastrointestinal system, black, tarry stool is observed. Besides bleeding, black-colored stool may also be observed due to iron or bismuth treatment. Red-colored stool is observed in lower gastrointestinal tract bleeding.\n"
        "Output: Stercobilinogen. The characteristic brown colour of feces is due to the presence of pigment stercobilin (derived from stercobilinogen).\nSo, actually it is stercobilin (not stercobilinogen) which is responsible for the brown color of feces. Stercobilin is derived from stercobilinogen and therefore stercobilinogen is the best answer amongst the options provided.\n"
        "Rating: [Retrieval]\n"
        "Explanation: The provided evidence does not mention stercobilin or stercobilinogen providing the color to stool, although it does mention bilirubin and bile contribute to the color. Therefore, additional evidence would be necessary to verify this output sentence.\n\n"
        "##\nInstruction: {instruction}\n"
        "Evidence: {evidence}\n"
        "Output: {target_output}\n"
        "Rating: "
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
        if explanation[0] == " ":
            explanation = explanation[1:]
        decision_token = raw_output.split("\nExplanation:")[0]
        if decision_token is None:
            return "", explanation
        else:
            return decision_token, explanation
    else:
        return "", ""


def process_input(example, multi_retrieval=False, three_way=False):
    if multi_retrieval is False:
        return PROMPT_DICT["context"].format_map(example)
        # return PROMPT_DICT["context"].format_map(example['input'])
    else:
        if example["sent_idx"] == 0 or len(example["preceding_sentences"]) == 0:
            if three_way is False:
                return PROMPT_DICT["multi_retrieval_no_preceding"].format_map(example)
            else:
                return PROMPT_DICT["multi_retrieval_three_way"].format_map(example)
        else:
            if three_way is False:
                return PROMPT_DICT["multi_retrieval"].format_map(example)
            else:
                return PROMPT_DICT["multi_retrieval_three_way_no_preceding"].format_map(example)
            

def process_bioinput(example, multi_retrieval=False, three_way=False):
    if multi_retrieval is False:
        return PROMPT_DICT["context"].format_map(example)
        # return PROMPT_DICT["context"].format_map(example['input'])
    else:
        if example["sent_idx"] == 0 or len(example["preceding_sentences"]) == 0:
            return PROMPT_DICT["bio_multi_retrieval_three_way_no_preceding"].format_map(example)
        else:
            return PROMPT_DICT["bio_multi_retrieval"].format_map(example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='+')
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--multi_retrieval', action="store_true")
    parser.add_argument('--three_way', action="store_true")
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--model_name', type=str, default="gpt-4")
    parser.add_argument('--random', action="store_true")
    parser.add_argument('--previous_results', type=str, nargs='+')
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
            

    for item in examples:
        if "id" not in item and "q_id" in item:
            item["id"] = item["q_id"]
    
    result_list = []
    if args.n is not None:
        if args.random is True:
            examples = random.sample(examples, args.n)
        else:
            examples = examples[:args.n]

    for idx, example in tqdm(enumerate(examples)):
        # input = process_input(
        #     example, multi_retrieval=args.multi_retrieval, three_way=args.three_way)
        input = process_bioinput(
            example, multi_retrieval=args.multi_retrieval, three_way=args.three_way)
        
        if idx % 5 == 0:
            print(input)
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
            decision_token, explanation = postprocess(results)
            result_list.append({"input": example, "decision_token": decision_token, "explanation": explanation,
                               "raw_output": results["choices"][0]["message"]["content"]})
            
            if idx % 5 == 0:
                print("Input: {}".format(example["instruction"]))
                print("explanation: {}".format(explanation))
                print("decision_token: {}".format(decision_token))

        except (APIError, Timeout, APIConnectionError):
            results = "ERROR: API error outputs"
        if idx % 100 == 0:
            with open(args.output_file_name + "_tmp", "w") as outfile:
                json.dump(result_list, outfile)

    with open(args.output_file_name, "w") as outfile:
        json.dump(result_list, outfile)


if __name__ == "__main__":
    main()
