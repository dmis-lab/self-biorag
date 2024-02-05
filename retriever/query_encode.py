import torch
import sys
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import scispacy
import spacy
import json

def query_preprocess(input_path, use_spacy = True): #using spacy to divide query into sentences and then inserting [SEP] token is time-intensive, so we leave it as user's design choice.
    with open(input_path, 'r') as jsfile:
        input_data = json.load(jsfile)
    input_instruction = [i['instruction'] for i in input_data]
    input_input = [i['input'] for i in input_data]

    if use_spacy:
        nlp = spacy.load("en_core_sci_scibert")

        split_data_instruction = []
        for instruction in tqdm(input_instruction):
            split_data_instruction.append(nlp(instruction))

        split_data_input = []
        for input in tqdm(input_input):
            split_data_input.append(nlp(input))

        query_list = []

        for inst_idx, inst in enumerate(split_data_instruction):
            query = ""
            for inst_text in split_data_instruction[inst_idx].sents:
                if len(inst_text.text) == 1:
                    continue
                query += inst_text.text + " [SEP] "
            for input_idx, input_text in enumerate(split_data_input[inst_idx].sents):
                if len(input_text.text) == 1:
                    continue        
                elif input_idx == len(list(split_data_input[inst_idx].sents))-1:
                    query += input_text.text
                else:
                    query += input_text.text + " [SEP] "
            query_list.append(query)
    else:
        query_list = []
        for inst_idx, inst in enumerate(input_instruction):
            query = inst + ' ' +input_input[inst_idx]
            query_list.append(query)    
    return query_list


def query_encode(input_list):
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
    if torch.cuda.is_available():
        model = model.to(0)
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

    queries=[]

    splits = [i for i in range(0, len(input_list), 100)]
    for i in tqdm(splits, desc="query encoding"):
        split_queries = input_list[i:i+100]
        with torch.no_grad():
            encoded = tokenizer(
                split_queries, 
                truncation=True, 
                padding=True, 
                return_tensors='pt', 
                max_length=192,
        )
            encoded = {key: tensor.to(0) for key, tensor in encoded.items()}
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            query_embeddings = embeds.detach().cpu().numpy()      
            queries.extend(query_embeddings)
            xq = np.vstack(queries)
    return xq
