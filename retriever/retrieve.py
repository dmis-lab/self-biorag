import faiss
import json
import numpy as np
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def pubmed_index_create(pubmed_embeddings_dir, start_index, pubmed_group_num):
    pubmed_index = faiss.IndexFlatIP(768)  
    for i in tqdm(range(start_index, min(38, start_index+pubmed_group_num)), desc="pubmed load and add", dynamic_ncols=True):
        embeds_chunk_path = f"{pubmed_embeddings_dir}/PubMed_Embeds_{i}.npy"
        embeds_chunk = np.load(embeds_chunk_path)
        embeds_chunk = embeds_chunk.astype(np.float32)
        pubmed_index.add(embeds_chunk)
        del embeds_chunk
    return pubmed_index

def pmc_index_create(pmc_embeddings_dir):
    pmc_filename=["PMC_Main_Embeds.npy", "PMC_Abs_Embeds.npy"]
    pmc_index = faiss.IndexFlatIP(768)
    for i in tqdm(pmc_filename, desc="pmc load and add", dynamic_ncols=True):
        embeddings = np.load(os.path.join(pmc_embeddings_dir, i))
        embeddings = embeddings.astype(np.float32)
        pmc_index.add(embeddings)
        del embeddings
    return pmc_index

def cpg_index_create(cpg_embeddings_dir):
    cpg_index = faiss.IndexFlatIP(768)
    with tqdm(total=1, desc="cpg load and add", dynamic_ncols=True) as pbar:
        embeddings = np.load(os.path.join(cpg_embeddings_dir,"CPG_Total_Embeds.npy"))
        embeddings = embeddings.astype(np.float32)
        cpg_index.add(embeddings)
        del embeddings
        pbar.update(1)
    return cpg_index

def textbook_index_create(textbook_embeddings_dir):
    textbook_index = faiss.IndexFlatIP(768)
    with tqdm(total=1, desc="textbook load and add", dynamic_ncols=True) as pbar:
        embeddings = np.load(os.path.join(textbook_embeddings_dir, "Textbook_Total_Embeds.npy")) 
        embeddings = embeddings.astype(np.float32)
        textbook_index.add(embeddings)
        del embeddings
        pbar.update(1)
    return textbook_index


def find_value_by_index(articles, target_index):
    return articles[target_index]

def pubmed_decode(pubmed_I_array, pubmed_articles_dir, pubmed_group_num):
    def combine_articles(pubmed_articles_dir, start_index, pubmed_group_num):
        pubmed_articles = []
        for i in tqdm(range(start_index, min(38, start_index+pubmed_group_num)), desc="articles load and add", dynamic_ncols=True):
            with open(pubmed_articles_dir+f"/PubMed_Articles_{i}.json", 'r') as article_chunk:
                pubmed_articles.extend(json.load(article_chunk))
        return pubmed_articles

    #pubmed_I_array_savepath = "PubMed_128_I_array.npy"
    #output_json_path = "PubMed_retrieved.json"
    pubmed_evidences = []
    for start_index in range(0, 38, pubmed_group_num):
        pubmed_articles = combine_articles(pubmed_articles_dir, start_index, pubmed_group_num)

    #pubmed_I_array = np.load(idx_array_savepath)
        pubmed_evidences_temp = []
        for ith, indices in tqdm(enumerate(pubmed_I_array[start_index//10]), desc="decode and add", dynamic_ncols=True):
            evidence_list = [find_value_by_index(pubmed_articles, target_index) for target_index in indices]
            pubmed_evidences_temp.append(evidence_list)
        pubmed_evidences.append(pubmed_evidences_temp)
    
    pubmed_evidences_flat = []
    for subtuple in zip(*pubmed_evidences):
        group = []
        for sublist in subtuple:
            group.extend(sublist)
        pubmed_evidences_flat.append(group)

    #with open(output_json_path, 'w') as jsfile:
    #    json.dump(total_evidence, jsfile)
    #logging.info(f"evidence saved: {len(total_evidence)}")
    
    return pubmed_evidences_flat

def pmc_decode(pmc_I_array, pmc_articles_dir):
    def load_article(pmc_articles_dir):
        pmc_articles = []
        for i in ["PMC_Main_Articles.json", "PMC_Abs_Articles.json"]:
            with open(os.path.join(pmc_articles_dir, i), 'r') as jsfile:
                pmc_articles.extend(json.load(jsfile))
        return pmc_articles

    #idx_array_savepath = "PMC_128_I_array.npy"
    #output_json_path = "PMC_retrieved.json"

    pmc_articles = load_article(pmc_articles_dir)

    #pmc_I_array = np.load(idx_array_savepath)

    pmc_evidences = [] 

    for ith, indices in tqdm(enumerate(pmc_I_array), desc="decode and add", dynamic_ncols=True):
        evidence_list = [find_value_by_index(pmc_articles, j) for j in indices]
        pmc_evidences.append(evidence_list)

    #with open(output_json_path, 'w') as jsfile:
    #    json.dump(total_evidence, jsfile)
    #logging.info(f"evidence saved: {len(total_evidence)}")

    return pmc_evidences


def cpg_decode(cpg_index, cpg_articles_dir):
    def load_articles(cpg_articles_dir):
        with open(os.path.join(cpg_articles_dir,'CPG_Total_Articles.json'), 'r') as jsfile:
            cpg_articles = json.load(jsfile)
        return cpg_articles

    #idx_array_savepath = "CPG_128_I_array.npy"
    #output_json_path = "CPG_retrieved.json"

    cpg_articles = load_articles(cpg_articles_dir)

    #idx_array = np.load(idx_array_savepath)

    cpg_evidences = []

    for ith, indices in tqdm(enumerate(cpg_index), desc="decode and add", dynamic_ncols=True):
        evidence_list = [find_value_by_index(cpg_articles, j) for j in indices]
        cpg_evidences.append(evidence_list)

    #with open(output_json_path, 'w') as jsfile:
    #    json.dump(total_evidence, jsfile)
    #logging.info(f"evidence saved: {len(total_evidence)}")

    return cpg_evidences

def textbook_decode(textbook_index, textbook_articles_dir):
    def load_articles(textbook_articles_dir):
        with open(os.path.join(textbook_articles_dir, "Textbook_Total_Articles.json"), 'r') as jsfile:
            textbook_articles = json.load(jsfile)
        return textbook_articles

    #idx_array_savepath = "Textbook_128_I_array.npy"
    #output_json_path = "Textbook_retrieved.json"

    textbook_articles = load_articles(textbook_articles_dir)

    #logging.info("Loading indices")
    #idx_array = np.load(idx_array_savepath)
    #logging.info(f"Indices loaded: {idx_array.shape}")

    textbook_evidences = []

    for ith, indices in tqdm(enumerate(textbook_index), desc="decode and add", dynamic_ncols=True):
        evidence_list = [find_value_by_index(textbook_articles, j) for j in indices]
        textbook_evidences.append(evidence_list)

    #with open(output_json_path, 'w') as jsfile:
    #    json.dump(total_evidence, jsfile)
    #logging.info(f"evidence saved: {len(total_evidence)}")

    return textbook_evidences
    