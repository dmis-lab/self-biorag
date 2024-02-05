import os
import json
import torch
import faiss
import argparse
import scispacy
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import query_encode as qe
import retrieve as rt
import rerank as rr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--embeddings_dir', help='embeddings directory', default='embeddings')
    parser.add_argument('-a', '--articles_dir', help='articles directory', default='articles')
    parser.add_argument('-i', '--input_path', help='input file path', default='input/all_biomedical_instruction.json')
    parser.add_argument('-o', '--output_path', help='output file path', default='output/medcpt_top10_evidence.json')
    parser.add_argument('-spc', '--use_spacy', help='use scispacy to insert [SEP] token between sentences', default='True')
    parser.add_argument('-pmdn', '--pubmed_group_num', help='number of chunks of pubmed to concatenate for each step', default='10')

    args = parser.parse_args()

    embeddings_dir = args.embeddings_dir
    articles_dir = args.articles_dir
    input_path = args.input_path
    output_path = args.output_path
    use_spacy = args.use_spacy
    pubmed_group_num = args.pubmed_group_num

    # query preprocess
    input_list = qe.query_preprocess(input_path, use_spacy = use_spacy)
    
    # query encode
    xq = qe.query_encode(input_list)

    # query save
    # np.save('output/query_embeddings.npy', xq)


    # pubmed mips
    pubmed_I_array = []
    for start_index in range(0, 38, pubmed_group_num):
        pubmed_index = rt.pubmed_index_create(pubmed_embeddings_dir=os.path.join(embeddings_dir, "pubmed"), start_index=start_index, pubmed_group_num=pubmed_group_num)
        pubmed_I_array_temp = []
        splits = [i for i in range(0, len(xq), 1024)]

        for split_start in tqdm(splits, desc=f"PubMed FAISS MIPS {start_index}:"):
            D, I = pubmed_index.search(xq[split_start:split_start+1024], 10)   
            pubmed_I_array_temp.extend(I)
        pubmed_I_array.append(pubmed_I_array_temp)
        del pubmed_index
    print(len(pubmed_I_array), "x", len(pubmed_I_array[0]))
    # pubmed mips index save
    # np.save("PubMed_I_array.npy", pubmed_I_array)

    # pubmed decode
    pubmed_evidences = rt.pubmed_decode(pubmed_I_array, pubmed_articles_dir= os.path.join(articles_dir, "pubmed"), pubmed_group_num=pubmed_group_num)
    print(len(pubmed_evidences), "x", len(pubmed_evidences[0]))


    # pmc mips
    pmc_index = rt.pmc_index_create(pmc_embeddings_dir = os.path.join(embeddings_dir, "pmc"))
    pmc_I_array = []

    for i in tqdm(splits, desc="PMC FAISS MIPS"):
        D, I = pmc_index.search(xq[i:i+1024], 10)   
        pmc_I_array.extend(I)
    del pmc_index

    # pmc mips index save
    # np.save("PMC_I_array.npy", pmc_I_array)

    # decode pmc
    pmc_evidences = rt.pmc_decode(pmc_I_array, pmc_articles_dir = os.path.join(articles_dir, "pmc"))


    # cpg mips
    cpg_index = rt.cpg_index_create(cpg_embeddings_dir = os.path.join(embeddings_dir, "cpg"))
    cpg_I_array = []

    for i in tqdm(splits, desc="CPG FAISS MIPS"):
        D, I = cpg_index.search(xq[i:i+1024], 10)   
        cpg_I_array.extend(I)
    del cpg_index

    # cpg mips index save
    # np.save("CPG_I_array.npy", cpg_I_array)

    # decode cpg
    cpg_evidences = rt.cpg_decode(cpg_I_array, cpg_articles_dir = os.path.join(articles_dir, "cpg"))


    # textbook mips
    textbook_index = rt.textbook_index_create(textbook_embeddings_dir = os.path.join(embeddings_dir, "textbook"))
    textbook_I_array = []

    for i in tqdm(splits, desc="textbook FAISS MIPS"):
        D, I = textbook_index.search(xq[i:i+1024], 10)   
        textbook_I_array.extend(I)
    del textbook_index

    # textbook mips index save
    #np.save("Textbook_I_array.npy", textbook_I_array)

    # decode textbook
    textbook_evidences = rt.textbook_decode(textbook_I_array, textbook_articles_dir = os.path.join(articles_dir, "textbook"))


    # rerank evidences from 4 corpora
    query_evidences, evidences = rr.combine_query_evidence(input_list, pubmed_evidences, pmc_evidences, cpg_evidences, textbook_evidences)

    # save output of 10 reranked evidences
    reranked_10evidences = rr.rerank(query_evidences, evidences)
    with open (input_path, 'r') as jsfile:
        input_file = json.load(jsfile)
    
    for ith, rr10ev in enumerate(reranked_10evidences):
        input_file[ith]['evidence'] = rr10ev[:10]

    with open (output_path, 'w') as jsfile:
        json.dump(input_file, jsfile)

if __name__ == "__main__":
    main()