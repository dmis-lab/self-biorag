"""
{
    "instruction": str, # input instruction 
    "target_output": str, # segment-level output
    "evidence": str, # retrieved Wikipedia paragraph
    "preceding_sentences": str, # previously generated sentences
    "output": str # full output (only used for utility),
    "q_id": str # unique instance id,
    "sent_id": int # sentence index
    "p_id": int # paragraph index
}
"""

"""
Checklist
1. Collect utility
1-a. python chatgpt_utility.py --input_file_name /nvme1/minbyul/biomedical_instruction_data/ --model_name gpt-4 --output_file_name /nvme1/minbyul/biomedical_instruction_data/bio_instruction_data.json
before (3) Need retrieved evidence
2. Retrieve with general domain (Wikipedia) or biomedical domain (PubMed+PMC)
2-a. Contriever-msmarco
2-a-1. python create_retrieval_data.py --input_files /nvme1/minbyul/biomedical_instruction_data/ --output_file /nvme1/minbyul/biomedical_instruction_data/bio_instruction_data_create_ret.json
2-a-1. python passage_retrieval.py --model_name_or_path facebook/contriever-msmarco --passages psgs_w100.tsv --passages_embeddings "wikipedia_embeddings/*" --input_files /nvme1/minbyul/biomedical_instruction_data/
3-1. Retrieve evidence with target_output, preceding sentences
3. Collect Retrieval tokens
3-a.  python chatgpt_need_retrieval.py --input_files /nvme1/minbyul/biomedical_instruction_data/ --output_file_name /nvme1/minbyul/biomedical_instruction_data/ --model_name gpt-4 --multi_retrieval --three_way
4. Collect Relevant tokens
4-a.
5. Collect Supportive tokens
"""

