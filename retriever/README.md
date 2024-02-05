1. Save embeddings and articles into their corresponding folders:
ex) for pmc embeddings, save PMC_Abs_Embeds.npy, PMC_Main_Embeds.npy in embeddings/pmc

2. Navigate to the directory where main.py is located.

3. Execute the Python script main.py using the following command:
```
python main.py
```

4. For PubMed, we grouped the 38 chunks into 10, 10, 10, and 8 subgroups, respectively.
We retrieved 10 evidences from each subgroup using MIPS, totaling 40 evidences.
For PMC, CPG, and textbook, we retrieved 10 evidences from each.
With a total of 70 evidences, we reranked them and obtained the final 10 evidences.
The number of PubMed subgroup chunks can be adjusted using the --pubmed_group_num argument.

5. We used SciSpacy en_core_sci_scibert to add [SEP] tokens for encoding queries and articles for MIPS but not for the reranker following MEDCPT (Jin et al., 2023).
This functionality is optional and can be enabled using the --use_spacy argument.

