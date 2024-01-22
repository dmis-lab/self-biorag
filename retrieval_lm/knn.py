import os
# os.environ['TRANSFORMERS_CACHE'] = '/scratch/x2696a10/'
import json
import tqdm
import torch
import random
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from metrics import match, accuracy

import torch.nn.functional as F
from peft import PeftModel, PeftConfig

PROMPT_DICT = {
    "med_qa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "medmc_qa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "pubmed_qa": "Given the provided context, analyze the relationship or connection between two concepts or factors mentioned in the text. Based on this analysis, generate a response that falls into one of the three classes: 'yes' if there is a connection, 'no' if there is no connection, or 'maybe' if the connection is uncertain or inconclusive.",
    "mmlu": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "live_qa": "Answer the following question. The question may be hard to answer but you have to provide a long-form answer including all correct answers.",
    "medication_qa": "Answer the following question. The question may be hard to answer but you have to provide a long-form answer including all correct answers."
}

FEW_SHOT = {
    "med_qa": (
        # "You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines.\n\n"
            "Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n"
               "###QUESTION: A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?\n"
               "Option A: anterior scalene\nOption B: latissimus dorsi\nOption C: pectoralis minor\nOption D: quadratus lumborum\n"
               "Explanation: We refer to Wikipedia articles on medicine for help. Among the options, only pectoralis minor muscle origins from the outer surfaces of the 3rd to 5th ribs.\n"
               "Answer: (C)\n\n"
               "###QUESTION: A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is\n"
               "Option A: right-on-right sacral torsion\nOption B: left-on-right sacral torsion\nOption C: right unilateral sacral flexion\nOption D: left-on-left sacral torsion\n"
               "Explanation: We refer to Wikipedia articles on medicine for help. The deep sulcus on the left, a posterior ILA on the right, with a negative spring test suggests a right-on-right sacral torsion. All other options have a deep sulcus on the right.\n"
               "Answer: (A)\n\n"),
            #    "###QUESTION: A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient’s symptoms?\n"
            #    "Option A: Allergic rhinitis\nOption B: Epstein-Barr virus\nOption C: Mycoplasma pneumonia\nOption D: Rhinovirus\n"
            #    "Explanation: We refer to Wikipedia articles on medicine for help. The symptoms, especially the headache, suggest that the most likely cause is Rhinovirus. Epstein-Barr virus will cause swollen lymph nodes but there is no palpable cervical adenopathy. Lungs are clear to auscultation suggests it’s not Mycoplasma pneumonia.\n"
            #    "Answer: (B)\n\n"),
            #    "###QUESTION: A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?\n"
            #    "Option A: Dopamine\nOption B: Glutamate\nOption C: Norepinephrine\nOption D: Serotonin\n"
            #    "Explanation: We refer to Wikipedia articles on medicine for help. The patient feels sad and among the options, only Dopamine and Serotonin can help increase positive emotions. Serotonin also affects digestion and metabolism, which can help the patient’s decreased appetite and sleep difficulty.\n"
            #    "Answer: (D)\n\n"),
    "medmc_qa": (
                "You are a medical doctor answering real-world medical entrance exam questions. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following multiple-choice question. Select one correct answer from A to D. Base your answer on the current and standard practices referenced in medical guidelines.\n\n"
                # "Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n"
                "###QUESTION: Maximum increase in prolactin level is caused by:\n"
                "Option A: Risperidone\nOption B: Clozapine\nOption C: Olanzapine\nOption D: Aripiprazole\n"
                "Explanation: Clozapine generally does not raise prolactin levels. Atypicals such as olanzapine and aripiprazole cause small if no elevation. Risperidone is known to result in a sustained elevated prolactin level. Therefore risperidone is likely to cause the maximum increase in prolactin level.\n"
                "Answer: (A)\n\n"
                "###QUESTION: What is the age of routine screening mammography?"
                "Option A: 20 years\nOption B: 30 years\nOption C: 40 years\nOption D: 50 years\n"
                "Explanation: The age of routine screening depends on the country you are interested in and varies widely. For the US, it is 40 years of age according to the American Cancer Society. In Europe, it is typically closer to 50 years. For a patient based in the US, the best answer is 40 years.\n"
                "Answer: (C)\n\n"
                "###QUESTION: A 65-year-old male complains of severe back pain and inability to move his left lower limb. Radiographic studies demonstrate the compression of nerve elements at the intervertebral foramen between vertebrae L5 and S1. Which structure is most likely responsible for this space-occupying lesion?\n"
                "Option A: Anulus fibrosus\nOption B: Nucleus pulposus\nOption C: Posterior longitudinal ligament\nOption D: Anterior longitudinal ligament\n"
                "Explanation: This man describes a herniated invertebral disk through a tear in the surrounding annulus fibrosus. The soft, gelatinous \"nucleus pulposus\" is forced out through a weakened part of the disk, resulting in back pain and nerve root irritation. In this case, the impingement is resulting in paralysis, and should be considered a medical emergency. Overall, the structure that is causing the compression and symptoms is the nucleus pulposus.\n"
                "Answer: (B)\n\n"),
                # "###QUESTION: Neuroendocrine cells in the lungs are:\n"
                # "Option A: Dendritic cells\nOption B: Type I pneumocytes\nOption C: Type II pneumocytes\nOption D: APUD cells\n"
                # "Explanation: Neuroendocrine cells, which are also known as Kultschitsky-type cells, Feyrter cells and APUD cells, are found in the basal layer of the surface epithelium and in the bronchial glands.\n"
                # "Answer: (D)\n\n"
                # "###QUESTION: Presence of it indicates remote contamination of water\n"
                # "Option A: Streptococci\nOption B: Staphalococci\nOption C: Clastridium pertringes\nOption D: Nibrio\n"
                # "Explanation: Because Clostridium perfringens spores are both specific to sewage contamination and environmentally stable, they are considered as possible conservative indicators of human fecal contamination and possible surrogates for  nvironmentally stable pathogens.\n"
                # "Answer: (C)\n\n"),
    "pubmed_qa": (
        # "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe.\n\n"
                "Given the provided context, analyze the relationship or connection between two concepts or factors mentioned in the text. Based on this analysis, generate a response that falls into one of the three classes: 'yes' if there is a connection, 'no' if there is no connection, or 'maybe' if the connection is uncertain or inconclusive.\n\n"
                "###CONTEXT: To describe the interstitial fluid (ISF) and plasma pharmacokinetics of meropenem in patients on continuous venovenous haemodiafiltration (CVVHDF). This was a prospective observational pharmacokinetic study. Meropenem (500 mg) was administered every 8 h. CVVHDF was targeted as a 2-3 L/h exchange using a polyacrylonitrile filter with a surface area of 1.05 m2 and a blood flow rate of 200 mL/min. Serial blood (pre- and post-filter), filtrate/dialysate and ISF concentrations were measured on 2 days of treatment (Profiles A and B). Subcutaneous tissue ISF concentrations were determined using microdialysis. A total of 384 samples were collected. During Profile A, the comparative median (IQR) ISF and plasma peak concentrations were 13.6 (12.0-16.8) and 40.7 (36.6-45.6) mg/L and the trough concentrations were 2.6 (2.4-3.4) and 4.9 (3.5-5.0) mg/L, respectively. During Profile B, the ISF trough concentrations increased by ∼40%. Meropenem ISF penetration was estimated at 63% (60\%-69\%) and 69% (65\%-74\%) for Profiles A and B, respectively, using comparative plasma and ISF AUCs. For Profile A, the plasma elimination t1/2 was 3.7 (3.3-4.0) h, the volume of distribution was 0.35 (0.25-0.46) L/kg, the total clearance was 4.1 (4.1-4.8) L/h and the CVVHDF clearance was 2.9 (2.7-3.1) L/h.\n"
                "QUESTION: Are interstitial fluid concentrations of meropenem equivalent to plasma concentrations in critically ill patients receiving continuous renal replacement therapy?\n"
                "Option A: Yes\nOption B: No\nOption C: Maybe\n"
                "Explanation: This is the first known report of concurrent plasma and ISF concentrations of a meropenem antibiotic during CVVHDF. We observed that the ISF concentrations of meropenem were significantly lower than the plasma concentrations,although the present dose was appropriate for infections caused by intermediately susceptible pathogens (MIC≤4 mg/L).\n"
                "Answer: (B)\n\n"
                "###CONTEXT: Family caregivers of dementia patients are at increased risk of developing depression or anxiety. A multi-component program designed to mobilize support of family networks demonstrated effectiveness in decreasing depressive symptoms in caregivers. However, the impact of an intervention consisting solely of family meetings on depression and anxiety has not yet been evaluated. This study examines the preventive effects of family meetings for primary caregivers of community-dwelling dementia patients. A randomized multicenter trial was conducted among 192 primary caregivers of community dwelling dementia patients. Caregivers did not meet the diagnostic criteria for depressive or anxiety disorder at baseline. Participants were randomized to the family meetings intervention (n=96) or usual care (n=96) condition. The intervention consisted of two individual sessions and four family meetings which occurred once every 2 to 3 months for a year. Outcome measures after 12 months were the incidence of a clinical depressive or anxiety disorder and change in depressive and anxiety symptoms (primary outcomes), caregiver burden and quality of life (secondary outcomes). Intention-to-treat as well as per protocol analyses were performed. A substantial number of caregivers (72/192) developed a depressive or anxiety disorder within 12 months. The intervention was not superior to usual care either in reducing the risk of disorder onset (adjusted IRR 0.98; 95% CI 0.69 to 1.38) or in reducing depressive (randomization-by-time interaction coefficient=-1.40; 95% CI -3.91 to 1.10) or anxiety symptoms (randomization-by-time interaction coefficient=-0.55; 95% CI -1.59 to 0.49). The intervention did not reduce caregiver burden or their health related quality of life.\n"
                "QUESTION: Does a family meetings intervention prevent depression and anxiety in family caregivers of dementia patients?\n"
                "Option A: Yes\nOption B: No\nOption C: Maybe\n"
                "Explanation: This study did not demonstrate preventive effects of family meetings on the mental health of family caregivers. Further research should determine whether this intervention might be more beneficial if provided in a more concentrated dose, when applied for therapeutic purposes or targeted towards subgroups of caregivers.\n"
                "Answer: (B)\n\n"
                "###CONTEXT: To compare adherence to follow-up recommendations for colposcopy or repeated Papanicolaou (Pap) smears for women with previously abnormal Pap smear results. Retrospective cohort study. Three northern California family planning clinics. All women with abnormal Pap smear results referred for initial colposcopy and a random sample of those referred for repeated Pap smear. Medical records were located and reviewed for 90 of 107 women referred for colposcopy and 153 of 225 women referred for repeated Pap smears. Routine clinic protocols for follow-up–telephone call, letter, or certified letter–were applied without regard to the type of abnormality seen on a Pap smear or recommended examination. Documented adherence to follow-up within 8 months of an abnormal result. Attempts to contact the patients for follow-up, adherence to follow-up recommendations, and patient characteristics were abstracted from medical records. The probability of adherence to follow-up vs the number of follow-up attempts was modeled with survival analysis. Cox proportional hazards models were used to examine multivariate relationships related to adherence. The rate of overall adherence to follow-up recommendations was 56.0% (136/243). Adherence to a second colposcopy was not significantly different from that to a repeated Pap smear (odds ratio, 1.40; 95\% confidence interval, 0.80-2.46). The use of as many as 3 patient reminders substantially improved adherence to follow-up. Women without insurance and women attending 1 of the 3 clinics were less likely to adhere to any follow-up recommendation (hazard ratio for no insurance, 0.43 [95\% confidence interval, 0.20-0.93], and for clinic, 0.35 [95\% confidence interval, 0.15-0.73]).\n"
                "QUESTION: Do follow-up recommendations for abnormal Papanicolaou smears influence patient adherence?\n"
                "Option A: Yes\nOption B: No\nOption C: Maybe\n"
                "Explanation: Adherence to follow-up was low in this family planning clinic population, no matter what type of follow-up was advised. Adherence was improved by the use of up to 3 reminders. Allocating resources to effective methods for improving adherence to follow-up of abnormal results may be more important than which follow-up procedure is recommended.\n"
                "Answer: (B)\n\n"),
    "mmlu": ("Given four answer candidates, A, B, C and D, choose the best answer choice. Let's think step by step.\n\n"
             "###QUESTION: The energy for all forms of muscle contraction is provided by:\n"
             "Option A: ATP\nOption B: ADP\nOption C: phosphocreatine\nOption D: oxidative phosphorylation\n"
             "Explanation: The sole fuel for muscle contraction is adenosine triphosphate (ATP). During near maximal intense exercise the muscle store of ATP will be depleted in less than one second. Therefore, to maintain normal contractile function ATP must be continually resynthesized. These pathways include phosphocreatine and muscle glycogen breakdown, thus enabling substrate-level phosphorylation (‘anaerobic’) and oxidative phosphorylation by using reducing equivalents from carbohydrate and fat metabolism (‘aerobic’).\n"
             "Answer: (A)\n\n"
             "###QUESTION: Which of the following conditions does not show multifactorial inheritance?\n"
             "Option A: Pyloric stenosis\nOption B: Schizophrenia\nOption C: Spina bifida (neural tube defects)\nOption D: Marfan syndrome\n"
             "Explanation: Multifactorial inheritance refers to when a condition is caused by multiple factors, which may be both genetic or environmental. Marfan is an autosomal dominant trait. It is caused by mutations in the FBN1 gene, which encodes a protein called fibrillin-1. Hence, Marfan syndrome is not an example of multifactorial inheritance.\n"
             "Answer: (D)\n\n"
             "###QUESTION: What is the embryological origin of the hyoid bone?\n"
             "Option A: The first pharyngeal arch\nOption B: The first and second pharyngeal arches\nOption C: The second pharyngeal arch\nOption D: The second and third pharyngeal arches\n"
             "Explanation: In embryology, the pharyngeal arches give rise to anatomical structure in the head and neck. The hyoid bone, a small bone in the midline of the neck anteriorly, is derived from the second and third pharyngeal arches.\n"
             "Answer: (D)\n\n"
             "###QUESTION: A high school science teacher fills a 1 liter bottle with pure nitrogen and seals the lid. The pressure is 1.70 atm, and the room temperature is 25◦C. Which two variables will both increase the pressure of the system, if all other variables are held constant?\n"
             "Option A: Decreasing volume, decreasing temperature\nOption B: Increasing temperature, increasing volume\nOption C: Increasing temperature, increasing moles of gas\nOption D: Decreasing moles of gas, increasing volume\n"
             "Explanation: According to the ideal gas law, PV = nRT (P = pressure, V = volume, n = number of moles, R = gas constant, T = temperature). Hence, increasing both temperature (T) and moles of gas (n), while other variables stay constant, will indeed increase the pressure of the system.\n"
             "Answer: (C)\n\n"
             "###QUESTION: A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?\n"
             "Option A: anterior scalene\nOption B: latissimus dorsi\nOption C: pectoralis minor\nOption D: quadratus lumborum\n"
             "Explanation: All of the muscles have an insertion on the rib cage; however only one has an insertion at ribs 4-5 and could be responsible for right-sided rib pain: pectoralis minor. Pectoralis minor inserts to the costal cartilage of the anterior third to fifth ribs.\n"
             "Answer: (C)\n\n"),
    "live_qa": ("Answer the following question. The question may be hard to answer but you have to provide a long-form answer including all correct answers."),
    "medication_qa": ("Answer the following question. The question may be hard to answer but you have to provide a long-form answer including all correct answers.")
}

def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer

def format_prompt(input, paragraph=None):
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
    if paragraph is not None:
        prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    return prompt



# from openai import OpenAI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="ncbi/MedCPT-Article-Encoder")
    
    args = parser.parse_args()
    
    ##OPENAI
    # client = OpenAI()
    # def get_embedding(text, model="text-embedding-ada-002"):
    #     text = text.replace("\n", "")
    #     return client.embeddings.create(input=[text], model=model).data[0].embedding

    # medcpt
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # "med_qa", "medmc_qa", "pubmed_qa", "mmlu", "live_qa", "medication_qa", "mmlu_clinical_knowledge", "mmlu_anatomy", "mmlu_college_biology", "mmlu_college_medicine", "mmlu_medical_genetics", "mmlu_professional_medicine"
    evaluation_list = ["med_qa"]
    for eval_name in tqdm.tqdm(evaluation_list, desc="total evaluation"):
        test_examples = []
        with open(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_test.jsonl", "r") as fp:
            lines = fp.readlines()
            for line in lines:
                test_examples.append(json.loads(line))

        # train_examples = []
        # with open(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4.jsonl", "r") as fp:
        #     lines = fp.readlines()
        #     for line in lines:
        #         train_examples.append(json.loads(line))
        train_examples = json.load(open(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4.json"))

        # if os.path.isfile(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_embed.npy") and os.path.isfile(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_test_embed.npy"):

            # tr_embed_load = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_embed.npy")
        # tr_embed_load_1 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_500_embed.npy")
        # tr_embed_load_2 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_1000_embed.npy")
        # tr_embed_load_3 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_1500_embed.npy")
        # tr_embed_load_4 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_2000_embed.npy")
        # tr_embed_load_5 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_2500_embed.npy")
        # tr_embed_load_6 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_3000_embed.npy")
        # tr_embed_load_7 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_3500_embed.npy")
        # tr_embed_load_8 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_4000_embed.npy")
        # tr_embed_load_9 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_4500_embed.npy")
        # tr_embed_load_10 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_5000_embed.npy")
        # tr_embed_load_11 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_5500_embed.npy")
        # tr_embed_load_12 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_6000_embed.npy")
        # tr_embed_load_13 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_6500_embed.npy")
        # tr_embed_load_14 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_7000_embed.npy")
        # tr_embed_load_15 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_7500_embed.npy")
        # tr_embed_load_16 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_8000_embed.npy")
        # tr_embed_load_17 = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_8500_embed.npy")
        # # tr_result = torch.from_numpy(tr_embed_load).to('cuda')
        # tr_result_1 = torch.from_numpy(tr_embed_load_1).to('cuda')
        # tr_result_2 = torch.from_numpy(tr_embed_load_2).to('cuda')
        # tr_result_3 = torch.from_numpy(tr_embed_load_3).to('cuda')
        # tr_result_4 = torch.from_numpy(tr_embed_load_4).to('cuda')
        # tr_result_5 = torch.from_numpy(tr_embed_load_5).to('cuda')
        # tr_result_6 = torch.from_numpy(tr_embed_load_6).to('cuda')
        # tr_result_7 = torch.from_numpy(tr_embed_load_7).to('cuda')
        # tr_result_8 = torch.from_numpy(tr_embed_load_8).to('cuda')
        # tr_result_9 = torch.from_numpy(tr_embed_load_9).to('cuda')
        # tr_result_10 = torch.from_numpy(tr_embed_load_10).to('cuda')
        # tr_result_11 = torch.from_numpy(tr_embed_load_11).to('cuda')
        # tr_result_12 = torch.from_numpy(tr_embed_load_12).to('cuda')
        # tr_result_13 = torch.from_numpy(tr_embed_load_13).to('cuda')
        # tr_result_14 = torch.from_numpy(tr_embed_load_14).to('cuda')
        # tr_result_15 = torch.from_numpy(tr_embed_load_15).to('cuda')
        # tr_result_16 = torch.from_numpy(tr_embed_load_16).to('cuda')
        # tr_result_17 = torch.from_numpy(tr_embed_load_17).to('cuda')
        
        # tr_result = torch.cat([tr_result_1, tr_result_2, tr_result_3, tr_result_4, tr_result_5, tr_result_6, tr_result_7, tr_result_8, tr_result_9, tr_result_10, tr_result_11, tr_result_12, tr_result_13, tr_result_14, tr_result_15, tr_result_16, tr_result_17], dim=0)
        # tr_result = tr_result.cpu().numpy()
        # np.save(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_embed.npy", tr_result)
        # te_embed_load = np.load(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_test_embed.npy")
        # te_result = torch.from_numpy(te_embed_load).to('cuda')
        
        # k = 5  # Change this value based on your requirement
        # for i_idx,te_inst in enumerate(te_result):
        #     cosine_similarities = F.cosine_similarity(te_inst.reshape(1,768), tr_result, dim=1)

        #     # Get indices of the top k nearest neighbors
        #     top_k_values, top_k_indices = torch.topk(cosine_similarities, k)
        #     topk_np = top_k_indices.cpu().numpy()
        #     train_insts = [tr_inst for tr_idx,tr_inst in enumerate(train_examples) if tr_idx in topk_np]
            
        
        # else:
        articles = []
        for inst_idx, inst in enumerate(train_examples):
            articles.append([inst['instances']['input'], inst['instances']['output']])
        
        with torch.no_grad():
            encoded = tokenizer(
                articles,
                truncation=True,
                padding=True,
                return_tensors='pt',
                max_length=512,
            )
            embeds = model(**encoded).last_hidden_state[:, 0, :]

        embed_np = embeds.cpu().numpy()
        np.save(f"/nvme1/minbyul/self-rag/data/benchmark/{eval_name}_train_gpt4_embed.npy", embed_np)


if __name__ == "__main__":
    main()