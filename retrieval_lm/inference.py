import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/x2696a10/'
import json
import tqdm
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F

import vllm
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from vllm import LLM, SamplingParams
from utils import PROMPT_DICT, TASK_INST, load_jsonlines, control_tokens, load_special_tokens
from metrics import match, accuracy

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
               "Answer: (A)\n\n"
               "###QUESTION: A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient’s symptoms?\n"
               "Option A: Allergic rhinitis\nOption B: Epstein-Barr virus\nOption C: Mycoplasma pneumonia\nOption D: Rhinovirus\n"
               "Explanation: We refer to Wikipedia articles on medicine for help. The symptoms, especially the headache, suggest that the most likely cause is Rhinovirus. Epstein-Barr virus will cause swollen lymph nodes but there is no palpable cervical adenopathy. Lungs are clear to auscultation suggests it’s not Mycoplasma pneumonia.\n"
               "Answer: (B)\n\n"
               "###QUESTION: A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?\n"
               "Option A: Dopamine\nOption B: Glutamate\nOption C: Norepinephrine\nOption D: Serotonin\n"
               "Explanation: We refer to Wikipedia articles on medicine for help. The patient feels sad and among the options, only Dopamine and Serotonin can help increase positive emotions. Serotonin also affects digestion and metabolism, which can help the patient’s decreased appetite and sleep difficulty.\n"
               "Answer: (D)\n\n"),
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
                "Answer: (B)\n\n"
                "###QUESTION: Neuroendocrine cells in the lungs are:\n"
                "Option A: Dendritic cells\nOption B: Type I pneumocytes\nOption C: Type II pneumocytes\nOption D: APUD cells\n"
                "Explanation: Neuroendocrine cells, which are also known as Kultschitsky-type cells, Feyrter cells and APUD cells, are found in the basal layer of the surface epithelium and in the bronchial glands.\n"
                "Answer: (D)\n\n"
                "###QUESTION: Presence of it indicates remote contamination of water\n"
                "Option A: Streptococci\nOption B: Staphalococci\nOption C: Clastridium pertringes\nOption D: Nibrio\n"
                "Explanation: Because Clostridium perfringens spores are both specific to sewage contamination and environmentally stable, they are considered as possible conservative indicators of human fecal contamination and possible surrogates for  nvironmentally stable pathogens.\n"
                "Answer: (C)\n\n"),
    "pubmed_qa": (
                "Given three answer candidates, A, B, and C, choose the best answer choice.\n\n"
                # "As an expert doctor in clinical science and medical knowledge, can you tell me if the following statement is correct? Answer yes, no, or maybe.\n\n"
                # "Given the provided context, analyze the relationship or connection between two concepts or factors mentioned in the text. Based on this analysis, generate a response that falls into one of the three classes: 'yes' if there is a connection, 'no' if there is no connection, or 'maybe' if the connection is uncertain or inconclusive.\n\n"
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
    "mmlu": ("Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n"
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
    "mmlu_evidence": ("I will provide context that may be helpful in answering the question. Feel free not to use it if it's not needed. Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n"
             "###QUESTION: The energy for all forms of muscle contraction is provided by:\n"
             "Context: most representative can be presented in a textbook of neurology. The Nature of Primary Metabolic Myopathies The chemical energy for muscle contraction is provided by the hydrolysis of adenosine triphosphate (ATP) to adenosine diphosphate (ADP); ATP is restored by phosphocreatine and ADP acting in combination. These reactions are particularly important during brief, high-intensity exercise. During periods of prolonged muscle activity, rephosphorylation requires the availability of carbohydrates, fatty acids, and ketones, which are catabolized in mitochondria. Glycogen is the main sarcoplasmic source of carbohydrate, but blood glucose also moves freely in and out of muscle cells as needed during sustained exercise. The fatty acids in the blood, derived mainly from adipose tissue and intracellular lipid stores, constitute the other major source of energy. Carbohydrate is metabolized during aerobic and\n"
             "Option A: ATP\nOption B: ADP\nOption C: phosphocreatine\nOption D: oxidative phosphorylation\n"
             "Answer: (A)\n\n"
             "###QUESTION: Which of the following conditions does not show multifactorial inheritance?\n"
             "Context: Some feel this type of eugenic abortion is already underway (sex-selective, etc.) - Knowing about certain birth defects such as spina bifida and teratoma before birth may give the option of fetal surgery during pregnancy, or to assure that the appropriate treatment and/or surgery be provided immediately after birth. - Questions of the value of mentally/physically disabled people in society? - How to ensure that information about testing options is given in a non-directive and supportive way. - That parents are well informed if they have to consider abortion vs. continuing a pregnancy. See wrongful abortion. ## Will the result of the test affect treatment of the fetus? In some genetic conditions, for instance cystic fibrosis, an abnormality can only be detected if DNA is obtained from the\n"
             "Option A: Pyloric stenosis\nOption B: Schizophrenia\nOption C: Spina bifida (neural tube defects)\nOption D: Marfan syndrome\n"
             "Answer: (D)\n\n"
             "###QUESTION: What is the embryological origin of the hyoid bone?\n"
             "Context: An Unusual Case of Bony Styloid Processes That Extend to the Hyoid Bone. The embryological origin of the hyoid bone is a point of uncertainty, with controversy surrounding the relative contribution of the second pharyngeal arch to hyoid development. We encountered a 52-year-old male with bilateral bony styloid extension to the lesser cornu of the hyoid bone during the workup of a patient with laryngeal cancer. This embryological malformation clearly supports the hypothesis that the second pharyngeal arch gives rise to the lesser cornu and demonstrates an unusual clinical finding that may be encountered by otolaryngologists. We demonstrate the imaging findings and surgical management of this unusual anatomical variant and review the embryological basis for this rare malformation.\n"
             "Option A: The first pharyngeal arch\nOption B: The first and second pharyngeal arches\nOption C: The second pharyngeal arch\nOption D: The second and third pharyngeal arches\n"
             "Answer: (D)\n\n"
             ),
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

def _generate(args, query, inst, inst_idx, model, tokenizer, evidences=None, eval_name=None, ret_tokens=None, rel_tokens=None, 
              grd_tokens=None, ut_tokens=None, use_seqscore=False, threshold=0.5, beam_width=2, max_depth=1, w_rel=1.0, w_sup=1.0, w_use=0.5,
                mode="adaptive_retrieval", closed=False):
    
    results = {}
    if mode != "always_retrieve":
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, logprobs=32016)
        preds = model.generate([query], sampling_params)
        pred_token_ids = preds[0].outputs[0].token_ids
        pred_text = preds[0].outputs[0].text
        pred_log_probs = preds[0].outputs[0].logprobs
        results["no_retrieval"] = pred_text

    # save relevance token scores
    if mode == "always_retrieve":
        do_retrieve = True
    elif mode == "no_retrieval":
        do_retrieve = False
    else:
        if threshold is not None:
            score_dict = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict[tok] = float(prob)
            do_retrieve = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieve = "[Retrieval]" in pred_text
    
    if do_retrieve is True:
        try:
            evidence_augmented_inputs = [query + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
                para["title"], para["text"]) for para in evidences[inst_idx]["ctxs"]]
        except:
            evidence_augmented_inputs = [query + "[Retrieval]<paragraph>{0}</paragraph>".format(
                para) for para in evidences[inst_idx]["evidence"]]
        
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, logprobs=5000)
        preds = model.generate(evidence_augmented_inputs, sampling_params)
        
        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / \
                max(len(pred.outputs[0].token_ids), 1)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            if grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        grd_score_dict[p_idx][token] = np.exp(float(prob))

            if ut_tokens is not None:
                utility_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in ut_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(float(prob))

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values())))

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum(
                    [ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
            else:
                utility_score = 0.0

            if use_seqscore is True:
                final_score = np.exp(seq_score) + w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score

            overall_scores[p_idx] = {"final_score": final_score,
                                     "relevance_score": relevance_score,
                                     "ground_score": ground_score,
                                     "utility_score": utility_score,
                                     "relevance_score_dict": relevance_score_dict,
                                     "grd_score_dict": grd_score_dict,
                                     "ut_score_dict": utility_score}
            
            try:
                results["retrieval_{}".format(p_idx)] = {
                    "pred": pred_text, "score": final_score, "ctx": evidences[inst_idx]['ctxs'][p_idx], "overall_score": overall_scores[p_idx]}
            except:
                results["retrieval_{}".format(p_idx)] = {
                    "pred": pred_text, "score": final_score, "ctx": evidences[inst_idx]['evidence'][p_idx], "overall_score": overall_scores[p_idx]}

    else:
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
        query += "[No Retrieval]"
        preds = model.generate([query], sampling_params)
        pred = preds[0].outputs[0].text

    # Aggregating answers
    if len(results) == 1:
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        return postprocessed_pred, results, do_retrieve
    else:
        answer2score = {}
        if closed is True:
            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(
                answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
        else:
            path2score = {key: item["score"] for key,
                          item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(),
                               key=lambda x: x[1], reverse=True)[0][0]
            best_option = results[best_path]["pred"]
        return best_option, results, do_retrieve

def vllm_infer(client, tokenizer, prompt, stop_seq, max_new_tokens=1024, cot=False, temperature=0.0):
    """
    Generates a single output for a given input prompt using the VLLM backend (offline mode).
    Returns the output text.

    Reference:

    :param client: vllm.LLM, the LLM offline generation engine to use for querying the VLLM backend
    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param prompt: str, the prompt to generate from
    :param stop_seq: list, the stop sequence to use for generation
    :param max_new_tokens: int, the maximum number of tokens to generate
    :param cot: bool, whether to use chain-or-thought or not
    :param temperature: float, the temperature to use for sampling
    """

    response = client.generate(prompt, sampling_params=vllm.SamplingParams(
        # See https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
        best_of=1,
        presence_penalty=0.0,
        frequency_penalty=1.0,
        top_k=-1,
        top_p=1.0,
        temperature=temperature,
        stop=stop_seq,
        use_beam_search=False,
        max_tokens=max_new_tokens,
        logprobs=5
    ))

    def top_answer(logprob):
        top_token = max(logprob, key=logprob.get)
        output_text = tokenizer.decode(top_token, skip_special_tokens=True)
        return output_text

    if len(response) > 0:
        return [r.outputs[0].text for r in response]

    if not cot:
        return top_answer(response[0].outputs[0].logprobs[0])
    else:
        return response[0].outputs[0].text

def tokenizer_param(tokenizer, target, shots=0, cot=False, task_type="mcq"):
    """
    Determines the maximum number of tokens to generate for a given prompt and target.
    Also determines the stop sequence to use for generation.

    :param tokenizer: transformers.PreTrainedTokenizer, the tokenizer to use for inference
    :param target: str, the target to generate
    :param shots: int, the number of shots to use for few-shot learning
    :param cot: bool, whether to use chain-or-thought or not
    :param task_type: str, the type of answer to generate (mcq or open)
    """
    max_new_tokens = len(tokenizer(target, add_special_tokens=True)['input_ids'])
    stop_seq = [tokenizer.eos_token, tokenizer.pad_token, "###"]

    if not cot and task_type == "mcq":
        max_new_tokens = len(tokenizer(target[0], add_special_tokens=False)['input_ids'])
        if shots > 0:
            max_new_tokens += 8
    if cot:
        max_new_tokens = 1024

    return max_new_tokens, stop_seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="selfrag/selfrag_llama2_7b")
    parser.add_argument('--write_name', type=str)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--use_few_shot', action="store_true")
    parser.add_argument("--do_retrieve", action="store_true")
    parser.add_argument('--use_lora', action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.")
    parser.add_argument('--lora_model_path', type=str, default="")
    parser.add_argument("--dtype",  type=str, default="half",
                        help="We use bfloat16 for training. If you run inference on GPUs that do not support BF16, please set this to be `half`.")
    # parser.add_argument('--download_dir', type=str, help="specify vllm model download dir", default=".cache")
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default="/scratch/x2696a10/self-rag/retrieval_lm/output/")
    parser.add_argument("--ndocs", type=int, default=10,
                        help="Number of documents to retrieve per questions")
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    # Decoding hyperparams
    parser.add_argument('--threshold', type=float,
                        default=None, help="Adaptive threshold.")
    parser.add_argument("--use_seqscore", action="store_true")
    parser.add_argument("--use_groundness", action="store_true",
                        help="use ground score")
    parser.add_argument(
        "--use_utility", action="store_true", help="tree search")
    parser.add_argument("--beam_width",  type=int,
                        default=2, help="beam search width")
    parser.add_argument("--max_depth",  type=int,
                        default=2, help="tree depth width")
    parser.add_argument("--k", type=int,
                        default=3, help="fewshot (k-shot) examples")
    parser.add_argument("--w_rel",  type=float, default=1.0,
                        help="reward weight for document relevance")
    parser.add_argument("--w_sup",  type=float, default=1.0,
                        help="reward weight for generation support (attribution)")
    parser.add_argument("--w_use",  type=float, default=1.0,
                        help="reward weight for overall completeness / utility.")
    parser.add_argument("--ignore_cont", action="store_true",
                        help="filter out sentences that include [No support / Contradictory] ")
    parser.add_argument('--mode', type=str, help="mode to control retrieval.",
                        default="default", choices=['adaptive_retrieval', 'no_retrieval', 'always_retrieve'],)
    parser.add_argument("--use_train_dataset", action="store_true",
                        help="use train dataset as few shot examples")
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not args.write_name:
        args.write_name = args.model_name_or_path.split("/")[1]

    if "selfbiorag" in args.model_name_or_path or "selfrag" in args.model_name_or_path:
        model = LLM(model=args.model_name_or_path, download_dir=args.download_dir,
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        # Get token ids for reflection tokens.
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
            tokenizer, use_grounding=args.use_groundness, use_utility=args.use_utility)
        
    elif "meta-llama" in args.model_name_or_path:
        if args.use_lora:
            config = PeftConfig.from_pretrained(args.lora_model_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
        else:
            model = LLM(model=args.model_name_or_path, download_dir=args.download_dir,
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    elif "galactica" in args.model_name_or_path:
        model = LLM(model=args.model_name_or_path, download_dir=args.download_dir,
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    elif "Alpaca" in args.model_name_or_path:
        model = LLM(model=args.model_name_or_path, download_dir=args.download_dir,
                    dtype=args.dtype, tensor_parallel_size=args.world_size,)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")

    elif "flan-t5" in args.model_name_or_path:
        #google/flan-t5-xl google/flan-t5-xxl
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    elif "meditron" in args.model_name_or_path:
        # model = AutoModel
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        kwargs = {
            "model": args.model_name_or_path,
            "tokenizer": args.model_name_or_path,
            "trust_remote_code": True,
            "max_num_seqs": 1024,
            "tensor_parallel_size": torch.cuda.device_count(),
        }
        if "7b" in args.model_name_or_path:
            kwargs["tensor_parallel_size"] = 4

        client = vllm.LLM(**kwargs)
    

    # "med_qa", "medmc_qa", "pubmed_qa", "mmlu", "live_qa", "medication_qa", "mmlu_clinical_knowledge", "mmlu_anatomy", "mmlu_college_biology", "mmlu_college_medicine", "mmlu_medical_genetics", "mmlu_professional_medicine"
    evaluation_list = ["med_qa", "medmc_qa", "pubmed_qa", "mmlu_clinical_knowledge", "mmlu_anatomy", "mmlu_college_biology", "mmlu_college_medicine", "mmlu_medical_genetics", "mmlu_professional_medicine"]
    for eval_name in tqdm.tqdm(evaluation_list, desc="total evaluation"):
        test_examples = []
        evidences = []
        train_evidences = []
        with open(f"../data/benchmark/{eval_name}_test.jsonl", "r") as fp:
            lines = fp.readlines()
            for line in lines:
                test_examples.append(json.loads(line))

        if args.do_retrieve:
            with open(f"../data/benchmark/evidence/all/retrieved_{eval_name}_test.jsonl", "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    evidences.extend(json.loads(line))

            # mmlu use evidence with medpalm examples
            if "mmlu" in eval_name:
                with open(f"../data/benchmark/evidence/training/medpalm_mmlu_evidence.json", "r") as fp:
                    lines = fp.readlines()
                    for line in lines:
                        train_evidences.extend(json.loads(line))
            else:
                # evidence - 240105
                with open(f"../data/benchmark/evidence/training/retrieved_{eval_name}_train.jsonl", "r") as fp:
                    lines = fp.readlines()
                    for line in lines:
                        train_evidences.extend(json.loads(line))            
            
        for inst_idx,inst in tqdm.tqdm(enumerate(test_examples), desc="instance"):
            if args.use_few_shot:
                if args.use_train_dataset:
                    train_examples = []
                    train_examples = json.load(open(f"../data/benchmark/{eval_name}_train_gpt4.json"))
                    # with open(f"./data/benchmark/{eval_name}_train.json", "r") as fp:
                    # with open(f"./data/benchmark/{eval_name}_train_gpt4.jsonl", "r") as fp:
                    #     lines = fp.readlines()
                    #     for line in lines:
                    #         train_examples.append(json.loads(line))
                    
                    # knn sampling - use 1000
                    # if os.path.isfile(f"./data/benchmark/{eval_name}_train_embed.npy") and os.path.isfile(f"./data/benchmark/{eval_name}_test_embed.npy"):

                    #     tr_embed_load = np.load(f"./data/benchmark/{eval_name}_train_embed.npy")
                    #     tr_result = torch.from_numpy(tr_embed_load).to('cuda')
                    #     te_embed_load = np.load(f"./data/benchmark/{eval_name}_test_embed.npy")
                    #     te_result = torch.from_numpy(te_embed_load).to('cuda')
                    
                    # knn sampling - use all
                    if os.path.isfile(f"../data/benchmark/{eval_name}_train_gpt4_embed.npy") and os.path.isfile(f"../data/benchmark/{eval_name}_test_embed.npy"):
                        tr_embed_load = np.load(f"./data/benchmark/{eval_name}_train_gpt4_embed.npy")
                        tr_result = torch.from_numpy(tr_embed_load).to('cuda')
                        te_embed_load = np.load(f"./data/benchmark/{eval_name}_test_embed.npy")
                        te_result = torch.from_numpy(te_embed_load).to('cuda')

                        cosine_similarities = F.cosine_similarity(te_result[inst_idx].reshape(1,768), tr_result, dim=1)
                        # Get indices of the top k nearest neighbors
                        top_k_values, top_k_indices = torch.topk(cosine_similarities, args.k)
                        topk_np = top_k_indices.cpu().numpy()
                        train_insts = [tr_inst for tr_idx,tr_inst in enumerate(train_examples) if tr_idx in topk_np]

                        # evidence - 240105
                        if args.do_retrieve:
                            topk_train_evidences = [tr_inst for tr_idx,tr_inst in enumerate(train_evidences) if tr_idx in topk_np]

                    # random sampling
                    # train_insts = random.sample(train_examples, args.k)

                    if eval_name == "med_qa":
                        num2opt = {1:"(A)", 2:"(B)", 3:"(C)", 4:"(D)", 5:"(E)"}
                        for train_inst_idx, train_inst in enumerate(train_insts):
                            
                            split_list = train_inst['instances']['input'].split('\n')[-4:]
                            for split_idx, split_inst in enumerate(split_list):
                                if train_inst['instances']['output'] in split_inst:
                                    answer = num2opt[split_idx+1]
                                    
                            if train_inst_idx == 0:
                                # CoT
                                inst['instruction'] = "Given four answer candidates, A, B, C and D, choose the best answer choice. Let's think step by step.\n\n"
                                
                                # inst['instruction'] = "Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n"

                                # evidence 240105
                                # inst['instruction'] = "I will provide context that may be helpful in answering the question. Feel free not to use it if it's not needed. Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n"

                            # CoT
                            inst['instruction'] += "###" + train_inst['instances']['input'] + "\n" + "Explanation: " + train_inst['explanation'] + "\n" + "Answer: " + answer + "\n\n"
                            
                            # inst['instruction'] += "###" + train_inst['instances']['input'] + "\n" + "Answer: " + answer + "\n\n"
                                
                            # evidence 240105
                            # inst['instruction'] += "###Context: " + topk_train_evidences[train_inst_idx]['evidence'][0] + "\n" + train_inst['instances']['input'] + "\n" + "Answer: " + answer + "\n\n"

                            # evidence with CoT - 240105
                            # inst['instruction'] += "###Context: " + topk_train_evidences[train_inst_idx]['evidence'][0] + "\n" + train_inst['instances']['input'] + "\n" + "Explanation: " + train_inst['explanation'] + "\n" + "Answer: " + answer + "\n\n"

                    elif eval_name == "pubmed_qa":
                        for train_inst_idx, train_inst in enumerate(train_insts):
                            answer = train_inst['instances']['output']
                            if train_inst_idx == 0:
                                inst["instruction"] = "Given three answer candidates, A, B, and C, choose the best answer choice. Let's solve this step by step.\n\n"
                                # inst['instruction'] = "Given the provided context, analyze the relationship or connection between two concepts or factors mentioned in the text. Based on this analysis, generate a response that falls into one of the three classes: 'yes' if there is a connection, 'no' if there is no connection, or 'maybe' if the connection is uncertain or inconclusive. Let's think step by step.\n\n"
                                # inst['instruction'] = train_inst['instruction'] + "\n\n"

                            # inst['instruction'] += "###" + train_inst['instances']['input'] + "\n" + "Answer: " + answer + "\n\n"
                                
                            # CoT
                            # inst["instruction"] += "###" + train_inst['instances']['input'] + "\n" + "Explanation: Let's solve this step by step. " + train_inst['gpt4_explanation'] + '\n\n'
                                
                            # evidence + CoT
                            inst["instruction"] += "###Context: " + topk_train_evidences[train_inst_idx]['evidence'][0] + "\n" + train_inst['instances']['input'] + "\n" + "Explanation: Let's solve this step by step. " + train_inst['gpt4_explanation'] + '\n\n'
                            
                    elif eval_name == "medmc_qa":
                        for train_inst_idx,train_inst in enumerate(train_insts):
                            answer = train_inst['instances']['output']
                            if train_inst_idx == 0:
                                inst['instruction'] = "Given four answer candidates, A, B, C and D, choose the best answer choice. Let's think step by step.\n\n"

                                # evidence 240105
                                # inst['instruction'] = "I will provide context that may be helpful in answering the question. Feel free not to use it if it's not needed. Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n"

                            # inst['instruction'] += "###" + train_inst['instances']['input'] + answer + "\n\n"
                                
                            # evidence 240105
                            # inst['instruction'] += "###Context: " + topk_train_evidences[train_inst_idx]['evidence'][0] + "\n" + train_inst['instances']['input'] + answer + "\n\n"
                            
                            # CoT
                            inst["instruction"] += "###" + train_inst['instances']['input'] + "Explanation: Let's solve this step by step. " + train_inst['gpt4_explanation'] + '\n\n'

                else:
                    try:
                        inst['instruction'] = FEW_SHOT[eval_name]
                    except:
                        # inst['instruction'] = FEW_SHOT["mmlu"]
                        # evidence 240105
                        inst['instruction'] = FEW_SHOT["mmlu_evidence"]
            else:
                try:
                    inst['instruction'] = PROMPT_DICT[eval_name]
                except:
                    inst['instruction'] = PROMPT_DICT["mmlu"]
                    

            # change fewshot examples to training examples
            if eval_name == "med_qa":
                query = inst['instruction'] + "###" + inst['instances']['input']
                # evidence 240105
                # query = inst['instruction'] + "###Context: " + evidences[inst_idx]['evidence'][0] + "\n" + inst['instances']['input'] + "Answer: "
            elif eval_name == "medmc_qa":
                query = inst['instruction'] + "###" + inst['instances']['input'][:-8]
                
                # evidence 240105
                # query = inst['instruction'] + "###Context: " + evidences[inst_idx]['evidence'][0] + "\n" + inst['instances']['input'][:-8]

            elif eval_name == "pubmed_qa":
                input_split = inst['instances']['input'].split('\n')
                context = input_split[0]
                question = input_split[1]
                option = "Option A: Yes\nOption B: No\nOption C: Maybe\n"
                # query = inst['instruction'] + "###CONTEXT: " + context + "\n" + "Question: " + question + "\n" + option

                # evidence 240105
                query = inst['instruction'] + "###Context: " + evidences[inst_idx]['evidence'][0] + "\n" + context + "\n" + "Question: " + question + "\n" + option

            elif "mmlu" in eval_name:
                # query = inst['instruction'] + "###" + inst['instances']['input'][:-8]

                # evidence 240105
                query = inst['instruction'] + "###Context: " + evidences[inst_idx]['evidence'][0] + "\n" + inst['instances']['input'][:-8]

            elif eval_name == "live_qa":
                query = inst['instruction'] + "\n\n###" + inst['instances']['input']
            elif eval_name == "medication_qa":
                query = inst['instruction'] + "\n\n###" + inst['instances']['input']
            
            if "selfbiorag" in args.model_name_or_path or "selfrag" in args.model_name_or_path:
                pred, results, do_retrieve = _generate(args, query, inst, inst_idx, model, tokenizer, evidences=evidences,
                                                       eval_name=eval_name, ret_tokens=ret_tokens, rel_tokens=rel_tokens, 
                                                       grd_tokens=grd_tokens, ut_tokens=ut_tokens, use_seqscore=args.use_seqscore, mode=args.mode, beam_width=args.beam_width, max_depth=args.max_depth, threshold=args.threshold,)
                
                inst['prediction'] = pred
                inst['results'] = results
                inst['do_retrieve'] = do_retrieve
            elif "meta-llama" in args.model_name_or_path:
                # use evidence - inital version 
                # for evi_idx,evidence in enumerate(evidences[inst_idx]['evidence'][:1]):
                #     query += f"Evidence {evi_idx+1}: " + evidence + "\n"

                # query += "Answer: "
                # inputs = tokenizer(query, return_tensors="pt", max_length=args.max_length).to(device)
                # preds = model.generate(**inputs)
                # inst['prediction'] = tokenizer.decode(preds[0]).split("Response:")[-1].strip()
                # inst['prediction'] = tokenizer.decode(preds[0])
                sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
                preds = model.generate([query], sampling_params)
                pred = preds[0].outputs[0].text
                inst['prediction'] = pred
                
            elif "galactica" in args.model_name_or_path:
                sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
                preds = model.generate([query], sampling_params)
                pred = preds[0].outputs[0].text
                inst['prediction'] = pred
                # input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
                # outputs = model.generate(input_ids)
                # inst['prediction'] = tokenizer.decode(outputs[0])
            elif "Alpaca" in args.model_name_or_path:
                sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
                preds = model.generate([query], sampling_params)
                pred = preds[0].outputs[0].text
                inst['prediction'] = pred
            elif "flan-t5" in args.model_name_or_path:
                input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
                outputs = model.generate(input_ids)
                inst['prediction'] = tokenizer.decode(outputs[0])
                
            elif "meditron" in args.model_name_or_path:
                max_len, stop_seq = tokenizer_param(
                    tokenizer, inst['instances']['output'],
                    shots=args.k,
                    cot=True,)        
                preds = vllm_infer(
                    client, tokenizer,
                    [query], stop_seq, max_len,
                    cot=True, temperature=1.0)     
                inst['prediction'] = preds[0]

            if (inst_idx+1) % 10 == 0:
                with open(f"../data/predictions/{args.write_name}_{eval_name}_test.jsonl_tmp", "w") as outfile:
                    for inst in test_examples:
                        outfile.write(json.dumps(inst))
                        outfile.write("\n")

        with open(f"../data/predictions/{args.write_name}_{eval_name}_test.jsonl", "w") as outfile:
            for inst in test_examples:
                outfile.write(json.dumps(inst))
                outfile.write("\n")

if __name__ == "__main__":
    main()