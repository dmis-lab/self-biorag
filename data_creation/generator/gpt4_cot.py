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

openai.api_key_path = "/nvme1/minbyul/self-rag/key.txt"
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

FEW_SHOT = {
    "medmc_qa": (
                "Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n"
                "###QUESTION: Maximum increase in prolactin level is caused by:\n"
                "Option A: Risperidone\nOption B: Clozapine\nOption C: Olanzapine\nOption D: Aripiprazole\n"
                "Explanation: Let's think step by step. Clozapine generally does not raise prolactin levels. Atypicals such as olanzapine and aripiprazole cause small if no elevation. Risperidone is known to result in a sustained elevated prolactin level. Therefore risperidone is likely to cause the maximum increase in prolactin level.\n"
                "Answer: (A)\n\n"
                "###QUESTION: What is the age of routine screening mammography?"
                "Option A: 20 years\nOption B: 30 years\nOption C: 40 years\nOption D: 50 years\n"
                "Explanation: Let's think step by step. The age of routine screening depends on the country you are interested in and varies widely. For the US, it is 40 years of age according to the American Cancer Society. In Europe, it is typically closer to 50 years. For a patient based in the US, the best answer is 40 years.\n"
                "Answer: (C)\n\n"
                "###QUESTION: A 65-year-old male complains of severe back pain and inability to move his left lower limb. Radiographic studies demonstrate the compression of nerve elements at the intervertebral foramen between vertebrae L5 and S1. Which structure is most likely responsible for this space-occupying lesion?\n"
                "Option A: Anulus fibrosus\nOption B: Nucleus pulposus\nOption C: Posterior longitudinal ligament\nOption D: Anterior longitudinal ligament\n"
                "Explanation: Let's think step by step. This man describes a herniated invertebral disk through a tear in the surrounding annulus fibrosus. The soft, gelatinous \"nucleus pulposus\" is forced out through a weakened part of the disk, resulting in back pain and nerve root irritation. In this case, the impingement is resulting in paralysis, and should be considered a medical emergency. Overall, the structure that is causing the compression and symptoms is the nucleus pulposus.\n"
                "Answer: (B)\n\n"
                "###QUESTION: Neuroendocrine cells in the lungs are:\n"
                "Option A: Dendritic cells\nOption B: Type I pneumocytes\nOption C: Type II pneumocytes\nOption D: APUD cells\n"
                "Explanation: Let's think step by step. Neuroendocrine cells, which are also known as Kultschitsky-type cells, Feyrter cells and APUD cells, are found in the basal layer of the surface epithelium and in the bronchial glands.\n"
                "Answer: (D)\n\n"
                "###QUESTION: Presence of it indicates remote contamination of water\n"
                "Option A: Streptococci\nOption B: Staphalococci\nOption C: Clastridium pertringes\nOption D: Nibrio\n"
                "Explanation: Let's think step by step. Because Clostridium perfringens spores are both specific to sewage contamination and environmentally stable, they are considered as possible conservative indicators of human fecal contamination and possible surrogates for  nvironmentally stable pathogens.\n"
                "Answer: (C)\n\n"),
    "pubmed_qa": (
                "Given three answer candidates, A, B, and C, choose the best answer choice.\n\n"
                "###CONTEXT: To describe the interstitial fluid (ISF) and plasma pharmacokinetics of meropenem in patients on continuous venovenous haemodiafiltration (CVVHDF). This was a prospective observational pharmacokinetic study. Meropenem (500 mg) was administered every 8 h. CVVHDF was targeted as a 2-3 L/h exchange using a polyacrylonitrile filter with a surface area of 1.05 m2 and a blood flow rate of 200 mL/min. Serial blood (pre- and post-filter), filtrate/dialysate and ISF concentrations were measured on 2 days of treatment (Profiles A and B). Subcutaneous tissue ISF concentrations were determined using microdialysis. A total of 384 samples were collected. During Profile A, the comparative median (IQR) ISF and plasma peak concentrations were 13.6 (12.0-16.8) and 40.7 (36.6-45.6) mg/L and the trough concentrations were 2.6 (2.4-3.4) and 4.9 (3.5-5.0) mg/L, respectively. During Profile B, the ISF trough concentrations increased by ∼40%. Meropenem ISF penetration was estimated at 63% (60\%-69\%) and 69% (65\%-74\%) for Profiles A and B, respectively, using comparative plasma and ISF AUCs. For Profile A, the plasma elimination t1/2 was 3.7 (3.3-4.0) h, the volume of distribution was 0.35 (0.25-0.46) L/kg, the total clearance was 4.1 (4.1-4.8) L/h and the CVVHDF clearance was 2.9 (2.7-3.1) L/h.\n"
                "QUESTION: Are interstitial fluid concentrations of meropenem equivalent to plasma concentrations in critically ill patients receiving continuous renal replacement therapy?\n"
                "Option A: Yes\nOption B: No\nOption C: Maybe\n"
                "Explanation: Let's think step by step. This is the first known report of concurrent plasma and ISF concentrations of a meropenem antibiotic during CVVHDF. We observed that the ISF concentrations of meropenem were significantly lower than the plasma concentrations,although the present dose was appropriate for infections caused by intermediately susceptible pathogens (MIC≤4 mg/L).\n"
                "Answer: (B)\n\n"
                "###CONTEXT: Family caregivers of dementia patients are at increased risk of developing depression or anxiety. A multi-component program designed to mobilize support of family networks demonstrated effectiveness in decreasing depressive symptoms in caregivers. However, the impact of an intervention consisting solely of family meetings on depression and anxiety has not yet been evaluated. This study examines the preventive effects of family meetings for primary caregivers of community-dwelling dementia patients. A randomized multicenter trial was conducted among 192 primary caregivers of community dwelling dementia patients. Caregivers did not meet the diagnostic criteria for depressive or anxiety disorder at baseline. Participants were randomized to the family meetings intervention (n=96) or usual care (n=96) condition. The intervention consisted of two individual sessions and four family meetings which occurred once every 2 to 3 months for a year. Outcome measures after 12 months were the incidence of a clinical depressive or anxiety disorder and change in depressive and anxiety symptoms (primary outcomes), caregiver burden and quality of life (secondary outcomes). Intention-to-treat as well as per protocol analyses were performed. A substantial number of caregivers (72/192) developed a depressive or anxiety disorder within 12 months. The intervention was not superior to usual care either in reducing the risk of disorder onset (adjusted IRR 0.98; 95% CI 0.69 to 1.38) or in reducing depressive (randomization-by-time interaction coefficient=-1.40; 95% CI -3.91 to 1.10) or anxiety symptoms (randomization-by-time interaction coefficient=-0.55; 95% CI -1.59 to 0.49). The intervention did not reduce caregiver burden or their health related quality of life.\n"
                "QUESTION: Does a family meetings intervention prevent depression and anxiety in family caregivers of dementia patients?\n"
                "Option A: Yes\nOption B: No\nOption C: Maybe\n"
                "Explanation: Let's think step by step. This study did not demonstrate preventive effects of family meetings on the mental health of family caregivers. Further research should determine whether this intervention might be more beneficial if provided in a more concentrated dose, when applied for therapeutic purposes or targeted towards subgroups of caregivers.\n"
                "Answer: (B)\n\n"
                "###CONTEXT: To compare adherence to follow-up recommendations for colposcopy or repeated Papanicolaou (Pap) smears for women with previously abnormal Pap smear results. Retrospective cohort study. Three northern California family planning clinics. All women with abnormal Pap smear results referred for initial colposcopy and a random sample of those referred for repeated Pap smear. Medical records were located and reviewed for 90 of 107 women referred for colposcopy and 153 of 225 women referred for repeated Pap smears. Routine clinic protocols for follow-up–telephone call, letter, or certified letter–were applied without regard to the type of abnormality seen on a Pap smear or recommended examination. Documented adherence to follow-up within 8 months of an abnormal result. Attempts to contact the patients for follow-up, adherence to follow-up recommendations, and patient characteristics were abstracted from medical records. The probability of adherence to follow-up vs the number of follow-up attempts was modeled with survival analysis. Cox proportional hazards models were used to examine multivariate relationships related to adherence. The rate of overall adherence to follow-up recommendations was 56.0% (136/243). Adherence to a second colposcopy was not significantly different from that to a repeated Pap smear (odds ratio, 1.40; 95\% confidence interval, 0.80-2.46). The use of as many as 3 patient reminders substantially improved adherence to follow-up. Women without insurance and women attending 1 of the 3 clinics were less likely to adhere to any follow-up recommendation (hazard ratio for no insurance, 0.43 [95\% confidence interval, 0.20-0.93], and for clinic, 0.35 [95\% confidence interval, 0.15-0.73]).\n"
                "QUESTION: Do follow-up recommendations for abnormal Papanicolaou smears influence patient adherence?\n"
                "Option A: Yes\nOption B: No\nOption C: Maybe\n"
                "Explanation: Let's think step by step. Adherence to follow-up was low in this family planning clinic population, no matter what type of follow-up was advised. Adherence was improved by the use of up to 3 reminders. Allocating resources to effective methods for improving adherence to follow-up of abnormal results may be more important than which follow-up procedure is recommended.\n"
                "Answer: (B)\n\n"),
}


def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='+')
    parser.add_argument('--model_name', type=str, default="gpt-4")
    parser.add_argument('--output_file_name', type=str)
    parser.add_argument('--eval_name', type=str)
    args = parser.parse_args()

    examples = []
    for input_file in args.input_files:
        examples += load_jsonlines(input_file)

    result_list = []
    for idx, example in tqdm(enumerate(examples)):
        if args.eval_name == "pubmed_qa":
            input = FEW_SHOT[args.eval_name]
            input += "###" + example['instances']['input'] + "\n"
            input += "Option A: Yes\nOption B: No\nOption C: Maybe\n"
            input += "Explanation: Let's think step by step. "

        elif args.eval_name == "medmc_qa":
            input = FEW_SHOT[args.eval_name]
            input += "###" + example['instances']['input']
            input += "Explanation: Let's think step by step. "
            
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
            example["gpt4_explanation"] = results["choices"][0]["message"]["content"]
            result_list.append(example)

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
