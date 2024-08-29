import json
import re
from eval_utils import overall_jga
from fuzzywuzzy import fuzz
from collections import defaultdict
import os
from dataset_utils import create_samples
def extract_and_parse_json(text):
    # Regular expression to find the JSON object within the text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError:
            # print("Error parsing JSON.")
            return None
    else:
        # print("No JSON found in the text.")
        return None
    
results = None
with open("finetuned_result/test/result_SFT_TAXI.json") as result_file:
    results_sft_taxi = json.load(result_file)
with open("log/result_all.json") as result_file:
    results_base = json.load(result_file)
with open("finetuned_result/test/result_sft_25_50.json") as result_file:
    result_sft_mixed= json.load(result_file)
with open("finetuned_result/test/result_ft_all.json") as result_file:
    result_ft_all= json.load(result_file)
def parse_results(results):
    parse_error = 0
    parsed_results = []
    for turn in results:
        try:
            bs_pred = json.loads(turn["bs_pred"])
        except:
            bs_pred = extract_and_parse_json(turn["bs_pred"]) #additional parsing if json.loads doesn't work
        bs_gt = turn["bs_gt"]
        if bs_pred == None: #parse error
            parse_error += 1
            # print(f"Unextracted:{turn['bs_pred']}")
        else:
            if "intent" in bs_pred.keys():
                try:
                    slots = bs_pred["slots"] #extract the results other than intent
                except:
                    # print(f"no slots:{bs_pred}")
                    slots = {}
            else:
                slots = bs_pred
            
            parsed_results.append({"bs_gt":bs_gt,"bs_pred":slots, "utt":turn["utt"]})
    print(f"total parse errors:{parse_error}")    
    return parsed_results
#filtering not specified values
#def calc_JGA
def get_results(parsed_results):
    for result in parsed_results:
        result["bs_pred"] = {key:value for key, value in result["bs_pred"].items() if value not in ["not specified","unknown"]}
    def eval_JGA(parsed_results):
        #check for JGA
        correct_turns= 0
        collected_correct_turns = []
        for result in parsed_results:
            gt = result["bs_gt"] #ground truth
            pred = result["bs_pred"] #predicted belief state
            pred = {k:v for k,v in pred.items() if v not in ["","null",None,"unspecified"]}
            gt_key = gt.keys()
            pred_key = pred.keys()
            match = gt_key == pred_key
            if match:
                for key in gt_key:
                    if fuzz.partial_ratio(str(pred[key]), str(gt[key])) <= 95:
                        match = False
            
            if match:
                correct_turns +=1
                collected_correct_turns.append(result)

        return correct_turns/len(parsed_results), collected_correct_turns
    jga, correct_turns = eval_JGA(parsed_results)
    with open("correct_turns_train_500.json","w") as correct:
        json.dump(correct_turns,correct,indent=4)
    print(f"Overall JGA:{jga} from {len(parsed_results)} turns")
    slots = None
    with open("mwz2.4/ontology.json","r") as ontology:
        slots = json.load(ontology).keys()
    slots = {slot.replace(" ","") for slot in slots}
    errors_by_domain_fn = {slot: 0 for slot in slots}
    errors_by_domain_fp = {slot: 0 for slot in slots}
    raw_results = {slot:{"fn":[],"fp":[]} for slot in slots}
    domain_counts = {"restaurant":0,"hotel":0,"train":0,"attraction":0,"taxi":0}

    for items in parsed_results:
        dialogue_domains= set()
        bs_gt = items["bs_gt"]
        bs_pred = items["bs_pred"]
        for slot, true_value in bs_gt.items():
            predicted_value = bs_pred.get(slot, None)
            domain = slot.split("-")[0]
            if domain in domain_counts.keys():
                dialogue_domains.add(domain)
            if predicted_value != true_value and slot in errors_by_domain_fn.keys():
                errors_by_domain_fn[slot] +=1
                raw_results[slot]["fn"].append(items)
        for domain in dialogue_domains:
            domain_counts[domain] += 1
        for slot, pred_value in bs_pred.items():
            true_value = bs_gt.get(slot, None)
            if true_value == None and slot in errors_by_domain_fp.keys():
                errors_by_domain_fp[slot] +=1
                raw_results[slot]["fp"].append(items)
    # Calculate error percentages
    error_percentages = {}
    for slot in errors_by_domain_fp.keys():
        domain = slot.split("-")[0]
        if errors_by_domain_fp[slot] >0 and domain_counts[domain] > 0:  # Avoid division by zero
            percentage_error = (errors_by_domain_fp[slot] / domain_counts[domain]) * 100
            error_percentages[slot] = percentage_error

    # Sort slots by highest error percentage and get the top n
    n = 5  # Set the value of n to get the top n highest errors
    top_n_errors = sorted(error_percentages.items(), key=lambda x: x[1], reverse=True)[:n]

    # Print the top n slots with the highest error percentages
    print(f"Top {n} slots with highest error percentages among false positive errors:")
    for slot, percentage in top_n_errors:
        print(f"{slot}: {percentage:.2f}%")


    # Calculate error percentages
    error_percentages_fn = {}
    for slot in errors_by_domain_fn.keys():
        domain = slot.split("-")[0]
        if errors_by_domain_fn[slot] >0 and domain_counts[domain] > 0:  # Avoid division by zero
            percentage_error = (errors_by_domain_fn[slot] / domain_counts[domain]) * 100
            error_percentages_fn[slot] = percentage_error

    # # Sort slots by highest error percentage and get the top n
    # n = 5  # Set the value of n to get the top n highest errors
    # top_n_errors = sorted(error_percentages_fn.items(), key=lambda x: x[1], reverse=True)[:n]

    # # Print the top n slots with the highest error percentages
    # print(f"Top {n} slots with highest error percentages among false negative errors:")
    # for slot, percentage in top_n_errors:
    #     print(f"{slot}: {percentage:.2f}%")
    #     path = f"problematic_cases/{slot}"
    #     sample_size = 40
    #     # if not os.path.isdir(path):
    #     #     os.mkdir(path)
    #     # with open(f"{path}/fn_train_500.json","w") as f:
    #     #     json.dump(create_samples(raw_results[slot]["fn"],sample_size), f, indent = 4)
    #     # with open(f"{path}/fp_train_500.json","w") as f:
    #     #     json.dump(create_samples(raw_results[slot]["fp"],sample_size),f, indent= 4)
    return error_percentages_fn,error_percentages



parsed_results_base = parse_results(results_base)
parsed_results_ft_taxi = parse_results(results_sft_taxi)
parsed_results_ft_mixed = parse_results(result_sft_mixed)
parsed_results_ft_all = parse_results(result_ft_all)
error_fn_base, error_fp_base = get_results(parsed_results_base)
error_fn_ft_taxi, error_fp_ft_taxi = get_results(parsed_results_ft_taxi)
error_fn_ft_mixed, error_fp_ft_mixed = get_results(parsed_results_ft_mixed)
error_fn_ft_all, error_fp_ft_all = get_results(parsed_results_ft_all)
top_5_errors = sorted(error_fn_base.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"top 5 erors:{top_5_errors}")
print(f"base/sft_taxi_departure_destination/sft_mixed for false negatives")
for error in top_5_errors:
    slot_name = error[0]
    print(f"{slot_name}:{error_fn_base[slot_name]} / {error_fn_ft_taxi[slot_name]} / {error_fn_ft_mixed[slot_name]} / {error_fn_ft_all[slot_name]}")

print("*"*10)
print(f"base/sft_taxi_departure_destination/sft_mixed for false positives")
top_5_errors = sorted(error_fp_base.items(), key=lambda x: x[1], reverse=True)[:5]
for error in top_5_errors:
    slot_name = error[0]
    print(f"{slot_name}:{error_fp_base[slot_name]} / {error_fp_ft_taxi[slot_name]} / {error_fp_ft_mixed[slot_name]} / {error_fp_ft_all[slot_name]}")