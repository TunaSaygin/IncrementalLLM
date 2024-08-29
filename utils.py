import json
import os
import pandas as pd
import copy
from datasets import Dataset, DatasetDict, load_dataset

#------------------------------------------------

domain_list = ["attraction", "hotel", "restaurant", "taxi", "train"]
attraction_slots = ["attraction-name", "attraction-type", "attraction-area"]
hotel_slots = ["hotel-name", "hotel-type", "hotel-parking", "hotel-area", "hotel-bookday", "hotel-bookstay", "hotel-internet", "hotel-bookpeople", "hotel-stars", "hotel-pricerange"]
restaurant_slots = ["restaurant-name", "restaurant-food", "restaurant-area", "restaurant-bookday", "restaurant-booktime", "restaurant-bookpeople", "restaurant-pricerange"]
taxi_slots = ["taxi-arriveby", "taxi-departure", "taxi-leaveat", "taxi-destination"]
train_slots = ["train-arriveby", "train-day", "train-leaveat", "train-destination", "train-departure", "train-bookpeople"]

hotel_detail = """
 - "hotel-area" that specifies the area where the hotel is located (north, east, west, south, centre, etc.)
 - "hotel-internet" that specifies if the hotel has internet (yes/no)
 - "hotel-parking" that specifies if the hotel has parking (yes/no)
 - "hotel-stars" that specifies the number of stars the hotel has (1/2/3/4/5)
 - "hotel-type" that specifies the type of the hotel (hotel/bed and breakfast/guest house)
 - "hotel-pricerange" that specifies the price range of the hotel (cheap/expensive)
 - "hotel-name" that specifies name of the hotel
 - "hotel-bookstay" specifies length of the stay
 - "hotel-bookday" specifies the day of the booking
 - "hotel-bookpeople" specifies how many people should be booked for
Do not capture any other slots for hotel!"""

restaurant_detail = """
 - "restaurant-pricerange" that specifies the price range of the restaurant (cheap/moderate/expensive)
 - "restaurant-area" that specifies the area where the restaurant is located (north, east, west, south, centre, etc.)
 - "restaurant-food" that specifies the type of food the restaurant serves
 - "restaurant-name" that is the name of the restaurant
Do not capture any other slots for restaurant!"""

attraction_detail = """
 - "attraction-type" that specifies the type of attraction (museum, entertainment, college, nightclub, etc.)
 - "attraction-area" that specifies the area where the attraction is located (north, east, west, south, centre, etc.)
 - "attraction-name" that specigies the name of the attraction
Do not capture any other slots for attraction!"""

taxi_detail = """
 - "taxi-arriveby" that specifies what time the train should arrive
 - "taxi-leaveat" that specifies what time the train should leave
 - "taxi-departure" that specifies the departure station
 - "taxi-destination" that specifies the destination station
 Do not capture any other slots for taxi!"""

train_detail = """
 - "train-arriveby" that specifies what time the train should arrive
 - "train-leaveat" that specifies what time the train should leave
 - "train-day" that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - "train-departure" that specifies the departure station
 - "train-destination" that specifies the destination station
 - "train-bookpeople" that specifies how many people the booking is for
Do not capture any other slots for train!"""

#slot_details_txt = f"# Hotel slots\nThe user can ask for a hotel by different slots described as follows. {hotel_detail}\n\n"
#slot_details_txt = slot_details_txt + f"# Attraction slots\nThe user can ask for a attraction by different slots described as follows. {attraction_detail}\n\n"
#slot_details_txt = slot_details_txt + f"# Restaurant slots\nThe user can ask for a restaurant by different slots described as follows.. {restaurant_detail}\n\n"
#slot_details_txt = slot_details_txt + f"# Taxi slots\nThe user can ask for a taxi by different slots described as follows. {taxi_detail}\n\n"
#slot_details_txt = slot_details_txt + f"# Train slots\nThe user can ask for a train by different slots described as follows. {train_detail}\n\n"

slot_details_txt = f"""The user can ask for a hotel by slots - {", ".join(hotel_slots)}. The user can ask for an attraction by slots - {", ".join(attraction_slots)}. The user can ask for a restaurant by slots - {", ".join(restaurant_slots)}. The user can ask for a taxi by slots - {", ".join(taxi_slots)}. The user can ask for a train by slots - {", ".join(train_slots)}. Do not capture any other slots!\n\n"""

max_token_limit = 900

#------------------------------------------------

slot_dict = {}

lst_slots = []
lst_slots.extend(attraction_slots)
lst_slots.extend(hotel_slots)
lst_slots.extend(restaurant_slots)
lst_slots.extend(taxi_slots)
lst_slots.extend(train_slots)

for i in range(len(lst_slots)):
    slot_dict[lst_slots[i]] = i

#------------------------------------------------

def get_slot_list():
    return lst_slots

def getBeliefSet(bs):
    bs_new = {}
    for i in range(len(bs)):
        sv = bs[i]['slots']
        for j in range(len(sv)):
            slot_key = sv[j][0]
            slot_key = slot_key.replace("book ","book")
            slot_value = sv[j][1]
            if(slot_key in slot_dict):
                bs_new[slot_key] = slot_value
    return bs_new

def get_slot_label(bs):
    lbl_slots = [0]*len(slot_dict)
    for slot in bs:
        idx = slot_dict[slot]
        lbl_slots[idx] = 1
    return lbl_slots

def is_valid_domain(dt_mwz24):
    is_valid = True
    #str_domain = " ".join(domain_list)
    for j in range(len(dt_mwz24['dialogue'])):
        domain = dt_mwz24['dialogue'][j]['domain'].strip()
        if(len(domain)>0 and domain not in domain_list):
            is_valid = False
            break
    return is_valid
    

def load_mwz_data(filename, mode, test_run, conv_limit, logger, tokenizer=None):
    with open(filename, "r") as f:
        data_mwz24 = json.load(f)
    
    c_conv = 0
    c_turn = 0
    data_dict = {"utt": [], "bs": [], "slots": []}
    for i in range(len(data_mwz24)):
        dt_mwz24 = data_mwz24[i]
        if(is_valid_domain(dt_mwz24)):
            c_conv+=1
            conv_hist = ""
            for j in range(len(dt_mwz24['dialogue'])):
                sys = dt_mwz24['dialogue'][j]['system_transcript']
                usr = dt_mwz24['dialogue'][j]['transcript']
                
                turn_hist = f"<|start_header_id|>system<|end_header_id|>\n{sys.strip()}<|eot_id|>\n"
                turn_hist = turn_hist + f"<|start_header_id|>user<|end_header_id|>\n{usr.strip()}<|eot_id|>\n"
                conv_hist = conv_hist + turn_hist
                
                bs = getBeliefSet(dt_mwz24['dialogue'][j]['belief_state'])
                lbl_slots = get_slot_label(bs) 
                tmp_hist = copy.deepcopy(conv_hist)

                add_sample = True
                # if(mode=="train"):
                #     inp = tokenizer.encode(tmp_hist)
                #     if(len(inp) > max_token_limit):
                #         add_sample = False
                    
                if(add_sample):
                    data_dict["utt"].append(tmp_hist)
                    data_dict["bs"].append(json.dumps(bs))
                    data_dict["slots"].append(lbl_slots)
                    c_turn+=1
            
        if(test_run and c_conv==conv_limit):
            break
    
    logger.info(f"{mode} data: total conversations = {c_conv} : total turns: {c_turn}")
    logger.info("-"*30)
    return data_dict
    
def get_mwz_dataset(test_run, conv_limit, logger, tokenizer, train_all):    
    train_file = "mwz2.4/train_dials.json"
    dev_file = "mwz2.4/dev_dials.json"
    #test_file = "../MultiWOZ2.4-main/data/mwz2.4/test_dials.json"

    train_data = load_mwz_data(train_file, "train", test_run, conv_limit, logger, tokenizer)
    dev_data = load_mwz_data(dev_file, "dev", test_run, conv_limit, logger)
    #test_data = load_mwz_data(test_file, "test", test_run, conv_limit)

    dataset = DatasetDict()
    if(test_run or train_all):
        dataset["train"] = Dataset.from_dict(train_data)
    else:
        #print("Sample train data")
        train_data = Dataset.from_dict(train_data).shuffle(seed=0)
        n = 20000
        s_idx = [v for v in range(n)]
        sample_train_data = train_data.select(s_idx)
        dataset["train"] = sample_train_data
            
    dataset["dev"] = Dataset.from_dict(dev_data)
    #dataset["test"] = Dataset.from_dict(test_data)
    
    return dataset

def get_mwz_test_dataset(test_run, conv_limit, logger):    
    test_file = "../MultiWOZ2.4-main/data/mwz2.4/test_dials.json"
    test_data = load_mwz_data(test_file, "test", test_run, conv_limit, logger)

    dataset = DatasetDict()
    dataset["test"] = Dataset.from_dict(test_data)
    return dataset

def get_prompt(is_verify, conv_history, bs_pred=None):
    if(is_verify):
        prompt = "You are a helpful assistant who can verify dialogue-state tracking. The user interacts with the system to book entities from multiple domains (hotel, restaurant, attraction, taxi, and train) in Cambridge. Your goal is to find the mistakes in the predicted user intents.\n\n"
    else:
        prompt = "You are a helpful assistant who can perform dialogue-state tracking. The user interacts with the system to book entities from multiple domains (hotel, restaurant, attraction, taxi, and train) in Cambridge. Your goal is to find all the intents shown by the user in the conversation.\n\n"
    prompt = prompt + slot_details_txt
    
    if(is_verify):
        prompt = prompt + "# Task\nYou will be provided with a chronological dialogue history between the system and the user and the predicted user intents. You must find the false negatives and the false positives in the prediction and output them in JSON format.n\n"
        prompt = prompt + """# Sample Output\n{"false_negatives":{"restaurant-name": "abc"}, "false_positives":{"restaurant-area": "xyz"}}\n\n"""
    else:
        prompt = prompt + "# Task\nYou will be provided with a chronological dialogue history between the system and the user. You must find all the user intents and output them in JSON format.\n\n"
        prompt = prompt + """# Sample Output\n{"restaurant-name": "abc", "restaurant-food": "xyz"}\n\n"""
        
    prompt = prompt + f"# Conversation History\n"
    prompt = prompt + f"{conv_history}\n"
    
    if(is_verify):
        prompt = prompt + f"# Predicted Intents\n{bs_pred}\n"
        
    prompt = prompt + f"<|start_header_id|>assistant<|end_header_id|>"
    return prompt

    
