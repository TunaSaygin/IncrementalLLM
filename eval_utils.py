from fuzzywuzzy import fuzz
from collections import Counter
def flatten(state_dict, single_domain=""):
    constraints = {}
    if single_domain:
        for s, v in state_dict.items():
            constraints[(single_domain, s)] = v
    else:
        for domain, state in state_dict.items():
            for s, v in state.items():
                constraints[(domain, s)] = v
    return constraints

def is_matching(hyp, ref, false_positives, false_negatives):
    hyp_k = hyp.keys()
    ref_k = ref.keys()
    if hyp_k != ref_k:
        for k in hyp_k - ref_k:
            if not (k == ("train", "leaveat") and ("train", "leave") in ref_k) and not (k == ("taxi", "leaveat") and ("taxi", "leave") in ref_k):
                false_positives.append((k, hyp[k]))
        for k in ref_k - hyp_k:
            if not (k == ("train", "leave") and ("train", "leaveat") in hyp_k) and not (k == ("taxi", "leave") and ("taxi", "leaveat") in hyp_k):
                false_negatives.append((k, ref[k]))
        return False
    
    for k in ref_k:
        if k == ("train", "leave"):
            if ("train", "leaveat") in hyp_k and fuzz.partial_ratio(hyp[("train", "leaveat")], ref[k]) > 95:
                continue
            else:
                false_negatives.append((("train", "leaveat"), ref[k]))
        elif k == ("taxi", "leave"):
            if ("taxi", "leaveat") in hyp_k and fuzz.partial_ratio(hyp[("taxi", "leaveat")], ref[k]) > 95:
                continue
            else:
                false_negatives.append((("taxi", "leaveat"), ref[k]))
        elif k in hyp_k and fuzz.partial_ratio(hyp[k], ref[k]) <= 95:
            false_negatives.append((k, ref[k]))
            # false_positives.append((k, hyp[k]))
            return False
    return True


def overall_jga(input_data, reference_states):
    """ Get dialog state tracking results: joint accuracy (exact state match), slot F1, precision and recall """
    print("overall_jga")
    joint_match = 0
    
    num_turns = 0
    false_positives = []
    false_negatives = []
    
    for i, turn in enumerate(input_data):
        ref = flatten(reference_states[i])
        hyp = flatten(turn)
        if is_matching(hyp, ref, false_positives, false_negatives):
            joint_match += 1

        num_turns += 1
    
    joint_match = joint_match / num_turns
    # Count occurrences of each slot name in false positives and false negatives
    false_positive_counts = Counter(slot for slot, _ in false_positives)
    false_negative_counts = Counter(slot for slot, _ in false_negatives)
    
    result = {
        'overall JGA': joint_match,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'false_positive_counts': false_positive_counts,
        'false_negative_counts': false_negative_counts
    }
    
    return result