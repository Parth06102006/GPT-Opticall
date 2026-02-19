SCORE_WEIGHTS = {
    "agent_opening":{
        "score":20,
        "fields":[
            "agent_opening_within_5_sec",
            "agent_used_opening_script",
            "agent_used_brand_name",
            "agent_used_closing_script",
            "agent_used_further_assistance_script"
        ],
        "max_score_column":"max_agent_opening_and_closing_score",
        "score_column":"agent_opening_and_closing_score"
    },
    "hold_and_transfer":{
        "score":20,
        "fields":[
            "agent_used_correct_hold_script",
            "agent_excessive_hold_duration",
            "agent_explained_issue_to_l2",
            "agent_followed_escalation_policy",
        ],
        "max_score_column":"max_hold_and_transfer_score",
        "score_column":"hold_and_transfer_score"
    },
    "comprehending_qrc":{
        "score":20,
        "fields":[
            "agent_attentive_throughout",
            "agent_probing_logical_and_correct",
            "agent_allowed_customer_to_complete",
        ],
        "max_score_column":"max_comprehending_qrc_score",
        "score_column":"comprehending_qrc_score"
    },
    "language":{
        "score":20,
        "fields":[
            "agent_used_jargon",
            "agent_spoke_in_customer_language",
            "agent_fumbled_or_stammered",
        ],
        "max_score_column":"max_language_score",
        "score_column":"language_score"
    },
    "proactive_info":{
        "score":20,
        "fields":[
            "agent_shared_proactive_info",
        ],
        "max_score_column":"max_proactive_info_score",
        "score_column":"proactive_info_score"
    }
}


FATAL_FIELDS = [
    "agent_verified_customer_details",
    "agent_hold_used_less_than_3_times",
    "agent_ensured_warm_transfer",
    "agent_excessive_hold_duration",
    "agent_tone_type",
    "agent_used_absurd_or_abusive_language",
    "agent_communicated_tat"
]

TOOGLE_FIELDS = ["agent_used_jargon",
    "agent_fumbled_or_stammered", 
    "agent_excessive_hold_duration",
    "agent_tone_type",
    "agent_used_absurd_or_abusive_language"
]
SCORE_WEIGHTS_KEYS=[
    ["max_agent_opening_and_closing_score","agent_opening_and_closing_score"],
    ["max_hold_and_transfer_score","hold_and_transfer_score"],
    ["max_comprehending_qrc_score","comprehending_qrc_score"],
    ["max_language_score","language_score"],
    ["max_proactive_info_score","proactive_info_score"]
]

def clean_data(output_data):

    for i in range(len(output_data)):
        
        if i==len(output_data)-1:
            output_data[i]["agent_ensured_warm_transfer"]=None
            output_data[i]["agent_explained_issue_to_l2"]=None
            output_data[i]["agent_followed_escalation_policy"]=None
            output_data[i]["call_escalated"]=False
        
        if i!=0:
            output_data[i]["agent_ensured_warm_transfer"]=None
        
        if i<len(output_data)-1:
            output_data[i]["agent_pitched_csat"]=None
            output_data[i]["agent_used_closing_script"]=None
            output_data[i]["agent_used_further_assistance_script"]=None
            output_data[i]["agent_communicated_tat"]=None
            output_data[i]["call_dropped"]=None
            output_data[i]["call_escalated"]=True
            output_data[i]["l2_transfer"]=True
            
        # if i>0:
        #     output_data[i]["duration_dead_air"]+=output_data[i-1]["duration_dead_air"]
        #     output_data[i]["duration_silence"]+=output_data[i-1]["duration_silence"]
        #     output_data[i]["total_hold_duration"]+=output_data[i-1]["total_hold_duration"]
        #     output_data[i]["no_of_call_holds"]+=output_data[i-1]["no_of_call_holds"]
        
        if i != 0:
            output_data[i]["agent_verified_customer_details"] = None
        
        if ("no_of_call_holds" in output_data[i].keys()):
            output_data[i]["agent_hold_used_less_than_3_times"]= True
            if (output_data[i].get("no_of_call_holds") == 0):
                output_data[i]["agent_used_correct_hold_script"] = None
                output_data[i]["agent_sought_hold_permission"] = None
            if (output_data[i].get("no_of_call_holds") >3):
                # output_data[i]["agent_excessive_hold_duration"] = True if output_data[i].get("total_hold_duration")/output_data[i].get("no_of_call_holds") > 51 else False
                output_data[i]["agent_hold_used_less_than_3_times"] = False
        
        if ("issue_resolved" in output_data[i].keys()):
            if (output_data[i].get("issue_resolved") == True):
                output_data[i]["agent_communicated_tat"] = None
            else:
                output_data[i]["agent_thanked_and_asked_additional_queries"] = None
        
        for field in TOOGLE_FIELDS:
            if output_data[i].get(field) is not None:
                output_data[i][field] = not output_data[i][field]
        
        for _,value in SCORE_WEIGHTS.items():
            output_data[i][value["max_score_column"]] = 0
            output_data[i][value["score_column"]] = None
            for field in value["fields"]:
                if output_data[i][field] is not None:
                    output_data[i][value["max_score_column"]] = value["score"]
                    output_data[i][value["score_column"]] = value["score"] if output_data[i][value["score_column"]]!=0 and output_data[i][field] else 0
            if output_data[i][value["score_column"]] == None:
                output_data[i][value["score_column"]] = 0
        # for i in FATAL_FIELDS:
        output_data[i]["is_fatal"] = False
        for field in FATAL_FIELDS:
            if output_data[i].get(field)== False:
                output_data[i]["is_fatal"] = True
                break
        
        output_data[i]["total_score"] = 0
        output_data[i]["calculated_score"] = 0
        for j in SCORE_WEIGHTS_KEYS:
            output_data[i]["total_score"] += output_data[i][j[0]]
            output_data[i]["calculated_score"] += output_data[i][j[1]]


    return output_data
