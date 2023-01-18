import os
from hypertension_treatment_policy_evaluation import results_dir
from hypertension_treatment_policy_evaluation import hypertension_treatment_policy_evaluation_function

if __name__ == '__main__':
    pt_sim_aha = hypertension_treatment_policy_evaluation_function("aha")   #-# Run policy evaluation using AHA guidelines
    pt_sim_esc = hypertension_treatment_policy_evaluation_function("esc")   #-# Run policy evaluation using ESC guidelines
    pt_sim_jnc7 = hypertension_treatment_policy_evaluation_function("jnc7") #-# Run policy evaluation using JNC7 guidelines
    pt_sim_jnc8 = hypertension_treatment_policy_evaluation_function("jnc8") #-# Run policy evaluation using JNC8 guidelines

    pt_sim_all_treatments = pt_sim_aha.merge(pt_sim_esc).merge(pt_sim_jnc7).merge(pt_sim_jnc8) #-# Merge all pt_data df's into one large dataframe

    os.chdir(results_dir)

    pt_sim_all_treatments.to_pickle('pt_sim_all_treatments.pkl') #-# Save dataframe of all guideline results as a pickle file
    # To unpack pickle file use:
    #     os.chdir(results_dir)
    #     pt_sim_all_treatments = pd.read_pickle("pt_sim_all_treatments.pkl")