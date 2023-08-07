from discret2di_utils import load_model, generate_discretizatons, assosciation_rule_mining, create_dict, save_files, check_truth
from preprocessing_rules import getHealthStates
from diagnoser import Diag_solver
import json 
from translate import parse_file
import pandas as pd

def check_normal(version, min_threshold, min_support, anomaly, threshold_likelihood, dict_path, rule_path):
    """ Load the learned model of the CatVAE and generate discretizations and their according likelihoods based on the nominal system behavior. 
      Based on the discretizations, frequent itemsets can be found and formulated as rules with a dummy as non-observable components.
      """
    # load CatVAE model and its hyperparameter
    model, hparam = load_model(version=version)

    # generate discretizations, their accoridng likelihoods and mu/sigma etc. 
    states, resulting_states, unique_disc, data_label, df_mu, df_sigma, cats_clean, likelihood_median, disc_likelihood, likelihood = generate_discretizatons(model=model, 
                                                                                                                            hparam=hparam,
                                                                                                                            anomaly=anomaly, 
                                                                                                                            threshold=threshold_likelihood)
    # calculation of the minimum support based on https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
    min_supp = (resulting_states[resulting_states.columns[0]].value_counts().min()/len(resulting_states))*min_support
    true_rules, rules = assosciation_rule_mining(states, unique_disc, min_threshold, min_supp)
    # create dictionary of categories to discretized likelihoods
    dict_states_char = create_dict(true_rules=true_rules)
    # dummy for non-observable components
    dummy_predicates = 'dummy'
    # saving rules at path "diagnosis"
    save_files(true_rules=true_rules, dict_states_char=dict_states_char, dummy=dummy_predicates, dict_path=dict_path, rule_path=rule_path)
    # check resulting states, if included in dictionary or whether category not included within rule base
    resulting_states = check_truth(resulting_states=resulting_states, dict_states_char=dict_states_char)
    anomaly_df1 = pd.concat([data_label, resulting_states], axis =1)
    return anomaly_df1, likelihood


def check_diagnosis(version, min_threshold, min_support, anomaly, threshold_likelihood, dict_path, rule_path):
    """ Load the learned model of the CatVAE and generate discretizations and their according likelihoods based on the anomalous system behavior. 
      Based on the discretizations and the likelihoods (discretized by thresholds), anomalies can be detected.
      """
    # load CatVAE model and its hyperparameter
    model, hparam = load_model(version=version)
    # generate discretizations, their accoridng likelihoods and mu/sigma etc. 
    states, resulting_states, unique_disc, data_label, df_mu, df_sigma, cats_clean, likelihood_median, disc_likelihood, likelihood = generate_discretizatons(model=model, 
                                                                                                                            hparam=hparam,
                                                                                                                            anomaly=anomaly, 
                                                                                                                            threshold=threshold_likelihood)
    # load dictionary
    dict_states_char = json.load(open(f"diagnosis/{dict_path}.txt"))
    # check resulting states, if included in dictionary or whether category not included within rule base
    resulting_states = check_truth(resulting_states=resulting_states, dict_states_char=dict_states_char)
    anomaly_df = pd.concat([data_label, resulting_states], axis =1)
    return anomaly_df, likelihood


def diagnose_system(rule_path, anom_df):
    """ Diagnosing the system based on the anomalous dataframe. Rules are parsed and the dataframe is checked on inconsistencies.
        The Diagnosis will output all causes and also minimal causes as well as the processing time. 
    """
    anomaly_obs = []
    diag = []
    anom_df["diag"] = ""
    # parse learned rule base
    rules = parse_file(f'diagnosis/{rule_path}.txt', is_sympy=False)
    # loop over the time series data and compute according diagnosis 
    for i in range(int(anom_df.index[0]), len(anom_df)+int(anom_df.index[0])):
        if anom_df.loc[i, "results"] == False:
            anomaly_obs = anom_df.loc[i, 'cats']
            # seet a observational state (category) to False in case of anomaly and create therefore a inconsistency within the rule base
            faultStates = {anomaly_obs: False}
            # update the rule base
            rules_healthStates = getHealthStates(rules=rules, faultStates=faultStates)
            # create diagnosis solver based on rule base and inconsisitent states
            diag_model = Diag_solver(rules=rules, health_dict=rules_healthStates)
            diag, min_causes, causes, delta_time = diag_model.solve()
            anom_df.loc[i, "diag"] = f"{diag}"

        else:
            anom_df.loc[i, "diag"] = "None"

    return anom_df




