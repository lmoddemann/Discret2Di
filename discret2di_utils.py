import pandas as pd 
import numpy as np
import yaml 
import torch
import json
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns.fpgrowth import fpgrowth
from mlxtend.frequent_patterns import association_rules

from utils import standardize_data
from catvae_tank import CategoricalVAE as StateVAE
from datamodule import Dataset
from check_truth import check_true_cats


def load_model(version):
    """ Loading the trained CatVAE from the location "CatVAE_training_hparams" using the identifier "version. """

    MODEL_VERSION = f'CatVAE_training_hparams/{version}'
    ckpt_file_name = os.listdir(f'./{MODEL_VERSION}/checkpoints/')[-1]
    ckpt_file_path = f'./{MODEL_VERSION}/checkpoints/{ckpt_file_name}'
    with open(f'./{MODEL_VERSION}/hparams.yaml') as f:
        hparam = yaml.safe_load(f)
    model = StateVAE.load_from_checkpoint(ckpt_file_path, hparams=hparam["hparams"]).to('cuda')
    
    return model, hparam


def generate_discretizatons(model, hparam, anomaly, threshold):
    """ Generate discretizations of the differnt datasets (nominal/anomalous system behavior). 
    In doing so, we discard the first 1000 samples in order to concentrate on the steady state."""

    if anomaly == 'Tank_normal':
        df = pd.read_csv('preprocessed_data/tank_simulation/norm.csv').reset_index(drop=True)[1000:]

    elif anomaly == 'Tank_q1fault':
        df = pd.read_csv('preprocessed_data/tank_simulation/q1_faulty.csv').reset_index(drop=True)[1000:]

    elif anomaly == 'Tank_v12fault':
        df = pd.read_csv('preprocessed_data/tank_simulation/v12_faulty.csv').reset_index(drop=True)[1000:]

    elif anomaly == 'Tank_v23fault':
        df = pd.read_csv('preprocessed_data/tank_simulation/v23_faulty.csv').reset_index(drop=True)[1000:]

    elif anomaly == 'Tank_v3fault':
        df = pd.read_csv('preprocessed_data/tank_simulation/v3_faulty.csv').reset_index(drop=True)[1000:]

    elif anomaly == 'Tank_q1fault_3cycles':
        df = pd.read_csv('preprocessed_data/tank_simulation/q1_faulty_3cycles.csv').reset_index(drop=True)[1000:]

    elif anomaly == 'Tank_v12fault_3cycles':
        df = pd.read_csv('preprocessed_data/tank_simulation/v12_faulty_3cycles.csv').reset_index(drop=True)[1000:]

    elif anomaly == 'Tank_v23fault_3cycles':
        df = pd.read_csv('preprocessed_data/tank_simulation/v23_faulty_3cycles.csv').reset_index(drop=True)[1000:]

    elif anomaly == 'Tank_v3fault_3cycles':
        df = pd.read_csv('preprocessed_data/tank_simulation/v3_faulty_3cycles.csv').reset_index(drop=True)[1000:]
    else: 
        df = 0
        raise TypeError(f'False input for anomaly. Try either "Tank_normal", "Tank_q1fault", "Tank_v12fault", "Tank_v23fault", "Tank_v3fault", "Tank_q1fault_3cycles", "Tank_v12fault_3cycles", "Tank_v23fault_3cycles" or "Tank_v3fault_3cycles"')

    df_sc = standardize_data(df, 'scaler_tank.pkl')
    data = Dataset(dataframe = df_sc)[:][0:]
    
    # Generate likelihoods for data
    likelihood = pd.DataFrame(model.function_likelihood(torch.tensor(data).to(device='cuda')).cpu().detach(), index=df.index).rolling(10).median().fillna(method='bfill')
    likelihood_median = pd.DataFrame(np.array(likelihood.values, dtype='f')).set_index(df.index)

    # Generate Discretization of states
    pzx_logits, pzx, mu_cat, sigma_cat, pxz, z = model.get_states(torch.tensor(data).to(device='cuda'))
    df_sigma = pd.DataFrame(torch.diagonal(sigma_cat.cpu().detach(), dim1=1, dim2=2), index=df.index)
    df_mu = pd.DataFrame(mu_cat.cpu().detach(), index=df.index)
    # Generate one hot states by using argmax()
    df_states = pd.DataFrame(torch.zeros(z.shape).to(device='cuda').scatter(1, torch.argmax(pzx_logits, dim=1).unsqueeze(1), 1).cpu().detach().numpy(), index=pd.DataFrame(df).index).astype(int)
    cats = pd.DataFrame(df_states.idxmax(axis=1), index=df.index)
    df_states_index = df_states.set_index(pd.Index(cats[cats.columns[0]]))
    unique_states = df_states.set_index(pd.Index(cats[cats.columns[0]])).drop_duplicates()
    # check whether categories are included within the rule base
    dict_truth_cats = check_true_cats(model, hparam, data, tolerance=0.001)

    if len(dict_truth_cats)!=0:
        idx_lst = []
        for key, value in dict_truth_cats.items():
                idx_lst = pd.Series([value if e==int(key) else e for e in df_states_index.index])
        df_states = pd.concat([pd.DataFrame(unique_states.loc[pd.Index([i])]) for i in idx_lst])
    disc_states = df_states.astype(str).agg(''.join, axis=1)
    unique_disc = list(disc_states.unique())
    cats_clean = pd.DataFrame(df_states.idxmax(axis=1))

    # Discretization of the likelihood
    disc_likelihood = pd.DataFrame(np.where(likelihood_median < threshold, '1', '0'), columns=["likelihood"]).set_index(disc_states.index)
    states = np.array(pd.concat([disc_states, disc_likelihood], axis=1))
    disc_char = disc_states.str.replace('0', 'a').str.replace('1', 'b')
    resulting_states = pd.concat([disc_char, disc_likelihood], axis=1).rename(columns={0:'cats'})

    return states, resulting_states, unique_disc, df, df_mu, df_sigma, cats_clean, likelihood_median, disc_likelihood, likelihood


def assosciation_rule_mining(states, unique_disc, min_threshold, min_support):
    """ Association Rule mining from discretized "states" with FPGrowth."""
    # Transform the dataset into an array format suitable 
    states_encoder = TransactionEncoder()
    states_encoder.fit(states)
    encoded_states = states_encoder.transform(states)
    df_encoded_states = pd.DataFrame(encoded_states, columns=states_encoder.columns_)
    # Learning of frequent itemsets
    frequent_itemsets = fpgrowth(df_encoded_states, min_support=min_support, use_colnames=True)
    # Generate association rules from frequent itemsets
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_threshold)
    # Filter out the rules by its implicants (unique discretizations)
    true_rules_bool = pd.DataFrame([rules[rules.columns[0]] == frozenset({f'{comp}'}) for comp in unique_disc]).T.any(axis=1)
    true_rules = rules[true_rules_bool.values]
    return true_rules, rules


def create_dict(true_rules):
    """ create dictionary of categories to its discretized likelihoods"""
    states_disc = list(true_rules[true_rules.columns[0]].apply(lambda x: ', '.join(list(x))).astype("unicode"))
    res_disc = list(true_rules[true_rules.columns[1]].apply(lambda x: ', '.join(list(x))).astype("unicode"))
    df_disc = pd.DataFrame({"states_disc":states_disc, "res_disc":res_disc})
    states_dict = df_disc.groupby('states_disc')['res_disc'].apply(list).to_dict()
    for state in list(states_dict.keys()): 
        dict_states_char = {key.replace('0', 'a').replace('1', 'b'): value for key, value in states_dict.items()}
    return dict_states_char


def save_files(true_rules, dict_states_char, dummy, dict_path, rule_path):
    """ saving rules at path "diagnosis" """
    json.dump(dict_states_char, open(f"diagnosis/{dict_path}.txt",'w'))
    with open(f'diagnosis/{rule_path}.txt', 'w') as f:
        [f.writelines([f"dummy IMPLIES {key}\n" for (key, value) in dict_states_char.items()])]
    return print("saved")


def check_truth(resulting_states, dict_states_char):
    """ check resulting states, if included in dictionary or whether category not included within rule base"""
    true_vals = []
    not_found = []
    for result_states0, result_states1 in zip(resulting_states[resulting_states.columns[0]], resulting_states[resulting_states.columns[1]]):
        if dict_states_char.get(result_states0) is None:
            not_found.append(result_states0)
            res = False 
            true_vals.append(res)
            continue
        else:
            res = result_states1 in dict_states_char.get(result_states0)
            true_vals.append(res)
    resulting_states = pd.concat([resulting_states, pd.DataFrame(true_vals, index=resulting_states.index, columns=['results'])], axis=1)
    return resulting_states

