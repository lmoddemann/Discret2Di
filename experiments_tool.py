import json


def make_grid():
    """
    creates a grid of experiments from the parameter lists batch_size, ...

    :return: experiments_dict
"""
   
    # In our experiments, we varied the parameters of temperature and beta.
    # The results were tested with a temperature between 0.1 and 1 with incremental gradations of 0.1 
    # With beta, the influence was also examined in the range between 0.1 and 1 with steps of 0.1.

    temperature = [0.5]
    beta = [0.1]
    experiments_dict = {}

    l = 0
    for i in temperature:
        for j in beta:
            experiments_dict[l] = {
                "IN_DIM": 3, # change in terms of dataset input dimension
                "ENC_OUT_DIM": 8,
                "DEC_OUT_DIM": 16, 
                "LATENT_DIM": 30,
                "CATEGORICAL_DIM": 10, 
                "TEMPERATURE": i, 
                "SOFTCLIP_MIN":0,
                "BATCH_SIZE": 128, 
                "BETA": j,
                "Experiment_ID": l
                }
            l +=1
    return experiments_dict

def experiments_to_json(experiment_dict: dict):
    """
    writing the experiment_dict into a .json file for documentation

    :param experiment_dict:
    :return: .json file containing the dictionary of hyperparameters
    """
    experiments = experiment_dict
    with open("../experiments_embed.json", "w") as json_file:
        json.dump(experiments, json_file, indent=4)
    print("experiments.json was created")
    return

def load_experiments(modus='run'):
    """
    reading the experiment .json file and returning a dictionary

    :param modus: (str) contains whether testing or training hyperparameters shall be used.
    :return: hparam <- dictionary with the content of the .json file
    """
    if "run" in modus:
        with open("../experiments_embed.json") as json_file:
            hparam = json.load(json_file)
    if "test" in modus:
        with open("testhparams.json") as json_file:
            hparam = json.load(json_file)
    return hparam


if __name__ == "__main__":
    experiments = make_grid()
    experiments_to_json(experiments)
    hparam = load_experiments(modus="run")
    print("funktioniert")