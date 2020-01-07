### script for writing meta information of datasets into master.csv
### for graph property prediction datasets.
import pandas as pd

dataset_list = []
dataset_dict = {}

### start molecule dataset
dataset_dict["ogbg-mol-bace"] = {"num tasks": 1, "task type": "binary classification", "download_name": "bace"}
dataset_dict["ogbg-mol-bbbp"] = {"num tasks": 1, "task type": "binary classification", "download_name": "bbbp"}
dataset_dict["ogbg-mol-clintox"] = {"num tasks": 2, "task type": "binary classification", "download_name": "clintox"}
dataset_dict["ogbg-mol-muv"] = {"num tasks": 17, "task type": "binary classification", "download_name": "muv"}
dataset_dict["ogbg-mol-pcba"] = {"num tasks": 128, "task type": "binary classification", "download_name": "pcba"}
dataset_dict["ogbg-mol-sider"] = {"num tasks": 27, "task type": "binary classification", "download_name": "sider"}
dataset_dict["ogbg-mol-tox21"] = {"num tasks": 12, "task type": "binary classification", "download_name": "tox21"}
dataset_dict["ogbg-mol-toxcast"] = {"num tasks": 617, "task type": "binary classification", "download_name": "toxcast"}
dataset_dict["ogbg-mol-hiv"] = {"num tasks": 1, "task type": "binary classification", "download_name": "hiv"}
dataset_dict["ogbg-mol-esol"] = {"num tasks": 1, "task type": "regression", "download_name": "esol"}
dataset_dict["ogbg-mol-freesolv"] = {"num tasks": 1, "task type": "regression", "download_name": "freesolv"}
dataset_dict["ogbg-mol-lipo"] = {"num tasks": 1, "task type": "regression", "download_name": "lipophilicity"}

mol_dataset_list = list(dataset_dict.keys())

for nme in mol_dataset_list:
    download_folder_name = dataset_dict[nme]["download_name"]
    dataset_dict[nme]["pyg url"] = "https://ogb.stanford.edu/data/graphproppred/pyg_mol_download/" + download_folder_name + ".zip"
    dataset_dict[nme]["dgl url"] = "https://ogb.stanford.edu/data/graphproppred/dgl_mol_download/" + download_folder_name + ".zip"
    dataset_dict[nme]["url"] = "https://ogb.stanford.edu/data/graphproppred/csv_mol_download/" + download_folder_name + ".zip"
    dataset_dict[nme]["add_inverse_edge"] = True
    dataset_dict[nme]["data type"] = "mol"
    dataset_dict[nme]["split"] = "scaffold"

dataset_list.extend(mol_dataset_list)

### end molecule dataset

df = pd.DataFrame(dataset_dict)
# saving the dataframe 
df.to_csv("master.csv")