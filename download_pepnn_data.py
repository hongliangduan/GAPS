import pandas as pd
import tqdm
import wget
import os

df_pepnn_data = pd.read_csv('util/pepnn_data.txt', header=None)
df_pepnn_data.columns = ['PDB']
df_pepnn_data = df_pepnn_data['PDB'].apply(lambda x: x[:4])

def mkdir(path):
    folder_exist = os.path.exists(path)
    if not folder_exist:
        os.makedirs(path)
    else:
        pass

def download_pdb(path, df):
    error = []
    mkdir(path)
    for i in tqdm.tqdm(df):
        try:
            url = f'https://files.rcsb.org/download/{i}.pdb1.gz'
            wget.download(url, path)
        except:
            error.append(i)
            print('error:'+i)
    print(error)

download_pdb('data/pepnn', df_pepnn_data)