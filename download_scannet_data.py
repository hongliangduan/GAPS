import pandas as pd
import tqdm
import wget
import os

df_scannet_data = pd.read_csv('./util/scannet_data.csv')
df_scannet_data = df_scannet_data.drop(df_scannet_data.columns[0], axis=1)
df_scannet_data = df_scannet_data.drop_duplicates('PDB ID')

df_pdbid = df_scannet_data['PDB ID']
df_pdbid = df_pdbid.apply(lambda x: x.split('_')[0])
pdbid = set(df_pdbid)

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

download_pdb('data/scannet', pdbid)