import pandas as pd 
import matplotlib.pyplot as plt

db = pd.read_csv("dataset.csv")

animal_phyla = [
    'Chordata_7711',
    'Echinodermata_7586',
    'Hemichordata_10219',
    'Arthropoda_6656',
    'Onychophora_27563',
    'Tardigrada_42241',
    'Nematoda_6231',
    'Nematomorpha_33310',
    'Kinorhyncha_51516',
    'Priapulida_33467',
    'phylum_Micrognathozoa_195505',
    'Chaetognatha_10229',
    'Gnathostomulida_66780',
    'Rotifera_10190',
    'Dicyemida_10215',
    'Orthonectida_33209',
    'Acanthocephala_10232',
    'Annelida_6340',
    'Brachiopoda_7568',
    'Bryozoa_10205',
    'Cycliophora_69815',
    'Entoprocta_43120',
    'Gastrotricha_33313',
    'Mollusca_6447',
    'Nemertea_6217',
    'Phoronida_120557',
    'Platyhelminthes_6157',
    'Xenacoelomorpha_1312402',
    'Cnidaria_6073',
    'Ctenophora_10197',
    'Placozoa_10226',
    'Porifera_6040'
]

dataset_animalia = db[db['Phylum'].isin(animal_phyla)]
dataset_rest = db[~db['Phylum'].isin(animal_phyla)]

dataset_animalia.to_csv('final_dataset.csv', index=False)
