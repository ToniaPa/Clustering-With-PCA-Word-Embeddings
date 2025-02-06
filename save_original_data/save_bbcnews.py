"""
Module for saving original data of BBC news to a folder
"""

import os
import pandas as pd

#Fetch the dataset
filename = '../datasets/bbc_news_test.csv'
df_bbc = pd.read_csv(filename)

#Create a directory to save the dataset
save_directory = 'bbc_news'
os.makedirs(save_directory, exist_ok=True)

#Save each document to a text file
for idx, row in df_bbc.iterrows():
    #file name based on index
    file_name = os.path.join(save_directory, f'document_bbc_{idx}.txt')
    doc = 'Id: ' + str(row['ArticleId']) + '\n'+ row['Text'] + '\nCategory: ' + row['Category']
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(doc)

print(f'Documents of BBC news saved to {save_directory} directory.')