"""
Module for saving original data of 20 newsgroyps to a folder
"""

import os
from sklearn.datasets import fetch_20newsgroups

#Fetch the dataset
save_20newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

#Create a directory to save the dataset
save_directory = '20_newsgroups'
os.makedirs(save_directory, exist_ok=True)

#Save each document to a text file
for idx, doc in enumerate(save_20newsgroups.data):
    #file name based on index
    file_name = os.path.join(save_directory, f'document_20news_{idx}.txt')
    target = save_20newsgroups.target[idx]
    doc = (doc + '\n\n---Target:' + str(target) +
           ' (' + save_20newsgroups.target_names[target] + ')')
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(doc)

print(f'Documents of 20 newsgroups saved to {save_directory} directory.')