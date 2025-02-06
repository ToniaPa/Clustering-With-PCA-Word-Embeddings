import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from utils.exc_helper_functions import (transform_tfidf, transform_word2vec,
                                        transform_fasttext, evaluate_clustering, kmeans_inertia_ssc,
                                        get_word2vec_embeddings, get_fasttext_embeddings, find_optimal_pca)
from clustering_algorithms.exc_kmeans import KMeansClustering
from clustering_algorithms.exc_hdbscan import HDBSCANClustering
from clustering_algorithms.exc_agglomerative import AgglomerativeClusteringAlgorithm
from sklearn.decomposition import PCA
import os

os.environ['OMP_NUM_THREADS'] = '1'

# Get Datasets-
# 1.
# BBC news
filename = 'datasets/bbc_news_test.csv'
df_bbc = pd.read_csv(filename)
cached_df_bbc = df_bbc.copy()
df_bbc.columns = [col.lower() for col in df_bbc.columns.to_list()]
category_mapping = {
    'politics': 0,
    'entertainment': 1,
    'business': 2,
    'tech': 3,
    'sports': 4
}
df_bbc['label'] = df_bbc['category'].map(category_mapping)
median_value = df_bbc['label'].median()
df_bbc['label'] = df_bbc['label'].fillna(median_value)
df_bbc['label'] = df_bbc['label'].astype(int)

# 2.
# 20 newsgroups
# copied from teacher:
data_20newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents_20newsgroups = data_20newsgroups.data
labels_20newsgroups = data_20newsgroups.target
labels_names_20newsgroups = data_20newsgroups.target_names
news_dict = {
    'label': labels_20newsgroups,
    'comment': documents_20newsgroups,
}
df_20news = pd.DataFrame(news_dict)
df_20news['label_name'] = df_20news['label'].apply(lambda x: labels_names_20newsgroups[x])
cached_df_20news = df_20news.copy()
#
print('bbc =', df_bbc.shape, ', columns =', df_bbc.columns)
print('20news =', df_20news.shape, ', columns =', df_20news.columns)
## -end of Get Datasets

# Transform both Datasets-

# 1.
# Transformation, TF-IDF
print('Transformation TD-IDF: BBC news...')
bbc_tfidf = transform_tfidf(df_bbc['text']) #size 1490x24724
print(bbc_tfidf.shape)
print('Transformation TD-IDF: 20 newsgroups...')
news20_tfidf = transform_tfidf(df_20news['comment']) #18846x134388
print(news20_tfidf.shape)
# PCA
# ΠΟΛΥ ΜΕΓΑΛΑ Datasets ΓΙΑ TD-IDF: (1490, 24724) και (18846, 134388)
# PCA ώστε να γίνουν χ100, όπως τα υπόλοιπα (παρακάτω) για συγκρισιμότητα
print('PCA reduce to 100, for TF-IDF both Datasets...')
pca = PCA(n_components=100)  # number of features, τον βάζω ίδιο και για τα δύο για συγκρισιμότητα
pca_bbc_tfidf = pca.fit_transform(bbc_tfidf)
pca_news20_tfidf = pca.fit_transform(news20_tfidf)
# breakpoint()

# 2.
# Transformation, Word2vec
print('Transformation Word2vec: BBC news...')
bbc_w2v = transform_word2vec(df_bbc['text']) #--> PROBLEM, GIVES ERROR που δεν ξέρω να φτιάξω --> ΤΟ ΕΦΤΙΑΞΑ
# bbc_w2v = get_word2vec_embeddings(df_bbc['text'])
print(bbc_w2v.shape)
print('Transformation Word2vec: 20 newsgroups...')
news20_w2v = transform_word2vec(df_20news['comment']) #--> PROBLEM, GIVES ERROR που δεν ξέρω να φτιάξω --> ΤΟ ΕΦΤΙΑΞΑ
# news20_w2v = get_word2vec_embeddings(df_20news['comment'])
print(news20_w2v.shape)

# 3.
# Transformation, FastΤext
print('Transformation FastΤext: BBC news...')
bbc_ft = transform_fasttext(df_bbc['text']) #--> PROBLEM, GIVES ERROR που δεν ξέρω να φτιάξω --> ΤΟ ΕΦΤΙΑΞΑ
# bbc_ft = get_fasttext_embeddings(df_bbc['text'])
print(bbc_ft.shape)
print('Transformation FastΤext: 20 newsgroups...')
news20_ft = transform_fasttext(df_20news['comment']) #--> PROBLEM, GIVES ERROR που δεν ξέρω να φτιάξω --> ΤΟ ΕΦΤΙΑΞΑ
# news20_ft = get_fasttext_embeddings(df_20news['comment'])
print(news20_ft.shape)
print('All Transformations: ok')
## -end of Transform both Datasets

# KMeans evaluation: Inertia & Silhouette Scores-

# 1. KMeans evaluation, TF-IDF
print('KMeans evaluation TD-IDF: BBC news...')
kmeans_inertia_ssc(11, pca_bbc_tfidf, ' - BBC news TF-IDF',True) #PLOT: ΕΧΕΙ ΕΝΑ ΜΙΚΡΟ ΣΗΜΕΙΟ ΚΑΜΠΗΣ ΣΤΟ k=7
print('KMeans evaluation TD-IDF: 20 newsgroups...')
kmeans_inertia_ssc(25, pca_news20_tfidf, ' - 20 newsgroups TF-IDF',False) #PLOT: ΕΧΕΙ ΕΝΑ ΜΙΚΡΟ ΣΗΜΕΙΟ ΚΑΜΠΗΣ ΣΤΟ 20
# # Silhouette scores -> ΠΟΛΥ ΜΙΚΡΑ ΚΑΙ ΓΙΑ ΤΑ ΔΥΟ Datasets, ΣΤΟ 0.02 και κάτι έως 0.092
# # Inertia -> ΜΕΓΑΛΕΣ ΤΙΜΕΣ, 300 ΚΑΙ ΚΑΤΙ ΓΙΑ ΤΟ BBC & 1,600 έως 2.117 ΓΙΑ ΤΟ 20newgroups
# # Result = ο KMeans είναι χάλια για TF-IDF
#
# 2.KMeans evaluation, Word2vec
print('KMeans evaluation Word2vec: BBC news...')
kmeans_inertia_ssc(11, bbc_w2v, ' - BBC news Word2vec',False) #PLOT: ΟΜΟΙΩΣ ΔΕΝ ΕΧΕΙ ΕΜΦΑΝΕΣ ΣΗΜΕΙΟ ΚΑΜΠΗΣ...
print('KMeans evaluation Word2vec: 20 newsgroups...')
kmeans_inertia_ssc(25, news20_w2v, ' - 20 newsgroups Word2vec',False) #PLOT: ΕΧΕΙ ΕΝΑ ΠΟΛΥ ΜΙΚΡΟ ΣΗΜΕΙΟ ΚΑΜΠΗΣ ΣΤΟ 3
# # Silhouette scores -> ΠΟΛΥ ΜΙΚΡΑ ΚΑΙ ΓΙΑ ΤΑ ΔΥΟ Datasets, ΣΤΟ 0.22 ΓΙΑ ΤΟ BBC KAI ΚΑΤΩ & 0.07 ΓΙΑ ΤΟ 20newsgroups
# # Inertia -> ΜΕΓΑΛΕΣ ΤΙΜΕΣ, ΣΤΟ 100 ΚΑΙ ΠΑΝΩ ΓΙΑ ΤΟ BBC & 6.100 KAI ΠΑΝΩ ΓΙΑ ΤΟ 20 newgroups
# # Result = ο KMeans δεν είναι καλός για Word2vec (με την παρατήρηση ότι είναι καλύτερος από ό,τι για τον TF-IDF)
#
# #3.KMeans evaluation, FastText
print('KMeans evaluation FastText: BBC news...')
kmeans_inertia_ssc(11, bbc_ft, ' - BBC news FastText',False) #P#PLOT: ΕΧΕΙ ΕΝΑ ΠΟΛΥ ΜΙΚΡΟ ΣΗΜΕΙΟ ΚΑΜΠΗΣ ΣΤΟ 2
print('KMeans evaluation FastText: 20 newsgroups...')
kmeans_inertia_ssc(21, news20_ft, ' - 20 newsgroups FastText',False) #PLOT: ΟΜΟΙΩΣ ΔΕΝ ΕΧΕΙ ΕΜΦΑΝΕΣ ΣΗΜΕΙΟ ΚΑΜΠΗΣ...
# # Silhouette scores -> ΠΟΛΥ ΜΙΚΡΑ ΚΑΙ ΓΙΑ ΤΑ ΔΥΟ Datasets, ΣΤΟ 0.19 ΓΙΑ ΤΟ BBC KAI ΚΑΤΩ & 0.05 ΓΙΑ ΤΟ 20newsgroups
# # Inertia -> ΠΟΛΥ ΜΕΓΑΛΕΣ ΤΙΜΕΣ, ΣΤΟ 2.400 ΚΑΙ ΠΑΝΩ ΓΙΑ ΤΟ BBC & 44.100 KAI ΠΑΝΩ ΓΙΑ ΤΟ 20 newgroups
# # Result = ο KMeans δεν είναι καθόλου καλός για FastText
# print('All KMeans evaluation: ok')

# Result = KMeans είναι κακός για όλα, και πολύ κακός για το FastText
## -end of KMeans Inertia & Silhouette Scores

# HDBSCAN Clustering-

# 1. HDBSCAN, TF-IDF
print('HDBSCAN, TF-IDF...')
model_hdbscan = HDBSCANClustering()
bbc_tfidf_hdbscan_labels = model_hdbscan.fit_predict(pca_bbc_tfidf) #labels = -1 έως 2,
                                                                # -1 noisy data, άρα είναι 3
                                                                # δηλ λιγότερα από 5 (το BBC news έχει 5 κατηγορίες)
news20_tfidf_hdbscan_labels = model_hdbscan.fit_predict(pca_news20_tfidf) #labels = -1 έως 191 ΠΑΡΑ ΠΟΛΛΑ Labels
#
# 2. HDBSCAN, Word2vec
print('HDBSCAN, Word2vec...')
bbc_w2v_hdbscan_labels = model_hdbscan.fit_predict(bbc_w2v) # labels = -1 έως 1,
                                                            # -1 noisy data, άρα είναι 2
                                                            # δηλ λιγότερα από 5 (το BBC news έχει 5 κατηγορίες)
news20_w2v_hdbscan_labels = model_hdbscan.fit_predict(news20_w2v) #labels = -1 έως 11,
                                                            # δηλ λιγότερα από 20 (το 20newsgroups έχει 20 κατηγορίες)
#
# 3. HDBSCAN, FastText
print('HDBSCAN, FastText...')
bbc_ft_hdbscan_labels = model_hdbscan.fit_predict(bbc_ft) #labels = -1 έως 1,
                                                          # -1 noisy data, άρα είναι 2
                                                          # δηλ λιγότερα από 5 (το BBC news έχει 5 κατηγορίες)
news20_ft_hdbscan_labels = model_hdbscan.fit_predict(news20_ft) #labels = -1 έως 14,
                                                            # δηλ λιγότερα από 20 (το 20newsgroups έχει 20 κατηγορίες)

# HDBSCAN evaluation
# 1. HDBSCAN evaluation TD-IDF
nmi_bbc_tfidf_hdbscan, ari_bbc_tfidf_hdbscan, ami_bbc_tfidf_hdbscan = (
    evaluate_clustering(df_bbc['label'], bbc_tfidf_hdbscan_labels))
print('HDBSCAN evaluation, TF-IDF for BBC news:' + '\n'+f'NMI = {nmi_bbc_tfidf_hdbscan}, ARI = {ari_bbc_tfidf_hdbscan}, '
                                          f'AMI = {ami_bbc_tfidf_hdbscan}')
nmi_20news_tfidf_hdbscan, ari_news20_tfidf_hdbscan, ami_news20_tfidf_hdbscan = (
    evaluate_clustering(df_20news['label'], news20_tfidf_hdbscan_labels))
print('HDBSCAN evaluation, TF-IDF for 20newsgroups:' + '\n'+f'NMI = {nmi_20news_tfidf_hdbscan}, ARI = {ari_news20_tfidf_hdbscan}, '
                                          f'AMI = {ami_news20_tfidf_hdbscan}')

# 2.HDBSCAN evaluation Word2vec
# print('HDBSCAN evaluation, Word2vec...')
nmi_bbc_w2v_hdbscan, ari_bbc_w2v_hdbscan, ami_bbc_w2v_hdbscan = (
    evaluate_clustering(df_bbc['label'], bbc_w2v_hdbscan_labels))
print('HDBSCAN evaluation, Word2vec for BBC news:' + '\n'+f'NMI = {nmi_bbc_w2v_hdbscan}, ARI = {ari_bbc_w2v_hdbscan}, '
                                          f'AMI = {ami_bbc_w2v_hdbscan}')
nmi_20news_w2v_hdbscan, ari_news20_w2v_hdbscan, ami_news20_w2v_hdbscan = (
    evaluate_clustering(df_20news['label'], news20_w2v_hdbscan_labels))
print('HDBSCAN evaluation, Word2vec for 20newsgroups:' + '\n'+f'NMI = {nmi_20news_w2v_hdbscan}, ARI = {ari_news20_w2v_hdbscan}, '
                                          f'AMI = {ami_news20_w2v_hdbscan}')
# print('HDBSCAN evaluation, Word2vec for 20newsgroups:', 'UNSOLVED ERROR -> ΔΕΣ exc_helper_functions ΣΧΟΛΙΑ transform_word2vec (ΓΡΑΜΜΗ 52) ')

# 3. HDBSCAN evaluation FastText
nmi_bbc_ft_hdbscan, ari_bbc_ft_hdbscan, ami_bbc_ft_hdbscan = (
    evaluate_clustering(df_bbc['label'], bbc_ft_hdbscan_labels))
print('HDBSCAN evaluation, FastText for BBC news:' + '\n'+f'NMI = {nmi_bbc_ft_hdbscan}, ARI = {ari_bbc_ft_hdbscan}, '
                                          f'AMI = {ami_bbc_ft_hdbscan}')
nmi_20news_w2v_hdbscan, ari_news20_w2v_hdbscan, ami_news20_w2v_hdbscan = (
    evaluate_clustering(df_20news['label'], news20_w2v_hdbscan_labels))
nmi_news20_ft_hdbscan, ari_news20_ft_hdbscan, ami_news20_ft_hdbscan = (
    evaluate_clustering(df_20news['label'], news20_ft_hdbscan_labels))
print('HDBSCAN evaluation, FastText for 20newsgroups:' + '\n'+f'NMI = {nmi_news20_ft_hdbscan}, ARI = {ari_news20_ft_hdbscan}, '
                                          f'AMI = {ami_news20_ft_hdbscan}')
# print('HDBSCAN evaluation, FastText for 20newsgroups:', 'UNSOLVED ERROR -> ΔΕΣ exc_helper_functions ΣΧΟΛΙΑ transform_fasttext (ΓΡΑΜΜΗ 95) ')

print('All HDBSCAN Clustering: ok')
## -end of HDBSCAN Clustering

# breakpoint()

# Agglomerative Clustering-
# model_agglo = AgglomerativeClusteringAlgorithm(n_clusters=)
model_agglo11 = AgglomerativeClusteringAlgorithm(n_clusters=11) #11 clusters για ομοιότητα με Kmeans
model_agglo25 = AgglomerativeClusteringAlgorithm(n_clusters=25) #25 clusters για ομοιότητα με Kmeans

# 1. Agglomerative, TF-IDF
print('Agglomerative, TF-IDF...ERROR: Sparse data...ERROR------------------------')
# ypeError: Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array.
# bbc_tfidf.toarray() #FAILED
# bbc_tfidf.todense() #FAILED
# bbc_tfidf_agglo_labels = model_agglo11.fit_predict(bbc_tfidf) # labels = -1 έως 1, είναι 3
#                                                             # δηλ λιγότερα από 5 (το BBC news έχει 5 κατηγορίες)
# news20_tfidf_agglo_labels = model_agglo25.fit_predict(news20_tfidf) #labels = -1 έως 11,
#                                                             # δηλ λιγότερα από 20 (το 20newsgroups έχει 20 κατηγορίες)

# 2. Agglomerative, Word2vec
print('Agglomerative, Word2vec...')
bbc_w2v_agglo_labels = model_agglo11.fit_predict(bbc_w2v) # labels = 0 έως 10
news20_w2v_agglo_labels = model_agglo25.fit_predict(news20_w2v) #labels = 0 έως 24

# 3. Agglomerative, FastText
print('Agglomerative, FastText...')
bbc_ft_agglo_labels = model_agglo11.fit_predict(bbc_ft) #labels = 0 έως 10
news20_ft_agglo_labels = model_agglo25.fit_predict(news20_ft) #labels = 01 έως 24

# breakpoint()

# Agglomerative evaluation
# 1. Agglomerative evaluation TF-IDF
print('Agglomerative evaluation, TF-IDF for BBC & 20newsgroups: ERROR...Sparse data...ERROR------------')
# nmi_bbc_tfidf_agglo, ari_bbc_tfidf_agglo, ami_bbc_tfidf_agglo = (
#     evaluate_clustering(df_bbc['label'], bbc_tfidf_agglo_labels))
# print('Agglomerative evaluation, TF-IDF for BBC news:' + '\n'+f'NMI = {nmi_bbc_tfidf_agglo}, ARI = {ari_bbc_tfidf_agglo}, '
#                                           f'AMI = {ami_bbc_tfidf_agglo}')
# nmi_20news_tfidf_agglo, ari_news20_tfidf_agglo, ami_news20_tfidf_agglo = (
#     evaluate_clustering(df_20news['label'], news20_tfidf_agglo_labels))
# print('Agglomerative evaluation, TF-IDF for 20newsgroups:' + '\n'+f'NMI = {nmi_20news_tfidf_agglo}, ARI = {ari_news20_tfidf_agglo}, '
#                                           f'AMI = {ami_news20_tfidf_agglo}')

# 2. Agglomerative evaluation Word2vec
nmi_bbc_w2v_agglo, ari_bbc_w2v_agglo, ami_bbc_w2v_agglo = (
    evaluate_clustering(df_bbc['label'], bbc_w2v_agglo_labels))
print('Agglomerative evaluation, Word2vec for BBC news:' + '\n'+f'NMI = {nmi_bbc_w2v_agglo}, ARI = {ari_bbc_w2v_agglo}, '
                                          f'AMI = {ami_bbc_w2v_agglo}')
nmi_20news_w2v_agglo, ari_news20_w2v_agglo, ami_news20_w2v_agglo = (
    evaluate_clustering(df_20news['label'], news20_w2v_agglo_labels))
print('Agglomerative evaluation, Word2vec for 20newsgroups:' + '\n'+f'NMI = {nmi_20news_w2v_agglo}, ARI = {ari_news20_w2v_agglo}, '
                                          f'AMI = {ami_news20_w2v_agglo}')
# print('Agglomerative evaluation, Word2vec for 20newsgroups:', 'UNSOLVED ERROR -> ΔΕΣ exc_helper_functions ΣΧΟΛΙΑ transform_word2vec (ΓΡΑΜΜΗ 52) ')

# 3.Agglomerative evaluation FastText
nmi_bbc_ft_agglo, ari_bbc_ft_agglo, ami_bbc_ft_agglo = (
    evaluate_clustering(df_bbc['label'], bbc_ft_agglo_labels))
print('Agglomerative evaluation, FastText for BBC news:' + '\n'+f'NMI = {nmi_bbc_ft_agglo}, ARI = {ari_bbc_ft_agglo}, '
                                          f'AMI = {ami_bbc_ft_agglo}')
nmi_20news_ft_agglo, ari_news20_ft_agglo, ami_news20_ft_agglo = (
    evaluate_clustering(df_20news['label'], news20_ft_agglo_labels))
print('Agglomerative evaluation, FastTexxt for 20newsgroups:' + '\n'+f'NMI = {nmi_20news_ft_agglo}, ARI = {ari_news20_ft_agglo}, '
                                          f'AMI = {ami_news20_ft_agglo}')
# print('Agglomerative evaluation, FastText for 20newsgroups:', 'UNSOLVED ERROR -> ΔΕΣ exc_helper_functions ΣΧΟΛΙΑ transform_fasttext (ΓΡΑΜΜΗ 95) ')

print('All Agglomerative Clustering: ok')

## -end of Agglomerative Clustering

# Results for Clustering evaluation based on AMI, NMI, ARI values:
# 1. HDBSCAN:
#     αριθμός clusters που δίνει
#       TF-IDF 3 για BBC KAI 190 για 20newsgroups !!!
#       Word2vec 2 για BBC & 12 για 20newsgroups
#       FastText 2 για BBC & 14 για 20newsgroups
# NMI is very close to 0 for both Datasets -> no mutual information
# ARI is even closer to 0 than NMI for both Datasets -> neutral similarity
# AMI is very close to 0 for both Datasets -> neutral information

# 2. Agglomerative:
#     έχω δηλώσει n_clusters=11 for BBC & n_clusters = 25 for 20newsgroups
# ---NO RESULTS FOR TF-IDF BECAUSE OF ERROR (Sparse matrix that failed to be dense)---
# NMI is 0.499 for Word2vec of BBC news -> medium agreement
# NMI is 0.3 for FastText of BBC news -> weak agreement
# NMI is close to 0 for all others -> no mutual information
#
# ARI is 0.316 for Word2vec of BBC news -> weak agreement
# ARI is 0.17 for FastText of BBC news -> very weak agreement
# ARI is close to 0 for all others -> neutral similarity
#
# AMI is 0.497 for Word2vec of BBC news -> medium agreement
# AMI is 0.3 for FastText of BBC news -> weak agreement
# AMI is 0.17 for Word2vec of 20newsgroups -> very weak agreement
# AMI is very close to 0 for all others -> neutral information
#
# 3. KMeans
#    έχω δηλώσει nclusters range 1-11 for BBC & 1-25 for 20 newsgroups
# KMeans είναι κακός για όλα, και πολύ κακός για το FastText ιδιαίτερα
#
# Conlusion:
# Winners are Agglomerative Clustering:
#      a. BBC news     -> Word2vec (medium), second best FastText (weak)
#      b. 20newsgroups -> Word2vec (very weak)


# Άρα: PCA fine-tuning for Word2vec με KMeans και Agglomerative Clustering
# Έχουμε bbc_w2v    (1490, 100)  -> BBC news
#        news20_w2v (18846, 100) -> 20newgroups

print('PCA Fine-tuning for both Datasets...')
best_kmeans_bbc, best_agglo_bbc = find_optimal_pca(bbc_w2v,3, 11, df_bbc['label'])
best_kmeans_news20, best_agglo_news20 = find_optimal_pca(news20_w2v,3, 25, df_20news['label'])

print('BBC, KMeans=', best_kmeans_bbc, ', Agglo=',best_agglo_bbc)
print('20newsgroups, KMeans=', best_kmeans_news20, ', Agglo=',best_agglo_news20)
# Results = χάλια μου φαίνονται γιατί είναι πολύ πάνω από το 1....αλλά οκ δεν είμαι ειδική
# δεν ξέρω τι σημαίνει και περι τίνος πρόκειται

# t-sne
# 1. BBC news
print('t-SNE for BBC news...')
pca = PCA(n_components=2)
reduced_bbc = pca.fit_transform(bbc_w2v)
# t-sne
tsne = TSNE(n_components=2, random_state=0)
tsne_bbc = tsne.fit_transform(reduced_bbc)
# scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(tsne_bbc[:, 0], tsne_bbc[:, 1], c=df_bbc['label'], cmap='viridis', alpha=0.6)
plt.title(f't-SNE Visualization of Clustering (PCA components: 2) - BBC news')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='True Labels')
plt.show()

# 1. 20newsgroups
print('t-SNE for 20newsgroups...')
pca = PCA(n_components=2)
reduced_20news = pca.fit_transform(news20_w2v)
# t-sne
tsne = TSNE(n_components=2, random_state=0)
tsne_20news = tsne.fit_transform(reduced_20news)
# scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(tsne_20news[:, 0], tsne_20news[:, 1], c=df_20news['label'], cmap='viridis', alpha=0.6)
plt.title(f't-SNE Visualization of Clustering (PCA components: 2) - 20newsgroups')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='True Labels')
plt.show()



