"""
Helper functions for implementing NLP models: TF-IDF, Word2Vec & FastText.
Downloads package punkt_tab (because copied functions from teacher are giving ERROR I cannot fix)
"""
import nltk
import numpy as np
from gensim.models import Word2Vec, FastText
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, \
    silhouette_score
from Exercises_to_give.Exercise_11_UL.clustering_algorithms.exc_kmeans import KMeansClustering
from Exercises_to_give.Exercise_11_UL.clustering_algorithms.exc_agglomerative import AgglomerativeClusteringAlgorithm
from sklearn.decomposition import PCA

# nltk.download('punkt_tab') #--> ΛΥΘΗΚΕ ΤΟ ΠΡΟΒΛΗΜΑ ΜΕ ΤΟ ValueError: setting an array element with a sequence.

def transform_tfidf(data):
    """
        Converts text data into TF-IDF vectors.

        Args:
            data (array): Array of strings.

        Returns:
            TF-IDF probability matrix.
        """
    custom_stop_words = ['i', 'a', 'an', 'and', 'but','or','in','on','at','with','he','she','it',
                         'they','is', 'am', 'are', 'was', 'were', 'be', 'being', 'been','of','when'
                    ]
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix

def get_word2vec_embeddings(data, mo=1):
    # Κώδικας από το Google
    """
            Converts text data into Word2Vec vectors. Κώδικας από το Google. Use of nltk library. Package punkt_tab is downloaded.

            Args:
                data (array): Array of strings.
                mo (int): Word2Vec model (0 for CBOW, 1 for Skip-Gram Model)

            Returns:
                np.array: Array of vectors.
            """

    tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in data]
    model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, sg=mo)
    embeddings = np.array([np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
                            for doc in tokenized_docs if doc])
    return embeddings

def transform_word2vec(data, mo=1):
    # PROBLEM, GIVES ERROR:
    # ValueError: setting an array element with a sequence. The requested array has an
    # inhomogeneous shape after 1 dimensions. The detected shape was (18846,) + inhomogeneous
    # part.
    # ΔΕΝ ΞΕΡΩ ΠΩΣ ΛΥΝΕΤΑΙ
    """
        Converts text data into Word2Vec vectors.

        Args:
            data (array): Array of strings.
            mo (int): Word2Vec model (0 for CBOW, 1 for Skip-Gram Model)

        Returns:
            np.array: Array of vectors.
        """
    sentences = [doc.split() for doc in data]  # convert to Tokens for each string
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=mo)

    vectors = []
    for doc in sentences:
        # vector = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0) # wv = word vector
        #
        # ΛΥΣΗ για το ValueError: setting an array element with a sequence. The requested array has an
        #     # inhomogeneous shape after 1 dimensions. The detected shape was (18846,) + inhomogeneous
        embeddings = [model.wv[word] for word in doc if word in model.wv]
        if embeddings:
            vector = np.mean(embeddings, axis=0)
        else: #empty
            vector = np.zeros(model.vector_size)
        vectors.append(vector)

    return np.array(vectors)


def get_fasttext_embeddings(data):
    # Κώδικας από το Google
    """
            Converts text data into Fasttext vectors. Κώδικας από το Google. Use of nltk library. Package punkt_tab is downloaded.

            Args:
                data (array): Array of strings.

            Returns:
                np.array: Array of vectors.
            """
    tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in data]
    model = FastText(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, epochs=10)
    embeddings = np.array([np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
                            for doc in tokenized_docs if doc])
    return embeddings

def transform_fasttext(data):
    # PROBLEM, GIVES ERROR:
    # ValueError: setting an array element with a sequence. The requested array has an
    # inhomogeneous shape after 1 dimensions. The detected shape was (18846,) + inhomogeneous
    # part.
    # ΔΕΝ ΞΕΡΩ ΠΩΣ ΛΥΝΕΤΑΙ
    """
        Converts text data into Fasttext vectors.

        Args:
            data (array): Array of strings.

        Returns:
            np.array: Array of vectors.
        """
    sentences = [doc.split() for doc in data] # convert to Tokens for each string
    model = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=10)

    vectors = []
    for doc in sentences:
        # vector = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
        #
        # ΛΥΣΗ για το ValueError: setting an array element with a sequence. The requested array has an
        #     # inhomogeneous shape after 1 dimensions. The detected shape was (18846,) + inhomogeneous
        embeddings = [model.wv[word] for word in doc if word in model.wv]
        if embeddings:
            vector = np.mean(embeddings, axis=0)
        else:  # empty
            vector = np.zeros(model.vector_size)
        vectors.append(vector)

    return np.array(vectors)


def evaluate_clustering(labels_true, labels_pred):
    """
       Calculates clustering metrics:
       NMI (Normalized Mutual Information), ARI (Adjusted Rand Index) & AMI (Adjusted Mutual Information),.

       Args:
           labels_true (array): Array of true labels.
           labels_pred (array): Array of predicted labels.

       Returns:
           nmi (float): Normalized Mutual Information
           ari (float): Adjusted Rand Index
           ami (float): Adjusted Mutual Information.
     """
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)

    return nmi, ari, ami

def kmeans_inertia_ssc(k_values, data, title, plot: bool=False):
    """
       Performs KMeans in a range of 1 to k values, calculates and prints Inertias(s) & Silhouette score(s), plots Inertia(s) vs k values if wanted.

       Args:
           k_values (int): the number of clusters for KMeans.
           data (Any): data to transform.
           title (string): the title to appear in printouts and plot
           plot (bool): True if Inertia vs k clusters is ploted in a graph, False otherwise
    """
    k_values = range(1, k_values)
    inertia_values = []
    silhouette_scores = []
    for k in k_values:
        # from Exercises_to_give.Exercise_11_UL.clustering_algorithms.exc_kmeans import KMeansClustering
        ckmeans = KMeansClustering(n_clusters=k)
        labels = ckmeans.fit_predict(data)
        inertia_values.append(round(ckmeans.get_inertia(), 2))
        if k > 1:
            silhouette_scores.append(round(silhouette_score(data, labels),6))

    print('Inertia values =', inertia_values)
    print('Silhouette scores =', silhouette_scores)
    if plot:
        # φτιάξε το γράφημα inertia vs k (=Elbow curve):
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, inertia_values, marker='o', linestyle='--', color='b')
        plt.title(f'Elbow Method for Optimal K {title}')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.grid()
        plt.show()


def find_optimal_pca(data, max_components, max_clusters, labels_true):
    """
       Performs iterative PCAs for variable components and iterative evaluations of KMeans & Agglomerative for each PCA iteration computing NMI, ARI, AMI scores.

       Args:
           data (Array): the input data for PCA and Clustering.
           max_components (int): n_components in PCA.
           max_clusters (int): the number of clusters for both KMeans and Agglomerative.
           labels_true (array): the true labels of input data.

        Returns:
           best_ncluster_kmeans (dictionary): KMeans evaluation results for best number of clusters and PCA variable components.
           best_ncluster_agglo (dictionary): Agglomerative evaluation results for best number of clusters and PCA variable components.

    """
    best_ncluster_kmeans = {'Method':'KMeans', 'components': -1,'NMI':-1, 'ARI':-1, 'AMI':-1}
    best_ncluster_agglo = {'Method':'Agglomerative','components': -1,'NMI':-1, 'ARI':-1, 'AMI':-1}
    n_values = range(1, max_components)
    for n in n_values:
        best_nmi_k =-1
        best_ari_k = -1
        best_ami_k = -1 # KMeans NMI, ARI, AMI
        best_nmi_a = -1
        best_ari_a = -1
        best_ami_a = -1 # Agglomerative NMI, ARI, AMI
        best_ncluster_kmeans['components'] = n #create row, KMeans dictionary
        best_ncluster_agglo['components'] = n #create row, Agglomerative dictionary
        #
        pca = PCA(n_components=n)
        reduced_data = pca.fit_transform(data)
        #
        k_values = range(1, max_clusters)
        for k in k_values:
            # KMeans
            ckmeans = KMeansClustering(n_clusters=k)
            kmeans_labels = ckmeans.fit_predict(reduced_data)
            nmi_kmeans, ari_kmeans, ami_kmeans = evaluate_clustering(labels_true, kmeans_labels)
            if nmi_kmeans > best_nmi_k:
                best_nmi_k = nmi_kmeans
                best_ncluster_kmeans.update({'NMI':k})
            if ari_kmeans > best_ari_k:
                best_ari_k = ari_kmeans
                best_ncluster_kmeans.update({'ARI':k})
            if ami_kmeans > best_ami_k:
                best_ami_k = ami_kmeans
                best_ncluster_kmeans.update({'AMI':k})

            # Agglomerative
            model_agglo = AgglomerativeClusteringAlgorithm(n_clusters=k)
            agglo_labels = model_agglo.fit_predict(reduced_data)
            nmi_agglo, ari_agglo, ami_agglo = evaluate_clustering(labels_true, agglo_labels)
            if nmi_agglo > best_nmi_a:
                best_nmi_a = nmi_agglo
                best_ncluster_agglo.update({'NMI':k})
            if ari_agglo > best_ari_a:
               best_ari_a = ari_agglo
               best_ncluster_agglo.update({'ARI':k})
            if ami_agglo > best_ami_a:
               best_ami_a = ami_agglo
               best_ncluster_agglo.update({'AMI':k})

    return best_ncluster_kmeans, best_ncluster_agglo