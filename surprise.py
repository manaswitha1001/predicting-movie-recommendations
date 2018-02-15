#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 16:30:48 2017

@author: manaswithachimakurthi
"""

from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
#from surprise import KNNBasic
#algo1 = KNNBasic()

# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=4)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)


trainset = data.build_full_trainset()
algo.train(trainset)


userid = str(1)
itemid = str(5)

print(algo.predict(userid, 5, 3))





from collections import defaultdict

def precision_recall_at_k(predictions, k=20, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


data = Dataset.load_builtin('ml-100k')
data.split(n_folds=4)
algo = SVD()

f=[]
f2=[]

for trainset, testset in data.folds():
    algo.train(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=20, threshold=4)

    # Precision and recall can then be averaged over all users
    pr=sum(prec for prec in precisions.values()) / len(precisions)
    rr=sum(rec for rec in recalls.values()) / len(recalls)
    
    f1=2*(pr*rr)/(pr+rr)
    print(f1)
    

    
