import pandas as pd
import numpy as np
from lenskit import batch, topn
from lenskit import crossfold as xf
from lenskit.algorithms import funksvd
from lenskit.algorithms import als 
from lenskit.algorithms import item_knn as knn
import random
import time

from flask import make_response, abort, jsonify
import sys

'''
get topN basd on MF
'''

# repo of the library
#/home/lijieg/anaconda3/lib/python3.7/site-packages/lens

# read in the movielens 100k ratings with pandas
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

# change data type
# ratings = ratings1.astype(int)

#algo = knn.ItemItem(20)
#algo = funksvd.FunkSVD(50)
algo = als.BiasedMF(10)

startTime = time.time()
# split the data into a test set and a training set, k-fold xf
num_folds = 5
splits = xf.partition_users(ratings, num_folds, xf.SampleFrac(0.2))
for (trainSet, testSet) in splits:
    train = trainSet  #?
    test = testSet    #?

print(1)
# train model
model = algo.fit(train)  # changed algo.train() to algo.fit()

spentTime = time.time() - startTime
print( "Training model takes %3.2f seconds\n" % spentTime)
	# training model takes time!!!

# extract unique users in test set
# users = test.user.unique()

users = ratings.user.unique()

# length = int(sys.argv[3])
num_recommendations = 10  # hard coded num of recommendations to 10


def get_topn(user,num_recommendations=10): #added argument num_recommendations=10

    #algo = algoKNN
    #model = modelKNN


    # merged the code for get_topn and getRecommendaions into one function:
    recs = batch.recommend(algo, users,num_recommendations, topn.UnratedCandidates(train), nprocs=None)  # changed arguments for batch.recommend
    recs = recs[recs['user'] == user]
    if not recs.empty:
        # select colums
        rows = recs[['item', 'score']]
        rows_dict = rows.to_dict(orient="records")
        recommendations = []
        for entry in rows_dict:
            recommendations.append({
                "item": int(entry['item']),
                "score": int(entry["score"])
            })
    # otherwise, nope, not found
    else:
        abort(
            404, "No ratings for user with user_id  {user_id} ".format(user_id=user)  #user_id=user_id was changed to user_id=user
        )
    return recommendations
