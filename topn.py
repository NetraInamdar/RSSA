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
model = algo.train(train)

spentTime = time.time() - startTime
print( "Training model takes %3.2f seconds\n" % spentTime)
	# training model takes time!!!

# extract unique users in test set
# users = test.user.unique()

users = ratings.user.unique()

# length = int(sys.argv[3])
num_recommendations = int(sys.argv[3])

def getRecommendations(user):
    """
    Generate a recommendation for the user
    :param algo: the given algorithm
    :param model: the given trained model
    :param user: the user
    :return: recommendation
    """
    # Generate $num_recommendations for the givenuser  
    recs = batch.recommend(algo, model, users, num_recommendations, topn.UnratedCandidates(train))
    # recs = batch.recommend(algo, users, num_recommendations, users, train, 0.5)
	## Jan. 2019: 
		# https://lkpy.readthedocs.io/en/latest/batch.html?highlight=batch
		# ? topn.UnratedCandidates(train)
		# generate recs topn for all input users
		# batch.recommend returns dataframe [user, rank, item, score], also possibly others?
		
    np.savetxt("./results/recs.csv", recs, delimiter=" ")
    return recs[recs['user'] == user], recs

# user = np.array(users[0])
# user = random.choice(users)

#userID = input("input an userID (1-943): ")
#user = int(userID)

user = int(sys.argv[1])
sectionID = int(sys.argv[2])


def get_topn(user):
	[rec, recs] = getRecommendations(user)
	# select columns
	if not rec.empty:
		recColumns = rec[['item', 'score']]
		# np.savetxt("./results/recColumns.csv", recColumns, delimiter=" ")
		# select row (normally not needed)
		# print (recColumns.values)
		# df = pd.DataFrame(recColumns)
		recColumns_dict = recColumns.to_dict(orient="records")
		recommendations = []
		for entry in recColumns_dict:
			recommendations.append({
				"item": int(entry['item']),
				"score": int(entry['score'])
			})
	else:
		abort(
			404, "No topN recommendations for user with user_id {user_id}".format(user_id = user_id)
		)
		
	return recommendations
	
#result = get_topn(user)
#print (result)
