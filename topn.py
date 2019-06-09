import pandas as pd
import numpy as np
from lenskit import batch, topn
from lenskit import crossfold as xf
from lenskit.algorithms import funksvd
from lenskit.algorithms import als 
from lenskit.algorithms import item_knn as knn
import random
import time

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

# ? extract unique users in test set
users = test.user.unique()

#num_recommendations = 10 
# length = int(sys.argv[3])
num_recommendations = int(sys.argv[3])

def getRecommendations(user,sectionID,num_recommendations):
    """
    Generate a recommendation for the user
    :param algo: the given algorithm
    :param model: the given trained model
    :param user: the user
    :return: recommendation
    """
    id=sectionID
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

[rec, recs] = getRecommendations(user,sectionID, num_recommendations)
'''
#testing failed, no column name showed
	print ('Column names in recs: \n')
	recs.columns
	ratings.columns
'''

"""
recsUser = recs['user']
recsRank = recs['rank']
recsItem = recs['item']
recsScore = recs['score']
np.savetxt("./results/recs_user.csv", recsUser, delimiter=" ")
np.savetxt("./results/recs_rank.csv", recsRank, delimiter=" ")
np.savetxt("./results/recs_item.csv", recsItem, delimiter=" ")
np.savetxt("./results/recs_score.csv", recsScore, delimiter=" ")
"""

# select columns
recColumns = rec[['item', 'score']]
# np.savetxt("./results/recColumns.csv", recColumns, delimiter=" ")
# select row (normally not needed)
# print (recColumns.values)
# df = pd.DataFrame(recColumns)

print("For userID:", user)
print("   Top-N items: ")
print("%20s%20s" % ("itemID", "Score"))
for index, row in recColumns.iterrows():
	print ("%20.0f%20.2f" % (row[0], row[1]))
'''
rowSeries = recColumns.iloc[0:num_recommendations]
	#Selecting data by row numbers (.iloc)
item = rowSeries.values[0]
score = rowSeries.values[1]
print('\t', item, end = "\t")
print(score)
#print(type(item))
'''
