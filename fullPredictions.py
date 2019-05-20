import pandas as pd
import numpy as np
from lenskit import batch, topn
from lenskit import crossfold as xf
from lenskit.algorithms import funksvd
#from lenskit.algorithms import als 
from lenskit.algorithms import item_knn as knn
import random
import time
from itertools import product
import statistics

startTime = time.time()
# read in the movielens 100k ratings with pandas
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

#algo = knn.ItemItem(20)
algo = funksvd.FunkSVD(50)
#algo = als.BiasedMF(50)
	# use als, paralleling computation

'''
# train and test 
# split the data into a test set and a training set, k-fold xf
num_folds = 5
splits = xf.partition_users(ratings, num_folds, xf.SampleFrac(0.2))
for (trainSet, testSet) in splits:
    train = trainSet  #?
    test = testSet    #?
'''
startTime = time.time()
# train model
model = algo.train(ratings)
spentTime = time.time() - startTime
print( "Training model takes %3.2f seconds" % spentTime)
	# training model takes time!!!
	
	
# extract unique user set and unique item set
users = ratings.user.unique()
items = ratings.item.unique()
print("# users: ", len(users))
print("# items: ", len(items))
# generate full dataset with all possible user-item pairs 
# i.e. fill the UI matrix in long format
fullPairs = product(users, items)

fullPairs = list(fullPairs)
#print(fullPairs.head())


# takes time !!
startTime = time.time()
fullPairs = pd.DataFrame(fullPairs)
spentTime = time.time() - startTime
print( "Time spent when converting from list to dataFrame: %3.2f" % spentTime)

fullPairs.columns = ['user', 'item']
#np.savetxt("./results/fullUIpairs.csv", combinations, delimiter=" ")
#users.sort();
#items.sort();
# print(users)
# print(items)
# merge observed ratings to full datasets
# print(fullPairs.head())

def getPredictions(algo, dataPairs, model):
    """
    Generate a recommendation for the user
    :param algo: the given algorithm
    :param model: the given trained model
    :param user: the user
    :return: recommendation
    """
    # Generate $num_recommendations for the givenuser  
    predictions = batch.predict(algo, dataPairs, model)
		# https://lkpy.readthedocs.io/en/latest/batch.ht ml?highlight=batch
		# batch.predict returns dataframe [dataPairs['all columns'], 'prediction']
    #np.savetxt("./results/full_prediction.csv", predictions, delimiter=" ")
    return predictions

fullPredct = getPredictions(algo, fullPairs, model)
#print(fullPredct.head())
#print(fullPredct.shape)
np.savetxt("./full_prediction.csv", fullPredct, delimiter=" ")
# np.savetxt("./results/rating100k.csv", ra, delimiter=" ")

# np.savetxt("./results/rating100k.csv", ratings, delimiter=" ")

merging = pd.merge(fullPredct, ratings, how = 'left', on = ['user', 'item'])
merging.columns = ['user', 'item', 'prediction', 'rating', 'timestamp']
np.savetxt("./mergePredictions.csv", merging, delimiter=" ")

print(merging.shape)


'''
#######
# list 1
meanItemRatings = pd.DataFrame(list(items))
meanItemRatings.columns = ['item']
meanItemRatings['mean'] = 0
#print (meanItemRatings.head())


def getHateItems(userID, numRec, ratings, predictions):
	#get mean/average rating each item based on observed ratings
	i = 0;
	for id in items:
		itemRatings = ratings[ratings['item'] == id]
		#print(itemRatings.head())
		meanItemRatings.iloc[i].values[1] = statistics.mean(itemRatings['rating'])
		#print(meanItemRatings.iloc[i].values[1])
		i = i+1
	
	#print(meanItemRatings.head())
	userPredctRatings = predictions[predictions['user'] == userID]
	merging = pd.merge(userPredctRatings, meanItemRatings, on = ['item'])
	#print(merging.head())
	merging['diff'] = merging['mean'] - merging['rating']
	diffForUser = merging['item','diff']
	sortForUser = diffForUser.sort_values(by='diff', ascending = False)
	return sortForUser.head(numRec)
	
		
		

num_recommendations = 10
user = random.choice(users)

#sortedDiff = getHateItems(user, num_recommendations, ratings, fullPredct)
#print(sortedDiff)
#for index, row in soretedDiff.iterrows():
#	print ("\t%-20.0f%-7.2f" % (row[0], row[1]))
'''