import pandas as pd
import numpy as np
import random
import time
from itertools import product
import statistics

input = pd.read_csv('results/mergePredictions.csv', sep=' ' , names=['user', 'item', 'prediction', 'rating', 'timestamp'])
# input = pd.read_csv('results/mergePredictions.csv', sep='\t' ,  header = None)
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
predictions = input
# input = pd.read_csv('results/rating100k.csv' , ' ', names=['user', 'item', 'rating', 'timestamp'])

users = ratings.user.unique()
items = ratings.item.unique()

countRatings = pd.DataFrame(list(items))
countRatings.columns = ['item']
countRatings['count'] = 0

def ratedItemCount(countRatings, ratings):
	for i in range(len(items)):
		id = items[i]
		itemRatings = ratings[ratings['item'] == id]
		countRatings.iloc[i, 1] = itemRatings.shape[0]
		
	# print(countRatings.head(20))
	return countRatings
	
itemCounts = ratedItemCount(countRatings, ratings)
mergeItemCounts = pd.merge(predictions, itemCounts, how = 'left', on = ['item'])

def hipItems(userID, numRec, mergedData):
	userMergedData = mergedData[mergedData['user'] == userID]
	filtered_userMergedData = userMergedData[userMergedData['rating'].isnull()]
		# filter out the items with observations
	
	
	filtered_sortByPrediction = filtered_userMergedData.sort_values(by='prediction', ascending = False).copy()
	# sortByPrediction['ranking'] = sortByPrediction.index.get_values()
	numRows = filtered_sortByPrediction.shape[0]
	filtered_sortByPrediction['ranking'] = range(1, numRows+1)
	topNNfiltered_userMergedData = filtered_sortByPrediction.head(200)
	# print(topNNfiltered_userMergedData.head(10))
	
	filtered_sortByCount = topNNfiltered_userMergedData.sort_values(by='count', ascending = True)
	print(filtered_sortByCount.head(10))
	return filtered_sortByCount.head(numRec)

	
num_recommendations = 10
user = random.choice(users)
novelItemsForTheUser = hipItems(user, num_recommendations, mergeItemCounts)
#sortedNovelItemsForTheUser = novelItemsForTheUser.sort_values(by = 'prediction', ascending = False)

recHateItems = novelItemsForTheUser[['item', 'prediction', 'count', 'ranking']]
#recHateItems = sortedNovelItemsForTheUser[['item', 'prediction']]
print("For userID:", user)
print("   Things you will be among the first to try: ")
print("%20s%20s%28s%20s" % ("itemID", "score", "ratingCounts", "rankingByScore"))
for index, row in recHateItems.iterrows():
	print ("%20.0f%20.2f%28d%20d" % (row[0], row[1], row[2], row[3]))
