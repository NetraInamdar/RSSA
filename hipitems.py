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

def hipItems(userID, num_recommendations = 10):
	userMergedData = mergeItemCounts[mergeItemCounts['user'] == userID]
	filtered_userMergedData = userMergedData[userMergedData['rating'].isnull()]
		# filter out the items with observations
	
	
	filtered_sortByPrediction = filtered_userMergedData.sort_values(by='prediction', ascending = False).copy()
	# sortByPrediction['ranking'] = sortByPrediction.index.get_values()
	numRows = filtered_sortByPrediction.shape[0]
	filtered_sortByPrediction['ranking'] = range(1, numRows+1)
	topNNfiltered_userMergedData = filtered_sortByPrediction.head(200)
	# print(topNNfiltered_userMergedData.head(10))
	
	filtered_sortByCount = topNNfiltered_userMergedData.sort_values(by='count', ascending = True)
	## change codes for swagger
	recs = filtered_sortByCount.head(num_recommendations)
	if not recs.empty:
		# select colums
		rows = recs[['item', 'prediction']]
		rows_dict = rows.to_dict(orient="records")
		recommendations = []
		for entry in rows_dict:
			recommendations.append({
				"item": int(entry['item']),
				"score": float(entry["prediction"])
			})
	# otherwise, nope, not found
	else:
		abort(
			404, "No ratings for user with user_id	{user_id} ".format(user_id=userID)	#user_id=user_id was changed to user_id=userID
		)
	return recommendations
