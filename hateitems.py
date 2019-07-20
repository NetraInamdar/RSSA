import pandas as pd
import numpy as np
import random
import time
from itertools import product
import statistics

# ignore warning
# pd.options.mode.chained_assignment = None

input = pd.read_csv('results/mergePredictions.csv', sep=' ' , names=['user', 'item', 'prediction', 'rating', 'timestamp'])
# input = pd.read_csv('results/mergePredictions.csv', sep='\t' ,  header = None)
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
predictions = input
# input = pd.read_csv('results/rating100k.csv' , ' ', names=['user', 'item', 'rating', 'timestamp'])

users = ratings.user.unique()
items = ratings.item.unique()

meanItemRatings = pd.DataFrame(list(items))
meanItemRatings.columns = ['item']
meanItemRatings['mean'] = 0.0

def itemMean(meanItems, ratings):
	# use mean of prediction instead of mean of actual ratings for each item

	for i in range(len(items)):
		id = items[i]
		itemRatings = ratings[ratings['item'] == id]
		# print (itemRatings.head())
		#meanItems.iloc[i].values[1] = statistics.mean(itemRatings['rating'])
		meanItems.iloc[i, 1] = itemRatings['prediction'].mean()
		
	#print(meanItems.head(20))
	return meanItems
	
itemMeans = itemMean(meanItemRatings, predictions)
#np.savetxt("./results/meanItemRatings.csv", itemMeans, delimiter=" ")
mergeItemMeans = pd.merge(predictions, itemMeans, how = 'left', on = ['item'])

#np.savetxt("./results/mergeItemMeans.csv", mergeItemMeans, delimiter=" ")

def hateItems(userID, num_recommendations=10):
	userMergedData = mergeItemMeans[mergeItemMeans['user'] == userID].copy()
	userMergedData['diff'] = userMergedData['mean'] - userMergedData['prediction']
		# use .copy() to copy data instead of reference/indexing view, to avoid SettingWithCopyWarning
	# print(mergeItemMeans.head(20))	
	## userMergedData.columns = [['user', 'item', 'prediction', 'rating', 'timestamp', 'mean', 'diff']]
	sortForUser = userMergedData.sort_values(by='diff', ascending = False)
	# filtered_sortForUser = sortForUser[sortForUser['rating'].notnull()]
	filtered_sortForUser = sortForUser[sortForUser['rating'].isnull()]
		# filter out the items with actual observations
	## change codes for swagger	
	recs = filtered_sortForUser.head(num_recommendations)
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

