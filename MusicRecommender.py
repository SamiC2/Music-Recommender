#!/usr/bin/env python
# coding: utf-8

# Make necessary package imports

# In[ ]:


from pyspark.mllib.recommendation import *
import random
from operator import *
from pyspark import SparkContext
sc = SparkContext.getOrCreate()


# artist_data_small.txt contains information about each user's unique artistId and its name. The file is read and then a map is created with key: artistId and value: artist Name

# In[ ]:


# Method to split artist Id and its name.
def splitArtistName(line):
    try:
        id, name = line.split("\t")
        return (int(id), name)
    except ValueError:
        return None

# Load text file where each line contains artist Id and its name.
artistData = sc.textFile("artist_data_small.txt")
# Split artist id: name and store in a map. 
artistData = artistData.map(splitArtistName).filter(lambda x: x!=None).collectAsMap()


# AudioScrobbler has provided a file which contains information about an artist's other alias' / misspelled names. This information is corrected to the user-artist information by replacing the aliases by its uniqueId. 

# In[ ]:


'''
Load artist correct id and its aliases
    2 columns: badid, goodid
    known incorrectly spelt artists and the correct artist id. 
'''
artistAlias = sc.textFile('artist_alias_small.txt')
# Split Artist Alias data into (badId, goodId)
def splitArtistAlias(line):
    try:
        # Catches error in data
        badId, goodId = line.split("\t")
        return (int(badId), int(goodId))
    except ValueError:
        return None

# Create map badId: goodId

artistAlias = artistAlias.map(splitArtistAlias).filter(lambda x: x!=None).collectAsMap()


# As mentioned above, user_artist_data_small.txt contains misspelled artistId. Hence, use the artistAlias map to correct the entries in the RDD.

# In[ ]:


'''
Load data about user's music listening history
Each line contains three features: userid, artistid, playcount
'''
userArtistData = sc.textFile("user_artist_data_small.txt")

# Return the corrected user information.
def parseUserHistory(line):
    try:
        # Catch error in line
        user, artist, count = line.split()
        # Return the corrected user information.
        if artist in artistAlias:
            return (int(user), artistAlias[artist], int(count))
        else:
            return (int(user), int(artist), int(count))
    except ValueError:
        return None


# Create corrected user history RDD.
userArtistData = userArtistData.map(parseUserHistory)


# Since userArtistData would be used repeatedly, I decided  to cache this to avoid re-computation of the same RDD.

# In[ ]:


userArtistData.cache()


# The following section creates a new RDD containing basic user listening stats.

# In[ ]:


userArtistPurge = userArtistData.map(lambda x: (x[0],x[2]))
# Create an RDD storing user information in the form of (total play count of all artists combined for the current user, (userId of the current user, number of unique artists listened by user))
songCountAgg = userArtistPurge.aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1), lambda a,b: (a[0] + b[0], a[1] + b[1])).map(lambda x: (x[1][0], (x[0], x[1][1])))
# Sort the RDD based on the total play counts so as to find the most active user.
sortedCount = songCountAgg.sortByKey(False)
# Find the top 3 user information
sortedCountTop3 = sortedCount.take(3)

# Print the top 3 user information.

print "User %s has a total play count of %d and a mean play count of %s" %(sortedCountTop3[0][1][0],sortedCountTop3[0][0], sortedCountTop3[0][0]/sortedCountTop3[0][1][1])

print "User %s has a total play count of %d and a mean play count of %s" %(sortedCountTop3[1][1][0],sortedCountTop3[1][0], sortedCountTop3[1][0]/sortedCountTop3[1][1][1])

print "User %s has a total play count of %d and a mean play count of %s" %(sortedCountTop3[2][1][0],sortedCountTop3[2][0], sortedCountTop3[2][0]/sortedCountTop3[2][1][1])

# Split the dataset into 3 parts, trainingData, validationData and testData in the ratio of 4:4:2

# In[ ]:


trainData, validationData, testData = userArtistData.randomSplit([0.4,0.4,0.2], seed=100)


# Since these dataset would be used repeatedly, cache this to avoid re-computation of the same RDDs.

# In[ ]:


trainData.cache()
validationData.cache()
testData.cache()


# In the next section we define a method which would evaluate the accuracy of the model based on the validation set. Instead of the common mean squared error which can be defined as (MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()) in python, I used a different error function. In this error function, calculate the fraction of the overlapping artists between the predicted artists/ true artists. Then sum this fraction for all users and the final score is the sum normalized by the total number of users. In my evaluation model, I purge out all the arists listed in the training Data for that particular user while making the top-K predictions, to reduce the bias in error estimate.

# In[ ]:


def modelEval(model, data):
    artistsList =  broadcastVar.value
    total = 0.0
    userList = validationData.map(lambda x: x[0]).distinct().collect()
    for user in users:
        trainArtists = set(trainData.filter(lambda x: x[0]==userList).map(lambda x: x[1]).collect())
        #Remove artists for the current user in training Dataset from the userArtistData.
        nonTrainArtists = sc.parallelize([(user,artist) for artist in artistsList if not artist in trainArtists])
        #use the model to predict all the ratings on nonTrainArtists
        prediction = model.predictAll(nonTrainArtists)
        #top X sorted by highest rating from the prediction for the current user
        X = len(trueArtists)
        topX = sorted(prediction.collect(), key=lambda x: x.rating, reverse=True)[:X]
        
        trueArtists = set(data.filter(lambda x: x[0]==userList).map(lambda x: x[1]).collect())
        topArtist = set(topX.map(lambda x: x[1]))
        #Compare predictResult to trueArtists
        total += float(len(topArtist & trueArtists))/len(trueArtists)
    return total/len(userList)


# The following command will fetch all the distinct artistIDs. Since the allArtist list is huge in size and would be sent to multiple node clusters, many times, it would make sense to broadcast the variable which would send it to all the node clusters and their respective partitions once and cache them for reusablity.

# In[ ]:


allArtists = userArtistData.map(lambda x: x[1]).distinct().collect()
broadcastVar = sc.broadcast(allArtists)


# Here the model is evaluated on the validation dataset based on different rank parameters. The rank parameter corresponds to the number of latent factors in the matrix factorization of the ALS algorithm. Hence run the model for different rank parameter and then choose the one which yielded the best accuracy.

# In[ ]:


# print the model accuracy score 
for val in [2, 10, 20]:
    model = ALS.trainImplicit(training, rank=val, seed=345)
    print("The model score for rank %d is %f" % (rank, modelEval(model, validationData)))

# We choose the model which yielded the highest accuracy as our final model to make recommendations.

# In[ ]:


bestModel = ALS.trainImplicit(trainData, rank=10, seed=345)
modelEval(bestModel, testData)


# The model is ready to make recommendations.  I now made top 5 artist recommendations for user 1059637. The function recommendProducts takes userId as the first input parameter and an integer n, while returning the top n highest ranked recommendations.

# In[ ]:


top5=bestModel.recommendProducts(1059637, 5)
i=1
for val in top5:
    print "Artist %d : %s" %(i,dictionary[val[1]])
    i=i+1

def main():
	
