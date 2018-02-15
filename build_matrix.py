# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:27:02 2017

@author: Krishna
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import scipy
def generate_matrix(filename):
    # Load the dataset
    df1 = pd.read_csv(filename)
    
    # Find the number of unique movies and users
    users = set()
    movies = set()
    
    for i in range(len(df1)):
        users.add(df1.ix[:,0][i])
        movies.add(df1.ix[:,1][i])
        
    # Convert ids to list and sort them in ascending order
    movies_lst = list(movies)
    movies_lst.sort()
    users_lst = list(users)
    users_lst.sort()
        
    # Create a matrix
    df2 = pd.DataFrame(index = users_lst, columns = movies_lst)
    
    for i in range(len(df1)):
        userid = df1.ix[:,0][i]
        movieid = df1.ix[:,1][i]
        rating = df1.ix[:,2][i]
        df2.set_value(userid, movieid, rating)
        
        if i % 10000 == 0:
            print (i)
    
    # Replacing nulls as zeros
    df2 = df2.fillna(0)
    
    return (df2)
    
##################################################

def get_movies_users_lst(filename):
    # Load the dataset
    df1 = pd.read_csv(filename)
    
    # Find the number of unique movies and users
    users = set()
    movies = set()
    
    for i in range(len(df1)):
        users.add(df1.ix[:,0][i])
        movies.add(df1.ix[:,1][i])
        
    # Convert ids to list and sort them in ascending order
    movies_lst = list(movies)
    movies_lst.sort()
    users_lst = list(users)
    users_lst.sort()
    
    return(movies_lst, users_lst)

###############################################################

    
    
base = '/Users/manaswithachimakurthi/Desktop/'
filename = base + 'ratings.csv'
df = generate_matrix(filename)
movies_lst, users_lst = get_movies_users_lst(filename)

# Access sample user and corresponding movie
df.loc[671][6565]
df.loc[2][153]
#df = df.fillna(0)
#df.head
########################

users_lst

movieid = 31
df.loc[1][31:34]


# Get list of movies and ratings of target user
user1_movies = []
user1_ratings = []
for movieid in range(len(movies_lst)):
    if df.loc[1][movies_lst[movieid]] != 0:
        user1_movies.append(movies_lst[movieid])
        #user1_movies.append([movies_lst[movieid], df.loc[1][movies_lst[movieid]]])
        user1_ratings.append(df.loc[1][movies_lst[movieid]])
        #print(df.loc[1][movies_lst[movieid]])
        
        
# Build a subset matrix based on the movies the target user has seen
X = df.ix[:,user1_movies[0:10]]
y = df.ix[:,user1_movies[10:20]]





user1_movies[0:10]
user1_ratings

X = user1_ratings[0:10]
y = np.array(user1_ratings[10:len(user1_ratings)])





df.loc[user1_movies[1]]

X = df.ix[:,user1_movies[0:10]]

Xsvtest=X
y = df.ix[:,user1_movies[10:20]]

X_train = np.array(X.ix[2:len(df)])
X_test = np.array(X.ix[1])
#y_train = np.array(y.ix[2:len(df)])
y_train = np.array(y.ix[2:,1371])
y_test = np.array(y.ix[1])

#knn = sklearn.neighbors.kneighbors(X_train,n_neighbors=3)

#sklearn.neighbors.NearestNeighbors.kneighbors(X_train, n_neighbors = 3)
#knn.fit(X_train , y_train)
#y_train.shape

#X_train = X.ix[2:len(df)]



import pandas as pd
import numpy as np

# Read the dataset into a data table using Pandas
df = pd.read_csv("/Users/manaswithachimakurthi/Desktop/movie_ratings_data_set.csv")

# Convert the running list of user ratings into a matrix using the 'pivot table' function
ratings_df = pd.pivot_table(df, index='user_id', columns='movie_id', aggfunc=np.max)

# Create a csv file of the data for easy viewing
ratings_df.to_csv("/Users/manaswithachimakurthi/Desktop/review_matrix.csv", na_rep="")



dfx=pd.read_csv("/Users/manaswithachimakurthi/Desktop/ratings.csv")


#names=['user_id','movieid','rating','timestamp']

movies_df=pd.read_csv("/Users/manaswithachimakurthi/Desktop/moviessvd.csv",sep='::',header=None,
                      names=['movieId','movie_title','genre'])




def normalize_ratings(ratings):
    """
    Given an array of user ratings, subtract the mean of each product's ratings
    :param ratings: 2d array of user ratings
    :return: (normalized ratings array, the calculated means)
    """
    mean_ratings = np.nanmean(ratings, axis=0)
    return ratings - mean_ratings, mean_ratings


def cost(X, *args):
    """
    Cost function for low rank matrix factorization
    :param X: The matrices being factored (P and Q) rolled up as a contiguous array
    :param args: Array containing (num_users, num_products, num_features, ratings, mask, regularization_amount)
    :return: The cost with the current P and Q matrices
    """
    num_users, num_products, num_features, ratings, mask, regularization_amount = args

    # Unroll P and Q
    P = X[0:(num_users * num_features)].reshape(num_users, num_features)
    Q = X[(num_users * num_features):].reshape(num_products, num_features)
    Q = Q.T

    # Calculate current cost
    return (np.sum(np.square(mask * (np.dot(P, Q) - ratings))) / 2) + ((regularization_amount / 2.0) * np.sum(np.square(Q.T))) + ((regularization_amount / 2.0) * np.sum(np.square(P)))


def gradient(X, *args):
    """
    Calculate the cost gradients with the current P and Q.
    :param X: The matrices being factored (P and Q) rolled up as a contiguous array
    :param args: Array containing (num_users, num_products, num_features, ratings, mask, regularization_amount)
    :return: The gradient with the current X
    """
    num_users, num_products, num_features, ratings, mask, regularization_amount = args

    # Unroll P and Q
    P = X[0:(num_users * num_features)].reshape(num_users, num_features)
    Q = X[(num_users * num_features):].reshape(num_products, num_features)
    Q = Q.T

    # Calculate the current gradients for both P and Q
    P_grad = np.dot((mask * (np.dot(P, Q) - ratings)), Q.T) + (regularization_amount * P)
    Q_grad = np.dot((mask * (np.dot(P, Q) - ratings)).T, P) + (regularization_amount * Q.T)

    # Return the gradients as one rolled-up array as expected by fmin_cg
    return np.append(P_grad.ravel(), Q_grad.ravel())


def low_rank_matrix_factorization(ratings, mask=None, num_features=15, regularization_amount=0.01):
    """
    Factor a ratings array into two latent feature arrays (user features and product features)

    :param ratings: Matrix with user ratings to factor
    :param mask: A binary mask of which ratings are present in the ratings array to factor
    :param num_features: Number of latent features to generate for users and products
    :param regularization_amount: How much regularization to apply
    :return: (P, Q) - the factored latent feature arrays
    """
    num_users, num_products = ratings.shape

    # If no mask is provided, consider all 'NaN' elements as missing and create a mask.
    if mask is None:
        mask = np.invert(np.isnan(ratings))

    # Replace NaN values with zero
    ratings = np.nan_to_num(ratings)

    # Create P and Q and fill with random numbers to start
    np.random.seed(0)
    P = np.random.randn(num_users, num_features)
    Q = np.random.randn(num_products, num_features)

    # Roll up P and Q into a contiguous array as fmin_cg expects
    initial = np.append(P.ravel(), Q.ravel())

    # Create an args array as fmin_cg expects
    args = (num_users, num_products, num_features, ratings, mask, regularization_amount)

    # Call fmin_cg to minimize the cost function and this find the best values for P and Q
    X = scipy.optimize.fmin_cg(cost, initial, fprime=gradient, args=args, maxiter=3000)

    # Unroll the new P and new Q arrays out of the contiguous array returned by fmin_cg
    nP = X[0:(num_users * num_features)].reshape(num_users, num_features)
    nQ = X[(num_users * num_features):].reshape(num_products, num_features)

    return nP, nQ.T


def RMSE(real, predicted):
    """
    Calculate the root mean squared error between a matrix of real ratings and predicted ratings
    :param real: A matrix containing the real ratings (with 'NaN' for any missing elements)
    :param predicted: A matrix of predictions
    :return: The RMSE as a float
    """
    return np.sqrt(np.nanmean(np.square(real - predicted)))




#ratings_df.as_matrix()

X_k= X.as_matrix()

U, M =low_rank_matrix_factorization(X_k, num_features=15,regularization_amount=0.1)

# Find all predicted ratings by multiplying U and M matrices
predicted_ratings = np.matmul(U, M)








dfx



print("Enter a user_id to get recommendations (Between 1 and 100):")
user_id_to_search = int(input())

print("Movies previously reviewed by user_id {}:".format(user_id_to_search))

reviewed_movies_df =dfx[dfx['userId']==user_id_to_search]
reviewed_movies_df =reviewed_movies_df.join(movies_df,on='movieId')

print(reviewed_movies_df[['title', 'genre', 'value']])

input("Press enter to continue.")

print("Movies we will recommend:")

user_ratings =predicted_ratings[user_id_to_search-1]
movies_df['rating'] =user_ratings

already_reviewed =reviewed_movies_df['movie_id']
recommended_df =movies_df[movies_df.index.isin(already_reviewed)==False]
recommended_df = recommended_df.sort_values(by=['rating'], ascending=False)

print(recommended_df[['title', 'genre', 'rating']].head(5))









