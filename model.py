#import packages
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

movie=pd.read_csv(r'Collaborative Filtering Dataset\dataset\movies.csv')
ratings=pd.read_csv(r"Collaborative Filtering Dataset\dataset\ratings.csv")
ratings=pd.merge(movie,ratings)
ratings.drop(columns=['movieId','genres','timestamp'],axis=1,inplace=True)
#convert it to pivot table like the above example
ratings=ratings.pivot_table(index='userId',columns='title',values='rating')
#there are many movies without any rating or very less ratings. drop movies that have less than 10 users
ratings.dropna(thresh=10,inplace=True,axis=1)#dropping movie columns
#fill the rest with 0
ratings.fillna(0,inplace=True)
#item_similarity using pearson correlation
item_similar_df=ratings.corr(method='pearson')#it will adjust for means
#defining the recommending function
#create the system
def recommend_model(movie,rating):
    score=item_similar_df[movie]*(rating-2.5)
    score.sort_values(ascending=False)
    return score
import pickle
pickle.dump(recommend_model,open('model.pkl','wb'))

