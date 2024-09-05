#%%
#Import packages
import pandas as pd
#%%
#load in data
#load in business file to get the relevant business ID's
business_df = pd.read_csv('/Users/avagonick/Desktop/Break Through Tech/DirecTV2A/Philly Folder/MakePhilly_Restaurant/yelp_business_Food_Restaurant.csv')
#%%
#load in the reviews.json file
review_chunks = pd.read_json('/Users/avagonick/Desktop/Break Through Tech/Data/yelp_dataset/yelp_academic_dataset_review.json', lines = True, chunksize = 10000)
# %%
all_chunks = []
for chunk in review_chunks:
    all_chunks.append(chunk[chunk['business_id'].isin(business_df['business_id'])])
# %%
review_df = pd.concat(all_chunks, ignore_index = True)
# %%
print(review_df.head())
print(review_df.shape)
print(review_df.columns)
# %%
review_df.to_csv('Philly_Restaurants_Food.csv', index = False)
# %%
