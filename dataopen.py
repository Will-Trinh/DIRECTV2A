#%%
#import libraries
import json
import pandas as pd
import csv
from sklearn.preprocessing import MultiLabelBinarizer
#%%

#load in the json file into a list
data_list = []

with open('yelp_dataset/yelp_academic_dataset_business.json', 'r') as file:
    for line in file:
        data = json.loads(line)
        data_list.append(data)

#%%
#convert the json file to a pandas dataframe
business_df = pd.DataFrame(data_list)
business_df.shape
print(business_df.columns)
# %%
"""
#When using LA zip codes get 0 values, there really isn't any data on LA 
#load in LA zip codes data from la county website dataset
zip_codes = []
with open('LA_County_ZIP_Codes.csv') as file:
    reader = csv.reader(file)
    for row in reader:
        #it is a csv file and the zip codes are held in the first column
        zip_codes.append(row[0])
#%%
#filter the business_df dataframe by these LA zip codes
filter = business_df['postal_code'].isin(zip_codes)
business_df = business_df[filter]
print(business_df.shape)
#%%
#try filtering by just the state of california 
business_df = business_df[business_df['state'] == 'CA']
business_df.head()
print(business_df.shape)
"""
#%%
#expand the attributes 
attributes_df = business_df['attributes'].apply(pd.Series)
business_df = business_df.drop('attributes', axis = 1).join(attributes_df)
#%%
#use the attributes_df to get all the attributes
attributes = attributes_df.columns
print(attributes)
# %%
#expand the hours column
hours_df = business_df['hours'].apply(pd.Series)
business_df = business_df.drop('hours', axis = 1).join(hours_df)
# %%
#expand the categories
mlb = MultiLabelBinarizer(sparse_output = True)

#categories is comma separated, convert everything to lists so the multi label binarizer can work
business_df['categories'] = business_df['categories'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
#%%
#binarize the categories
categories_sparse = mlb.fit_transform(business_df['categories'])
# %%
print(categories_sparse.shape)
#%%
#convert the sparse matrix into a pandas dataframe
categories_df = pd.DataFrame.sparse.from_spmatrix(categories_sparse, index = business_df.index, columns = mlb.classes_)
categories_df.head
#%%
#join the matrices
business_df = business_df.drop('categories', axis = 1).join(categories_df)
#%%
#get the categories from categories_df
categories = mlb.classes_
print(categories)
# %%
#convert and output to a csv so the preprocessing doesn't have to be redone 
business_df.head()
# %%
#convert and output to a csv
business_df.to_csv('yelp_business.csv', index = False)
#%%
#output the attributes and categories variables into a text file so it can be analyzed 
with open('Attributes_and_categories.txt', 'w') as file:
    #write attributes 
    file.write('Attributes\n)')
    for attribute in attributes:
        file.write(f'{attribute}, ')

    #write categories
    file.write('\nCategories\n')
    for category in categories:
        file.write(f'{category}, ')    
# %%
