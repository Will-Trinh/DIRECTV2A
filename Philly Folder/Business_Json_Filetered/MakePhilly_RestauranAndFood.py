#%%
#import libraries
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#%%
#convert the json file to a pandas dataframe
Philly_Restaurant_Food_df = pd.read_json('../../data/yelp_dataset/yelp_academic_dataset_business.json', lines = True)
#%%
#Filter by location only doing Philly
Philly_Restaurant_Food_df = Philly_Restaurant_Food_df[Philly_Restaurant_Food_df['city'] == 'Philadelphia']

#%%
#filter just for categories with restaurant and food in it so we can analyze this
Philly_Restaurant_Food_df = Philly_Restaurant_Food_df[Philly_Restaurant_Food_df['categories'].str.contains('Restaurant', case = True, na = False) | Philly_Restaurant_Food_df['categories'].str.contains('Food', case = True, na = False)]

#%%
#expand the attributes 
attributes_df = Philly_Restaurant_Food_df['attributes'].apply(pd.Series)
Philly_Restaurant_Food_df = Philly_Restaurant_Food_df.drop('attributes', axis = 1).join(attributes_df)
#%%
#use the attributes_df to get all the attributes
attributes = attributes_df.columns
print(attributes)
# %%
#expand the hours column
hours_df = Philly_Restaurant_Food_df['hours'].apply(pd.Series)
business_df = Philly_Restaurant_Food_df.drop('hours', axis = 1).join(hours_df)
# %%
#do the same for the restaurants version
mlb = MultiLabelBinarizer()

#convert the categories into so multiLayer binarizer can work
Philly_Restaurant_Food_df['categories'] = Philly_Restaurant_Food_df['categories'].apply(lambda x: x.split(',') if isinstance(x, str) else []) 
#%%
#convert to restaurant categories
categories = mlb.fit_transform(Philly_Restaurant_Food_df['categories'])
#%%
#convert categories to dataframe and then join it
categories_df = pd.DataFrame(categories, index = Philly_Restaurant_Food_df.index, columns = mlb.classes_)
Philly_Restaurant_Food_df = Philly_Restaurant_Food_df.drop('categories', axis = 1).join(categories_df)
categories_restaurant = mlb.classes_
# %%
#convert and output to a csv
Philly_Restaurant_Food_df.to_csv('yelp_business_Food_Restaurant.csv', index = False)
#%%
#output the attributes and categories variables into a text file so it can be analyzed 
with open('Attributes_and_categories_Philly_Food_And_Restaurant.txt', 'w') as file:
    #write attributes 
    file.write('Attributes\n)')
    for attribute in attributes:
        file.write(f'{attribute}, ')

    #write categories
    file.write('\nCategories\n')
    for category in categories:
        file.write(f'{category}, ')    
# %%
