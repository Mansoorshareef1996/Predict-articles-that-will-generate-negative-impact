import pandas as pd  #importing pandas for reading csv files and dataframes
import numpy as np   #importing numpy for handling array operations
import matplotlib.pyplot as plt #for making plots
from sklearn.model_selection import train_test_split #Split arrays or matrices into random train and test subsets
from sklearn.linear_model import LogisticRegression  # Importing data for Logistic Regression
from sklearn import metrics # For making accuracy matrix
import seaborn as sns       #All libraries for data visualtization
import warnings 
warnings.filterwarnings('ignore')

"""### Lets first import our dataset from the given csv file and reading it throught the pandas
Once all the necessary libarary are loaded we then load the dataset which have the article and the reactions of the user to predict which articles could have a negative imapct. The dataset is strored as facebook in csv format which is loaded using the pandas library and we have used the ISO-8859-1 encoding.
"""

from google.colab import drive
drive.mount("/content/gdrive")

old_df = pd.read_csv("/content/gdrive/My Drive/facebook.csv",encoding = "ISO-8859-1",engine='python', nrows=111144) # #Loading the dataset from the file in variable old_df

old_df = old_df.loc[:, ~old_df.columns.str.contains('^Unnamed')] #To avoid any Unnamed columns in pandas dataframe

old_df.head()  #visualize data

len(old_df.columns) # printing the length of the columns

"""## Removing missing values from dataset
Once the datset is loaded we observe that there are many rows which have a NaN values which are no use. So we remove the missing values for the data frame which helps us visulaize the meaningful data in the dataset. The other thing that is calculated in the below step is the total negative reaction which would be sum of sad and angry reaction given by the user on facebook
"""

old_df = old_df.dropna() # Remove missing values from the dataframe
old_df = old_df.sort_values(by = ["total_angry" ] , ascending = False)[:-6] # sorting the values based on the total_angry reaction
columns = ["total_like" , "shares" , "visibility" , "total_love" , "total_wow" , "total_haha" , "total_sad" , "total_angry"] # col in the dataset
for col in columns:
    old_df[col] = old_df[col].apply(pd.to_numeric) # Convert argument to a numeric type
df = old_df
df["total_negtive"] = df["total_sad"] + df["total_angry"] #negative reaction is sum of total_sad and total_angry10+9

df.sort_values(by = ["total_negtive" ] , ascending = False) # Sorting based on total_negative
df["impact_ratio"] = df["total_like"]/df["total_negtive"] # It determine the ratio which is total_like by total_negative
df["impact"] = [1 if x < 10 else 0 for x in df["impact_ratio"]] # Assigning the impact based on impact ratio
df  # #visualize data

"""## Removing colums that does not provide any information about the articles"""

df = df.dropna()  # Remove missing values from the dataframe
df = df.drop([ "pubdate" , "fb_wall_count" , "total_negtive","total_like" , "shares" , "visibility" , "total_love" , "total_wow" , "total_haha" , "total_sad" , "total_angry","fb_wall_urls","title","impact_ratio" ] ,axis = 1)
df

"""Once all the non useful columns are droped we are left withthe scopus subject and the publisher subject. We determine if the article have generated negative impact based on the scopus subject"""

df = df.drop(["subjects" , "abstract"] ,axis = 1)
df

data = pd.get_dummies(df) # Convert categorical variable into dummy/indicator variables
data # Visualizing the data

y = data.impact
x = data.drop("impact" , axis = 1)

"""##  For our model dividing all data into test and train set."""

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42) # #Splitting the data in 3: 7 ratio so that we can check accuracy

model = LogisticRegression() #create a obect of Logic Regression
model.fit(X_train, y_train)  #Training the model

model.score(X_train, y_train) #Finding scores of model on training data

y_pred = pd.Series(model.predict(X_test)) #Predicting test data
y_test = y_test.reset_index(drop=True)
z = pd.concat([y_test, y_pred], axis=1)
z.columns = ['True', 'Prediction'] 
z.head() # Visualizing the predicted value with the true value

from sklearn.metrics import classification_report
report = classification_report(y_test,y_pred)  # Making report of accuracy of our model
print(report)  # Making accuracy metrics for f1-score and recall

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

"""#### Therfore we got accuracy  = 0.98 means 98 percent accurate results for our model"""

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('True')
plt.xlabel('Predicted')

def get_agg_data(col_name):
    d = {}
    columns = ["total_like" , "shares" , "visibility" , "total_love" , "total_wow" , "total_haha" , "total_sad" , "total_angry"]
    err = 0
    for i ,row in old_df[:].iterrows(): # iterating over a pandas Data frame rows in the form of (index, series) pair
        gby = row[col_name]
        
        try:
            if(d.get(gby) == None ):
                d[gby]={}
                for col in columns:
                    d[gby][col] = int(row[col])
                continue
            for col in columns:
                d[gby][col] += int(row[col])
        except:
            err+=1
    return err , d

err , d = get_agg_data("publisher_subjects") # getting the agg data based on publisher subjects
new_df = pd.DataFrame.from_dict(d).T # storing the value of dictionary in new_df

new_df["total_negtive"] = new_df["total_sad"] + new_df["total_angry"] # calculating total_negative based on total_sad and total_angry reaction

new_df["impact"] = [1 if x > 100 else 0 for x in new_df["total_negtive"]] # calculating the impact for the new data frame
new_df

sns.countplot(x="impact",data=df,palette="hls")

"""## Top 5 article group by  publisher_subject

- Based on angry react
"""

new_df.sort_values(by ='total_angry' , ascending = False ).head()

"""- Based on like react"""

new_df.sort_values(by ='total_like' , ascending = False ).head()

"""- Based on total_shares"""

new_df.sort_values(by ='shares' , ascending = False ).head()

"""- Based on visibility"""

new_df.sort_values(by ='visibility' , ascending = False ).head()

def get_pn_agg_data(col_name):
    d = {} # Creating a dictionary to stroe all the values
    pos_col = ["total_like", "total_love" , "total_wow" , "total_haha" ] # All this reactions are considered as positive column
    neg_col = [ "total_sad" , "total_angry"]  # All this reactions are considered as negative column
    err = 0
    for i ,row in old_df[:].iterrows():
        gby = row[col_name] # Storing  the row of col name in variable name gby
        
        try:
            if(d.get(gby) == None ):
                d[gby]={}
                for col in pos_col:
                    d[gby]["pos"] = int(row[col]) # If the col name is present in pos_col then store the int value of row
                for col in neg_col:
                    d[gby]["neg"] = int(row[col]) # If the col name is present in neg_col then store the int value of row
                continue
            for col in pos_col:
                d[gby]["pos"] += int(row[col]) # Incrementing the value as more values is found in pos_col and store it
            for col in neg_col:
                d[gby]["neg"] += int(row[col]) # Incrementing the value as more values is found in neg_col and store it
        except:
            err+=1
    return err , d

err , d = get_pn_agg_data("publisher_subjects") # getting the agg data based on publisher subjects
new_df = pd.DataFrame.from_dict(d).T # storing the value of dictionary in new_df

new_df

"""## 5 Articles that have positive impact based on the reaction by the user on Facebook """

new_df.sort_values(by ='pos' , ascending = False ).head()

"""## 5 Articles that have negative impact based on the reaction by the user on Facebook """

new_df.sort_values(by ='neg' , ascending = False ).head()

