#!/usr/bin/env python
# coding: utf-8

# ## Predicting heart disease using machine learing
# this notebook looks into using various python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.
# 
# We're going to take the following approach:
# 1.Problem definition
# 2.Data
# 3.Evaluation
# 4.Features
# 5.Modelling 
# 6.Experimentation
# 
# ## 1. Problem Definition 
# 
# In a statement ,
# > Given clinical parameters about a patient, can we predict whether or not they have heart disease?
# 
# ## 2. Data 
# 
# The original data came from the cleavland data from the UCI Machine Learning Repoitory.https://archive.ics.uci.edu/dataset/45/heart+disease
# 
# There is also a version of it available on Kaggle. https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# 
# ## 3. Evaluation 
# 
# > If we can reach 95% accuracy at predicting whether or not a patient is having a heart disease or not during the proof of concept , we'll pursue the project.
# 
# ## 4. Features 
# This is where you'll get different information about each of teh features in your data.
# 
# **Create data dictionary**
# 
# 1. age - age in years
# 2. sex - (1 = male; 0 = female)
# 3. cp - chest pain type
#     0: Typical angina: chest pain related decrease blood supply to the heart
#     1: Atypical angina: chest pain not related to heart
#     2: Non-anginal pain: typically esophageal spasms (non heart related)
#     3: Asymptomatic: chest pain not showing signs of disease
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
# 5. chol - serum cholestoral in mg/dl
#     serum = LDL + HDL + .2 * triglycerides
#     above 200 is cause for concern
# 6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#      '>126' mg/dL signals diabetes
# 7. restecg - resting electrocardiographic results
#     0: Nothing to note
#     1: ST-T Wave abnormality
#     can range from mild symptoms to severe problems
#     signals non-normal heart beat
#     2: Possible or definite left ventricular hypertrophy
#     Enlarged heart's main pumping chamber
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more
# 11. slope - the slope of the peak exercise ST segment
#     0: Upsloping: better heart rate with excercise (uncommon)
#     1: Flatsloping: minimal change (typical healthy heart)
#     2: Downslopins: signs of unhealthy heart
# 12. ca - number of major vessels (0-3) colored by flourosopy
#     colored vessel means the doctor can see the blood passing through
#     the more blood movement the better (no clots)
# 13. thal - thalium stress result
#     1,3: normal
#     6: fixed defect: used to be defect but ok now
#     7: reversable defect: no proper blood movement when excercising
# 14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)

# ## Preparing the tools we need 
# 
# We're going to use pandas, Matplotlib and numpy for data analysis and manipulation. 

# In[4]:


# Import all the tools we need

# Regular EDA (exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# we want our plots to appear inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import RocCurveDisplay


# ## Load Data

# In[5]:


df = pd.read_csv("heart-disease.csv")
df


# In[6]:


df.shape
# (rows, colums)


# ## data Exploration (exploratory data analysis or EDA)
# 
# the goal here is to find out more about the data and become a subject matter export on the dataset you're working with.
# 
# 1. What questions are you trying to solve?
# 2. What kind of data do we have and how do we treat different types?
# 3. What's missing from the data and how do you deal with it 
# 4. what are the outliers and why you should create them?
# 5. How can you add, change or remove features to get more out of your data ?

# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


# let's find out how many each class is there 
df["target"].value_counts()


# In[10]:


df["target"].value_counts().plot(kind="bar", color=["salmon","lightblue"]);
# in this plot 1 denotes the number of patients having heart disease and 
# 0 denotes the number of patients who doesnot have a heart disease. 


# In[11]:


df.info()


# In[12]:


# Are there any missing values?
df.isna().sum()


# In[13]:


df.describe()


# ### Heart Disease frequency according to sex

# In[14]:


## here 1 denotes male and 0 denotes female
df.sex.value_counts()


# In[15]:


# compare target column wirh sex column 
pd.crosstab(df.target,df.sex)


# In[16]:


pd.crosstab(df.target, df.sex).plot(kind ="bar",figsize = [10,6], color=["red","lightblue"])

plt.title("heart disease frequency for sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"])


# In[17]:


df.head()


# In[18]:


df["thalach"].value_counts()


# ## age vs max heart rate for heart disease
# 

# In[19]:


# create another figure
plt.figure(figsize=(10,6))

# scatter with positive examples 
plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           c="salmon")
# scatter with negative examples 
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           c="lightblue")

# add some help info 
plt.title("heart disease in function of age and max heart rate")
plt.xlabel("age")
plt.ylabel("max heart rate")
plt.legend(["disease","no disease"]);


# In[20]:


# check the distribution of the age column with a histogram
df.age.plot.hist();


# ## heart disease frequency per chest pain type
# cp - chest pain type:
#     0: Typical angina: chest pain related decrease blood supply to the heart
#     1: Atypical angina: chest pain not related to heart
#     2: Non-anginal pain: typically esophageal spasms (non heart related)
#     3: Asymptomatic: chest pain not showing signs of disease

# In[21]:


pd.crosstab(df.target,df.cp)


# In[22]:


# make the crosstab more visual 
pd.crosstab(df.cp, df.target).plot(kind="bar",
                                  figsize=(10,6),
                                  color=["salmon","lightblue"])

# Add some communication
plt.title("heart disease frequency per chest pain type")
plt.xlabel("Chest Pain Type")
plt.ylabel("amount")
plt.legend(["No Disease", "Disease"])


# In[23]:


df.head()


# In[24]:


# Make a corelation matrix 
df.corr()


# In[25]:


# corelation matrix more understanding 
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize = (15,10))
ax = sns.heatmap(corr_matrix,
                annot = True,
                linewidths=0.5,
                fmt=".2f",
                cmap="YlGnBu")
# bottom, top = ax.get_


# In[26]:


df.head()


# ## 5. modeling 

# In[27]:


df.head()


# In[28]:


# split data into X and y
X = df.drop("target", axis=1)

y = df["target"]


# In[29]:


X


# In[30]:


y


# In[31]:


# Split data into train and test sets 
np.random.seed(42)

# split data 
X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.2)


# In[32]:


y_train, len(y_train)


# No we've split our data into train and test .
# Now its time to build a machine learning model.
# 
# We'll train it(find the patterns) on the training set.
# And we'll test it(use the patterns) on the test set.
# 
# We're going to try 3 different machine learning models:
# 1. Logistic Regression
# 2. K-nearest Neighbours 
# 3. Random Forest Classifier

# In[33]:


# Put models in a dictionary 
models = {"logistics Regressions": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier()}

# Create a Function to fit and score models 
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-learn machine learning models 
    X_train: training data
    X_test: testing data
    y_train: traing labels 
    y_test: test labels 
    """
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models 
    for name, model in models.items():
            # Fit the model to the data 
            model.fit(X_train, y_train)
            # evaluate the model
            model_scores[name] = model.score(X_test,y_test)
    return model_scores      


# In[34]:


model_scores = fit_and_score(models = models,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test)
model_scores 


# ## Model Comparison

# In[35]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();


# Now we've got a baseline model... and we know a model's first predictions aren't always what we should based on our next steps off. what we should do ?
# 
# Lets 's look at the following:
# * Hyperparameter tuning //these two are part of almost any  
# * Feature importance // machine learing modal that we are   working on
# * confusion matrix -
# * cross validation 
# * precision
# * Recall
# * F1 score 
# * classification report
# * ROC Curve
# * Area under curve (AUC)
# 
# ## Hyperparameter tuning 
# 

# In[36]:


## Let's tune knn

train_scores = []
test_scores = []

# create a list of different values for n-neighbors
neighbors = range(1,21)

# Setup KNN instance 
knn = KNeighborsClassifier()

# Loop through different n_neighbors 
for i in neighbors:
    knn.set_params(n_neighbors = i)
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    # update the trainig scores list
    train_scores.append(knn.score(X_train, y_train))
    
    #update the test scores list
    test_scores.append(knn.score(X_test, y_test))


# In[37]:


train_scores


# In[38]:


test_scores


# In[39]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1,21,1))
plt.xlabel("Number pf neighbors")
plt.ylabel("model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# ## Hyperparameter tuning with randomizedSearchcv
# we're going to tune:
# * LogisticRegression()
# * RandomForestClassifier()
# 
# ... using RandomizedSearchCV

# In[40]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4,4,20),
               "solver": ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10,1000, 50),
          "max_depth": [None, 3,5,10],
          "min_samples_split": np.arange(2,20,2),
          "min_samples_leaf": np.arange(1,20,2)}


# In[41]:


np.arange(10,1000,50)


# Now we've got hyperparameter grids for each of our models, let's tune them using RandomizedSearchCV...

# In[42]:


# tune logisticRegression

np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions = log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)


# In[43]:


rs_log_reg.best_params_


# In[44]:


rs_log_reg.score(X_test, y_test)


# Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier()...

# In[45]:


# Setup random seed 
np.random.seed(42)

# Setup random hyperprameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                          param_distributions=rf_grid,
                          n_iter=20,
                          verbose=True)
# Fit randomhyperparameter search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)


# In[46]:


# Find the best hyperparameters
rs_rf.best_params_


# In[47]:


# Evaluate teh randomized search RandomiforestClassifier model
rs_rf.score(X_test,y_test)


# In[48]:


model_scores


# 1. by hand
# 2. randomforestclassifier
# 3. gridsearchCv

# ## Hyperparameter tuning with GridSearchCV
# 
# Since our Logistic Regression model provides the best scores so far, we'll try and improve them again using GridSearchCV...

# In[49]:


# Different hyperparameters for our LogisticRegression model
log_reg_grid = {"C": np.logspace(-4, 4, 30),
               "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
# here CV stands for cross-validation
gs_log_reg = GridSearchCV(LogisticRegression(),
                         param_grid=log_reg_grid,
                         cv=5,
                         verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train);


# In[50]:


# Check the best hyperparameters 
gs_log_reg.best_params_


# In[51]:


# Evaluate the grid search LogisticRegression model
gs_log_reg.score(X_test, y_test)


# In[52]:


model_scores


# ## Evaluating aur machine learning classifier , beyond accuracy 
# * ROC curve and AUC score
# * Confusion matrix 
# * Classification report 
# * Precision 
# * Recall 
# * F1-score
# 
# ...and it would be great if cross-validation was used where possible.
# 
# To make comparisons and evaluate our trained model, first we need to make predictions.

# In[53]:


# Make predictions with tuned model
y_preds = gs_log_reg.predict(X_test)


# In[54]:


y_preds


# In[55]:


y_test


# In[56]:


# roc curve is a mertric or graph of estimating hoe your model is performing by comparing true positive with false positive 


# ![true_positive.png](attachment:true_positive.png)

# In[57]:


RocCurveDisplay.from_estimator(gs_log_reg, X_test, y_test)


# In[58]:


# Confusion matrix 
print(confusion_matrix(y_test,y_preds))


# In[59]:


sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    Plots a nice looking confusion matrix using Seaborns's heatmap
    """
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test,y_preds),
                    annot=True,
                    cbar=False)
    plt.xlabel("True label")
    plt.ylabel("predicted label")
    
plot_conf_mat(y_test, y_preds)


# Now we've got a ROC curve, an AUC metric and a confusion matrix.let's get a classification report as well as cross-validation precision, recall and f1-score.
# 

# In[60]:


print(classification_report(y_test,y_preds))


# ## calculate the evaluation metrics using cross-validation
# 
# We're going to calculate precision, recall and f1-score of our model using cross-validation and to do so we'll be using 'cross_val_score()'.

# In[61]:


# check best hyperparameters
gs_log_reg.best_params_


# In[62]:


# create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                        solver="liblinear")


# In[66]:


# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="accuracy")
cv_acc


# In[69]:


cv_acc = np.mean(cv_acc)
cv_acc


# In[71]:


# Cross-validated precision
cv_precision = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="accuracy")
cv_precision = np.mean(cv_precision)
cv_precision


# In[75]:


# Cross-validated recall
cv_recall = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="recall")
cv_recall = np.mean(cv_recall)
cv_recall


# In[76]:


# Cross-validated f1-score
cv_f1 = cross_val_score(clf,
                        X,
                        y,
                        cv=5,
                        scoring="f1")
cv_f1 = np.mean(cv_f1)
cv_f1


# In[78]:


#Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                          "Precision": cv_precision,
                          "Recall": cv_recall,
                          "F1": cv_f1},
                         index=[0])
cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                     legend=False)


# ### Feature Importance 
# 
# Feature importance is another way of asking, "which features contributed most to teh outcomes of the model and how did they contribute?"
# 
# Finding feature importance is different for each machine learning model.One way to find feature importance is to search for "(MODEL NAME) feature importance".
# 
# Let's find the feature importance for our LogisticRegression model....
# 

# In[82]:


df.head()


# In[84]:


# Fit an instance of LogisticRegression
gs_log_reg.best_params_

clf = LogisticRegression(C=0.20433597178569418,
                        solver="liblinear")

clf.fit(X_train, y_train);


# In[86]:


df.head()


# In[85]:


# Check coef_
clf.coef_


# In[87]:


# Match coef's of features to columns 
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[89]:


# Visualize feature importance 
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance",legend = False);


# In[90]:


pd.crosstab(df["sex"], df["target"])


# In[93]:


pd.crosstab(df["slope"], df["target"])


# slope - the slope of the peak exercise ST segment 
# * 0: Upsloping: better heart rate with excercise (uncommon) 
# * 1: Flatsloping: minimal change (typical healthy heart) 
# * 2: Downslopins: signs of unhealthy heart

# ## 6. Experimentation 
# 
# If you haven't hit your evaluation metric yet... ask yourself....
# 
# * Could you collect more data?
# * Could you try a better model? Like CatBoost or XGBoost?
# * Could you improve the current models? (beyond what we've done so far)
# * If your model is good enough (you have hit your evaluation metric)
# * how would you export or share it with others?

# In[ ]:


* you can find more classification machine learning dataset from google
   just search classification machine learning dataset
   kaggle or kdnuggets

