import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.feature_selection import RFE

titanic_nm = pd.read_csv('titanic3.csv')

print('======================= Quick Look At DataTypes ========================')
print('Data labeled as “non-null object” from the “info()” function output can be \nconsidered as text based, and we need to figure out what type of text it is.')
print('======================================================================')
print(titanic_nm.info())
print('====================== Summary of Quantitative Data =================================')
print(titanic_nm.describe())
print('====================== Counting Cabin Repeats =================================')
titanic_count  =titanic_nm.groupby('cabin')['cabin'].count().reset_index(name = "Group Count") 
sort_titanic_count = titanic_count.sort_values('Group Count',ascending=False).head(10)
print(sort_titanic_count)
print('====================== Counting Name Repeats =================================')
titanic_name_count = titanic_nm.groupby('name')['name'].count().reset_index(name = "Name Count")
sort_titanic_name = titanic_name_count.sort_values('Name Count',ascending=False).head(10)
print(sort_titanic_name)
print('====================== Cut Out First Character From Cabin Name =================================')
titanic_nm['cabin'] = titanic_nm['cabin'].replace(np.NaN, 'U')
titanic_nm['cabin'] =[ln[0] for ln in titanic_nm['cabin'].values]
titanic_nm['cabin'] = titanic_nm['cabin'].replace('U', 'Unknown')
print(titanic_nm['cabin'].head())
print('====================== Counting Cabin First Letter Groups =================================')
titanic_cabin_count = titanic_nm.groupby('cabin')['cabin'].count().reset_index(name ="Cabin Letter Count")
sort_titanic_cabin_count = titanic_cabin_count.sort_values('Cabin Letter Count', ascending=False).head(10)
print(sort_titanic_cabin_count)
print('====================== Extract title name ==============================================')
titanic_nm['title'] = [ln.split()[1] for ln in titanic_nm['name'].values]
titanic_nm['title'].value_counts()
titanic_nm['title'] = [title if title in ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Dr.', 'Rev.'] else 'Unknown'
                       for title in titanic_nm['title'].values ]
print(titanic_nm['title'].head())
print('')

titanic_nm['isfemale'] = np.where(titanic_nm['sex'] == 'female', 1, 0)

titanic_nm = titanic_nm[[f for f in list(titanic_nm) if f not in ['sex','name','boat','body','ticket','home.dest']]]
titanic_nm['pclass'] = np.where(titanic_nm['pclass'] ==1,'First',
	np.where(titanic_nm['pclass'] == 2, 'Second','Third'))

titanic_nm['embarked'] = titanic_nm['embarked'].replace(np.NaN, 'Unknown')

print(titanic_nm.head())
print('')

print('=============================== Creating Dummy Features ===========================================')
print('We need to turn categorical features into a numerical form so our models can use them.')
#Creating fummy features
print(pd.get_dummies(titanic_nm['cabin'], columns=['cabin'], drop_first=False).head(10)) 

def prepare_data_for_model(raw_dataframe, target_columns, drop_first = True, make_na_col = True):
	dataframe_dummy = pd.get_dummies(raw_dataframe, columns=target_columns,
		drop_first=drop_first,
		dummy_na=make_na_col)
	return (dataframe_dummy)

titanic_ready = prepare_data_for_model(titanic_nm, target_columns=['pclass','cabin','embarked','title'])
titanic_ready = titanic_ready.dropna()
print(list(titanic_ready))

print('============================== Split data into train and test portions ============================')
features = [feat for feat in list(titanic_ready) if feat != 'survived']
X_train, X_test, y_train, y_test = train_test_split(titanic_ready[features],
	titanic_ready[['survived']], test_size=0.5, random_state=42)
print(X_train.head(3))
print(y_train.head(3))

lr_model = LogisticRegression()
lr_model.fit(X_train,y_train.values.ravel())

coefs = pd.DataFrame({'Feature':features, 'Coef':lr_model.coef_[0]})
positive_coef_sort = coefs.sort_values('Coef',ascending=False).head(7)
negative_coef_sort = coefs.sort_values('Coef',ascending=False).tail(7)
print('=============================== Positive Features ================================================') 
print(positive_coef_sort)
print('=============================== Negative Features ================================================')
print(negative_coef_sort)

print("=============================== Test Case: Predict Fictional Survival ============================")
print("Create your fictional traveler and predict survival rate")
#Edit here accordingly
x_predict_pclass = 'First' 
x_predict_is_female=1  
x_predict_age=26 
x_predict_sibsp=3 
x_predict_parch = 0  
x_predict_fare = 200  
x_predict_cabin = 'A'  
x_predict_embarked = 'Q'  
x_predict_title = 'Mrs.'  

# make a copy of the original data set in order to create dummy categories that are the same as seen on 
# original data
titanic_nm_tmp = titanic_nm.copy()
# add new row to titanic df
titanic_nm_tmp = titanic_nm_tmp[['pclass', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked', 'title', 'isfemale', 'survived']] 

titanic_fictional_nm = pd.DataFrame([[x_predict_pclass, 
                                     x_predict_age,
                                     x_predict_sibsp,
                                     x_predict_parch,
                                     x_predict_fare,
                                     x_predict_cabin,
                                     x_predict_embarked,
                                     x_predict_title,
                                     x_predict_is_female,
                                     0]], columns = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked', 'title', 'isfemale', 'survived']) 

# titanic_df_tmp.append(titanic_fictional_df)
titanic_nm_tmp = pd.concat([titanic_fictional_nm, titanic_nm_tmp], ignore_index=True)
# titanic_df_tmp = pd.get_dummies(titanic_df_tmp)
titanic_nm_tmp = prepare_data_for_model(titanic_nm_tmp, target_columns=['pclass', 'cabin', 'embarked', 'title'])

Y_pred = lr_model.predict_proba(titanic_nm_tmp[features].head(1))
probability_of_surviving_fictional_character = Y_pred[0][1] * 100
print('Probability of surviving Titanic voyage: %.2f percent' % probability_of_surviving_fictional_character)

average_survival_rate = np.mean(titanic_nm['survived']) * 100
fig = plt.figure()
objects = ('Average Survival Rate', 'Fictional Traveler')
y_pos = np.arange(len(objects))
performance = [average_survival_rate, probability_of_surviving_fictional_character]
 
ax = fig.add_subplot(111)
colors = ['gray', 'blue']
plt.bar(y_pos, performance, align='center', color = colors, alpha=0.5)
plt.xticks(y_pos, objects)
plt.axhline(average_survival_rate, color="r")
plt.ylim([0,100])
plt.ylabel('Survival Probability')
plt.title('How Did Your Fictional Traveler Do? \n ' + str(round(probability_of_surviving_fictional_character,2)) + '% Chance of Surviving!')
 
plt.show()
