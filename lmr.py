# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:41:03 2018

@author: adithya
"""
#importing required libraries
import pandas as pd
import numpy as np

#importing datasets
Cust_Trans = pd.read_excel("Customer_Transaction.xlsx")
test = pd.read_excel('Test_Set.xlsx')
#sorting by id and code
Cust_Trans.sort_values(["Customer_ID","Store_Code"], axis = 0, ascending = True, 
                 inplace = True)
# reordering 
Cust_Trans = Cust_Trans[['Customer_ID','Store_Code','Territory', 'Business', 'Year', 'Week',  'City_Name',
                         'Store_Type', 'Transaction_Type', 'Return_Reason', 
                         'Invoices', 'Item_Count', 'Revenue', 'Discount', 'Units_Sold']]
#removing all returned transactions
Cust_Trans = Cust_Trans[Cust_Trans.Transaction_Type != "Return"]
#removing extra coloumn
Cust_Trans.drop(['Return_Reason'],axis=1,inplace=True)
#removing duplicates
duplicated = Cust_Trans.drop_duplicates(['Customer_ID','Store_Code'])
duplicated.sort_values(["Customer_ID","Store_Code"], axis = 0, ascending = True, 
                 inplace = True)

#importing reamining data
CD = pd.read_excel("Customer_Demographics.xlsx")
SM = pd.read_excel("Store_Master.xlsx")
SM.sort_values(["Store_Code"], axis = 0, ascending = True, 
                 inplace = True)
#test data
test = test.merge(SM,on="Store_Code",how='left')

#dropping test data from train data
SM.drop(index=[31,32,33,34],inplace=True)


id = CD['Customer_ID']
code = SM["Store_Code"]

# Create a list to store the data
grades = []

# For each row in the column, yet to remove test store codes
for ids in id:
    for codes in code:
        grades.append(str(ids)+" "+str(codes))
 #creating df from list       
full = pd.DataFrame(grades, columns=['id'])      
#seperator  
full = pd.DataFrame(full.id.str.split(' ',1).tolist(),
                                   columns = ["Customer_ID","Store_Code"])

du = duplicated[["Customer_ID","Store_Code"]]

#preparing to compare
full_m =  full["Customer_ID"].astype(str) + full["Store_Code"]
du_m = du["Customer_ID"].astype(str) + du["Store_Code"].astype(str)


            
#matching customerid with store where purchased
full['purchase'] = full_m.isin(du_m).astype(np.int8)
full["Customer_ID"] = full["Customer_ID"].apply(int)

#merging customer demographics and dropping common data
full = full.merge(CD,on="Customer_ID",how='left')
full = full.drop(columns = ['Territory','Language','First_txn_dt', 'Last_accr_txn_dt', 'Last_rdm_txn_dt','Age'])
#adding customer info to data 
test = test.merge(CD,on="Customer_ID",how='left')
test = test.drop(columns = ['Territory_x','Territory_y',
                            'Language','First_txn_dt', 'Last_accr_txn_dt', 'Last_rdm_txn_dt','Age'])

#preparing for age of customer
full["Birth_date"]= full["Birth_date"].str.split(":", n = 0, expand = True) 
full['Birth_date'] = full.Birth_date.str[5:9]
full["Birth_date"] = full["Birth_date"].fillna(0)
full["Birth_date"] = full["Birth_date"].apply(int)
full["Birth_date"]= full["Birth_date"].mask(full["Birth_date"] == 0,full["Birth_date"].mean())
full['age'] = (2018-full['Birth_date'].apply(int))
full = full.drop(columns = ['Birth_date'])

#preparing age for test data
test["Birth_date"]= test["Birth_date"].str.split(":", n = 0, expand = True) 
test['Birth_date'] = test.Birth_date.str[5:9]
test["Birth_date"] = test["Birth_date"].fillna(0)
test["Birth_date"] = test["Birth_date"].apply(int)
test["Birth_date"]= test["Birth_date"].mask(test["Birth_date"] == 0,test["Birth_date"].mean())
test['age'] = (2018-test['Birth_date'].apply(int))
test = test.drop(columns = ['Birth_date'])

#merging store master
full['Store_Code'] = full['Store_Code'].apply(int)
full = full.merge(SM,on="Store_Code",how='left')
full.drop(['Nationality','Territory', 'Business','Region','Customer_Count','Store_Name','Mall_Name',  
           'Train_Test_Store','Geo_Field'],axis=1,inplace=True)
#dropping same for test sets
test.drop([ 'Nationality','Business','Region','Customer_Count','Store_Name','Mall_Name',  
           'Train_Test_Store','Geo_Field'],axis=1,inplace=True)
#label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
full['Marital_Status'] = labelencoder_X.fit_transform(full['Marital_Status'])
test['Marital_Status'] = labelencoder_X.transform(test['Marital_Status'])
full['Gender'] = labelencoder_X.fit_transform(full['Gender'])
test['Gender'] = labelencoder_X.transform(test['Gender'])
full['Loyalty_Status'] = labelencoder_X.fit_transform(full['Loyalty_Status'])
test['Loyalty_Status'] = labelencoder_X.transform(test['Loyalty_Status'])
full['Store_Format'] = labelencoder_X.fit_transform(full['Store_Format'])
test['Store_Format'] = labelencoder_X.transform(test['Store_Format'])


#droping unwanted column
full = full.drop(columns = ['Store_Launch_Date'])
test = test.drop(columns = ['Store_Launch_Date'])

#adjusting income range
income = {'Below 5000':'2500','5001 to 10000': '7500','10001 to 20000': '15000',
          '20001 to 30000': '25000','30001 & Above': '40000','Unspecified':'0','Unknown':'0'}
full['Income_Range'] = full['Income_Range'].map(income)
full["Income_Range"] = full["Income_Range"].apply(int)
full["Income_Range"]= full["Income_Range"].mask(full["Income_Range"] == 0,full["Income_Range"].mean())
#test set
test['Income_Range'] = test['Income_Range'].map(income)
test["Income_Range"] = test["Income_Range"].apply(int)
test["Income_Range"]= test["Income_Range"].mask(test["Income_Range"] == 0,test["Income_Range"].mean())

#converting region code to object
full['Region_Code'] = full['Region_Code'].astype(str)
test['Region_Code'] = test['Region_Code'].astype(str)
#creating Y_train
Y = full['purchase']
full.drop(columns=['purchase'],axis=1,inplace=True)
#filling mean for null values
test['Points']=test['Points'].fillna(test['Points'].mean())
full['Points']=full['Points'].fillna(full['Points'].mean())

#storing id and store code
full_id = full[['Customer_ID','Store_Code']]
test_id = test[['Customer_ID','Store_Code']]

#dropping ids
full.drop(columns=['Customer_ID','Store_Code' ],axis=1,inplace=True)
test.drop(columns=['Customer_ID','Store_Code' ],axis=1,inplace=True)

#one hot encoding
full= pd.get_dummies(full,drop_first=True)
test= pd.get_dummies(test,drop_first=True)
final_train, final_test = full.align(test, join='inner', axis=1)  # inner join

#converting to numpy array
final_train = final_train.values
final_test = final_test.values
Y = Y.values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
final_train = sc.fit_transform(final_train)
final_test = sc.transform(final_test)

#checking for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
final_train = pca.fit_transform(final_train)
final_test = pca.transform(final_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Naive Bayes to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(final_train, Y)

y_pred = classifier.predict(final_test)

#preparing submission file
t_pred = pd.DataFrame(y_pred, columns = ['predictions'])
t_pred['predictions'] = t_pred['predictions'].astype(float)
submission = pd.concat([test_id,t_pred],axis=1)
                       
#writing to csv
submission.to_csv('submission_1.csv')




