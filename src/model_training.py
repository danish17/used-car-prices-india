import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("datasets/train-data.csv")
df = df.drop(columns='Unnamed: 0')

df['Name'] = df['Name'].apply(lambda x: x.lower())
df['Location'] = df['Location'].apply(lambda x: x.lower())

df['Mileage'] = df['Mileage'].astype(str)
df['Mileage'] = df['Mileage'].apply(lambda x: x.split()[0])

df['Engine'] = df['Engine'].astype(str)
df['Engine'] = df['Engine'].apply(lambda x: x.split()[0])

df['Power'] = df['Power'].astype(str)
df['Power'] = df['Power'].apply(lambda x: x.split()[0])

df = df.drop(columns='New_Price')

from sklearn.preprocessing import LabelEncoder
lbl1 = LabelEncoder()
lbl2 = LabelEncoder()
lbl3 = LabelEncoder()
lbl4 = LabelEncoder()
lbl5 = LabelEncoder()


df['Name'] = lbl1.fit_transform(df['Name'])
df['Location'] = lbl2.fit_transform(df['Location'])
df['Fuel_Type'] = lbl3.fit_transform(df['Fuel_Type'])
df['Transmission'] = lbl4.fit_transform(df['Transmission'])
df['Owner_Type'] = lbl5.fit_transform(df['Owner_Type'])

pickle.dump(lbl1, open("encoders/models/name.enc","wb"))
pickle.dump(lbl2, open("encoders/models/location.enc","wb"))
pickle.dump(lbl3, open("encoders/models/fuel.enc","wb"))
pickle.dump(lbl4, open("encoders/models/trans.enc","wb"))
pickle.dump(lbl5, open("encoders/models/owner.enc","wb"))

df[df.columns[0]] = df[df.columns[0]].astype(float)
df[df.columns[1]] = df[df.columns[1]].astype(float)
df[df.columns[2]] = df[df.columns[2]].astype(float)
df[df.columns[3]] = df[df.columns[3]].astype(float)
df[df.columns[4]] = df[df.columns[4]].astype(float)
df[df.columns[5]] = df[df.columns[5]].astype(float)
df[df.columns[6]] = df[df.columns[6]].astype(float)
df[df.columns[7]] = df[df.columns[7]].astype(float)
df[df.columns[8]] = df[df.columns[8]].astype(float)
df[df.columns[9]] = df[df.columns[9]].astype(float)
df[df.columns[10]] = df[df.columns[10]].astype(float)
df[df.columns[11]] = df[df.columns[11]].astype(float)

df= df.dropna(axis=0)
df = df[df['Power'] != 'null']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler
df['Kilometers_Driven'] = scaler.fit_transform(df[['Kilometers_Driven']])
pickle.dump(scaler, open("models/scale.scl", "wb"))

x_train = df.iloc[:,0:11].values
y_train = df.iloc[:,11].values

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=50)
rf_reg.fit(x_train,y_train)

pickle.dump(rf_reg, open("models/rf.mdl", "wb"))