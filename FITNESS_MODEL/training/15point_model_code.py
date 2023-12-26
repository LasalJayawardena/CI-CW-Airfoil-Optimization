#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# In[1]:


import pandas as pd
import numpy as np
d4_p15_df = pd.read_csv('./Datasets/NACA4Digit_Dataset15Point.csv')
d4_p15_df


# In[2]:


d5_p15_df = pd.read_csv('./Datasets/NACA5Digit_Dataset15Point.csv')
d5_p15_df


# In[3]:


df = pd.concat([d4_p15_df, d5_p15_df], axis=0, ignore_index=True)
df


# In[4]:


# Identify duplicate rows
df = df.drop('Unnamed: 0', axis=1)
duplicates = df.duplicated()

# Filter duplicate rows
duplicate_rows = df[duplicates]

df = df.drop_duplicates()
df


# In[5]:


df = df.dropna()


# In[6]:


df


# In[7]:


# df.to_csv('NACA_4digit_and_5digit_15point_concat.csv', index=False)


# In[7]:


from tqdm import tqdm

def assign_airfoil_ids(df):
    '''
    Takes in a dataframe and returns the dataframe with airfoil ids
    '''
    # Extract y-coordinate columns
    y_columns = df.columns[df.columns.str.startswith('y')]

    # Create a new column with a unique identifier for each airfoil
    df['Airfoil_No'] = pd.factorize(df[y_columns].apply(tuple, axis=1))[0] # using the factorize function from pandas to convert the tuples into integer IDs.

    df = pd.concat([df[['Airfoil_No']], df.drop(columns=['Airfoil_No'])], axis=1)
    
    # Count the number of unique airfoils
    num_airfoils = df['Airfoil_No'].nunique()

    print(f"Number of airfoils: {num_airfoils}")

    return df


# In[8]:


df = assign_airfoil_ids(df)
df


# In[9]:


df['Airfoil_No'].unique()


# In[11]:


from sklearn.model_selection import train_test_split


# Perform train/test split based on 'Airfoil_No'
train_indices, test_indices = train_test_split(df['Airfoil_No'].unique(), test_size=0.2, random_state=42)

# Extract training and testing data based on the split
train_df = df[df['Airfoil_No'].isin(train_indices)]
test_df = df[df['Airfoil_No'].isin(test_indices)]

# Print the lengths of the training and testing sets
print(f"Train set length: {len(train_df)}")
print(f"Test set length: {len(test_df)}")


# In[12]:


train_df['Airfoil_No'].unique(), test_df['Airfoil_No'].unique()


# In[13]:


# train_data['Airfoil_No'].unique(), test_data['Airfoil_No'].unique()
train_df


# In[14]:


X_train = train_df.drop(['Airfoil_No','Cl', 'Cd', 'Cm'], axis=1).values
y_train = train_df[['Cl', 'Cd', 'Cm']].values


X_test = test_df.drop(['Airfoil_No', 'Cl', 'Cd', 'Cm'], axis=1).values
y_test = test_df[['Cl', 'Cd', 'Cm']].values


# In[15]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ### Min Max Scaling

# In[16]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train[:2]


# In[17]:


X_train.shape


# # Model Training

# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


model = Sequential()

model.add(Dense(33, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(33, activation = 'relu'))

model.add(Dense(3))

model.compile(optimizer='adam', loss='mse')


# In[19]:


model.fit(x=X_train, y=y_train, epochs=20, validation_data=(X_test, y_test))


# In[20]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[21]:


predictions = model.predict(X_test)
predictions


# In[22]:


y_test


# In[23]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

print('explained_varience_score : ', explained_variance_score(y_test, predictions))
print('mean_absolute_error      : ', mean_absolute_error(y_test, predictions))
print('mean_squared_error       : ', mean_squared_error(y_test, predictions))
print('root_mean_squared_error  : ', np.sqrt(mean_squared_error(y_test, predictions)))


# ### Saving Models and Scalers

# In[24]:


# # SAVING THE MODEL
# model.save('./model3.h5')


# In[25]:


# # SAVING THE SCALER
# import joblib
# joblib.dump(scaler, './min_max_scaler_15point.pkl')


# ## Model Inference

# In[1]:


import tensorflow as tf
import joblib

min_max_scaler = joblib.load('./min_max_scaler_15point.pkl')

# Define Constants
DRAG_MODEL = tf.keras.models.load_model('./model3.h5')
REYNOLDS_NUMBER = 100000
MACH_NUMBER = 0.1
ATTACK_ON_ANGLE = -10

def model_inference_func() -> float: 
    
    yCoorUpper = [2.07692721e-02,  3.93104182e-02,  5.46992442e-02,
        6.60399430e-02,  7.26857201e-02,  7.44701301e-02,  7.17682176e-02,
        6.53878200e-02,  5.63418071e-02,  4.57425800e-02,  3.45854691e-02,
        2.37791669e-02,  1.41758894e-02,  6.57918271e-03,  1.68970028e-03]
    yCoorLower = [-2.07692721e-02, -3.93104182e-02, -5.46992442e-02, -6.60399430e-02,
       -7.26857201e-02, -7.44701301e-02, -7.17682176e-02, -6.53878100e-02,
       -5.63418071e-02, -4.57425800e-02, -3.45854691e-02, -2.37791669e-02,
       -1.41758894e-02, -6.57918271e-03, -1.68970028e-03]

    features = [yCoorUpper + yCoorLower + [REYNOLDS_NUMBER, MACH_NUMBER, ATTACK_ON_ANGLE]]
    
    features = min_max_scaler.transform(features)
    
    features = tf.convert_to_tensor(features, dtype=tf.float32)

    # Predict the drag coefficient
    pred = DRAG_MODEL.predict(features)[0]
    cl, cd, cm = pred[0], pred[1], pred[2]

    return cl


model_inference_func()


# In[33]:


test_df.iloc[0]


# In[ ]:




