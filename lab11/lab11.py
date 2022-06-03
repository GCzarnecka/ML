#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[2]:


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()


# In[3]:


def build_model(n_hidden=1, n_neurons=25, optimizer='sgd', learning_rate=10**(-5), momentum=0): 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=X_train.shape[1:]))
    for i in range(0,n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum=momentum if optimizer=='momentum' else 0, 
                                          nesterov=(optimizer=='nesterov') 
                                          if(optimizer=='nesterov' or optimizer=='sgd' or optimizer=='momentum') 
                                          else tf.keras.optimizers.Adam(learning_rate=learning_rate)),
        metrics=["mean_absolute_error"])
    model.add(tf.keras.layers.Dense(1))
    model.build()
    return model


# In[4]:


import os
root_logdir = os.path.join(os.curdir, "tb_logs")
def get_run_logdir(name, val): 
    import time
    ts = int(time.time())
    return os.path.join(root_logdir, str(ts)+'_'+name+'_'+str(value))


# In[5]:


es = tf.keras.callbacks.EarlyStopping(patience=10,min_delta=1,verbose=1, monitor='mean_absolute_error')
def reset():
    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)


# In[6]:


reset()
name='lr'

value=10**(-6)
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(learning_rate=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k1 = (value, mse, mae)

value=10**(-5)
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(learning_rate=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k2 = (value, mse, mae)

value=10**(-4)
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(learning_rate=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k3 = (value, mse, mae)

pickle.dump([k1,k2,k3], open( name+'.pkl', "wb" ) )


# In[7]:


reset()
name='hl'

value=0
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(n_hidden=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
# mae = mean_absolute_error(y_test, prediction)
# mse = mean_squared_error(y_test, prediction)
k1 = (value, np.NaN, np.NaN)

value=1
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(n_hidden=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k2 = (value, mse, mae)

value=2
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(n_hidden=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k3 = (value, mse, mae)

value=3
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(n_hidden=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k4 = (value, mse, mae)

pickle.dump([k1,k2,k3,k4], open( name+'.pkl', "wb" ) )


# In[8]:


reset()
name='nn'

value=5
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(n_neurons=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k1 = (value, mae, mse)

value=25
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(n_neurons=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k2 = (value, mse, mae)

value=125
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(n_neurons=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k3 = (value, mse, mae)

pickle.dump([k1,k2,k3], open( name+'.pkl', "wb" ) )


# In[144]:


reset()
name='opt'

value='sgd'
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(optimizer=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k1 = (value, mse, mae)

value='nesterov'
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(optimizer=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k2 = (value, mse, mae)

value='momentum'
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(optimizer=value, momentum=0.5)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k3 = (value, mse, mae)

value='adam'
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(optimizer=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k4 = (value, mse, mae)

pickle.dump([k1,k2,k3,k4], open( name+'.pkl', "wb" ) )


# In[146]:


reset()
name='mom'

value=0.1
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(optimizer='momentum', momentum=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k1 = (value, mse, mae)

value=0.5
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(optimizer='momentum', momentum=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k2 = (value, mse, mae)

value=0.9
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(name,value))
model = build_model(optimizer='momentum', momentum=value)
history = model.fit(X_train, y_train, epochs=100, callbacks=[tensorboard_cb, es])
prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
k3 = (value, mse, mae)

pickle.dump([k1,k2,k3], open( name+'.pkl', "wb" ) )


# In[9]:


param_distribs = {
    "model__n_hidden": [2,3],
    "model__n_neurons": list(range(20,30)),
    "model__learning_rate": [10**(-5), 15**(-5), 20**(-5)],
    "model__optimizer": ['sgd','momentum','nesterov', 'adam'],
    "model__momentum": [0.3,0.4,0.5,0.6]
}


# In[10]:


import scikeras
from scikeras.wrappers import KerasRegressor

es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)
keras_reg = KerasRegressor(build_model, callbacks=[es])


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
rnd_search_cv = RandomizedSearchCV(keras_reg,
                                    param_distribs,
                                    n_iter=30,
                                    cv=3,
                                    verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_split=0.1)


# In[14]:


pickle.dump( rnd_search_cv.best_params_, open( "rnd_search.pkl", "wb" ) )


# In[ ]:





# In[ ]:




