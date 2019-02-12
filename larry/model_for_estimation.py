


'''
This model solves the ETF0054 issue (originally, 0054 is very hard to train)

Input Features: (None,20,5)
20: n_days
5: all features into PCA (n_components=5)

NN Architecture: lstm *2 + dense*2 + Repeat vector (vecor*5 for TimeDistributed Dense) + TimeDistributed Dense(1)
Note: choose drop = 0
Epoch: 200

'''





from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import keras


from keras.optimizers import Adam,SGD


from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
from keras.layers import Dropout, TimeDistributed, RepeatVector
from  keras import initializers

from keras.layers import Reshape

from keras.layers import Lambda

input_length, input_dim = 20,5

d = 0.0





init_weight = 'glorot_normal'

main_input = Input(shape=(input_length,input_dim),  name='main_input')
lstm_output = LSTM(128, kernel_initializer = init_weight, return_sequences =True, activation='selu', name='lstm_1')(main_input)
lstm_output = Dropout(d)(lstm_output)

lstm_output = LSTM(164, kernel_initializer = init_weight, activation='selu', name='lstm_2')(lstm_output)
lstm_output = Dropout(d)(lstm_output)


encoder_output = Dense(64, kernel_initializer = init_weight,activation='relu', name='dense-1')(lstm_output)   # careful
encoder_output = Dropout(d)(encoder_output)

encoder_output = Dense(32, kernel_initializer = init_weight,activation='relu', name='dense_2')(encoder_output)   # careful
encoder_output = Dropout(d)(encoder_output)

logit = RepeatVector(5)(encoder_output)


output = TimeDistributed(Dense(1, kernel_initializer = init_weight,activation='sigmoid', name='output'))(logit)
output = Dropout(d)(output)


#loss_type = 'mse'
loss_type = 'binary_crossentropy'





model = Model(inputs=main_input, outputs=output)

adam_op = Adam(lr=0.001)

model.compile(optimizer=adam_op, loss=loss_type, metrics=['accuracy'])




print (model.summary())

'''
Total Parameter: 273K
'''
