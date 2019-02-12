import keras
import keras.backend as K  # noqa
from keras.models import Sequential, Model
from keras.layers import (
    Input, Embedding, LSTM, Lambda,
    Dense, Dropout, Activation, Reshape,
    TimeDistributed, RepeatVector, Concatenate
)
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, mean_squared_error

#from .hypers import timestep, window, feature_dim, num_classes, output_step, dropout



class NNFactory(object):

    def __init__(self, input_shape, num_classes, activation='sigmoid', output_activation=None, dropout=0.1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout
        self.output_activation = (output_activation if output_activation is not None 
                                                    else ('sigmoid' if self.num_classes == 1 else 'softmax'))

    def __call__(self, do_compile=True):
        # TODO: should I cache model?
        model = self.create()

        if do_compile:
            return self.compile(model)
        else:
            return model

    def getnet(self, name):
        try:
            return self.__getattribute__(name)
        except Exception:
            raise AttributeError(f'Network "{self.__name__}" has no subnet named "{name}".')

    def create(self):
        raise NotImplementedError

    def compile(self, model, **kwargs):
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
            **kwargs
        )

        return model

    
class EtfLstmFactory(NNFactory):
    def __init__(self, *args, n_lastdays=(5,), output_step=5, recurrent_activation='hard_sigmoid', **kwargs):
        """
        Params:
            input_shape:
            num_classes:
            activation:
            recurrent_activation:
            output_activation:
            dropout:
        """
        super().__init__(*args, **kwargs)
        self.n_lastdays = n_lastdays
        self.output_step = output_step
        self.recurrent_activation = recurrent_activation

    def create(self, optimizer='adam',
               loss=dict(price=mean_squared_error, updown=binary_crossentropy),
               loss_weights=dict(updown=1.0, price=0.2),
               metrics=dict(updown='accuracy', price='mae')):
        x_main_seq = Input(shape=self.input_shape)
        x_lastdays = Input(shape=self.n_lastdays)
        x_statics = Input(shape=(2,))

        ly1 = LSTM(128, return_sequences=True,
                   dropout=self.dropout,
                   activation=self.activation,
                   recurrent_activation=self.recurrent_activation)
        ly2 = LSTM(128, return_sequences=False,
                   dropout=self.dropout,
                   activation=self.activation,
                   recurrent_activation=self.recurrent_activation)
        ly3 = Dense(64, activation=self.activation)
        ly4 = Dropout(self.dropout)
        ly5 = Dense(32, activation=self.activation)
        ly6 = Dropout(self.dropout)

        pipes = [ly1, ly2, ly3, ly4, ly5, ly6]

        h = x_main_seq
        for f in pipes:
            h = f(h)
        # y = h
        merged = Concatenate()([h, x_lastdays, x_statics])
        rep = RepeatVector(self.output_step)

        timed_price = TimeDistributed(Dense(self.num_classes), name='price')
        timed_updown = TimeDistributed(Dense(self.num_classes, activation='sigmoid'), name='updown')

        y_price = timed_price(rep(merged))
        y_updown = timed_updown(rep(merged))
        

        model = Model([x_main_seq, x_lastdays, x_statics], [y_updown, y_price])
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        
        return model


#build_fn = EtfLstmFactory((timestep, feature_dim), num_classes, output_step=output_step, activation='relu', dropout=dropout)



if __name__ == '__main__':
    model = build_fn()
    model.summary()

