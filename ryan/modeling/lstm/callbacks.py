import os
from keras.callbacks import Callback
from collections import OrderedDict
from etf_tools.experimental import etf_overall_score, price_accuracy_score, updown_accuracy_score
# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# import numpy as np

#from .hypers import this_dir

this_dir = '/root/work/twetf/ryan/modeling/lstm/'
PJ = os.path.join
PDIR = os.path.dirname


class DirNotFoundError(FileNotFoundError):
    pass


class EtfCkptLogger(Callback):

    @classmethod
    def create_dirs(cls, *dirs, exist_ok=True, **kwargs):
        for d in dirs:
            os.makedirs(d, exist_ok=exist_ok, **kwargs)

    @classmethod
    def create_csv(cls, path, headers, delimiter=','):
        with open(path, 'w') as f:
            f.write(f'{delimiter.join(headers)}\n')


    def __init__(self, model, tags=OrderedDict(),
                 x_val=None, y_val=None,
                 keep_model=True, metrics_path=None, ckpt_dir=None,
                 skip=10, step=10, delimiter=','):
        self.tags = tags
        self.model = model
        self.x_val = x_val
        self.x_mean, self.x_std = x_val[2][:, :1], x_val[2][:, 1:2]
        self.x_lastdays = x_val[1]

        self.y_val = y_val
        self.skip = skip
        self.step = step
        self.keep_model = keep_model
        self.metrics_path = metrics_path or PJ(this_dir, 'validation', 'metrics.csv')
        self.ckpt_dir = ckpt_dir or PJ(this_dir, 'ckpt')
        self.delimiter = delimiter

        if not os.path.isdir(self.ckpt_dir):
            raise DirNotFoundError(f'{self.ckpt_dir} not exist.')

        if not os.path.isfile(self.metrics_path):
            raise FileNotFoundError(f'{self.metrics_path} not exist.')

    def val_score(self, threshold=0.5, **kwargs):
        print(self.x_val.shape)
        y_pred_updown, y_pred_price = self.model.predict(self.x_val)  # XXX: price

        y_actual_updown, y_actual_price = self.y_val

        y_pred_updown = (1 * (y_pred_updown >= threshold)).astype(int)

        y_pred_price = (y_pred_price * self.x_std) + self.x_mean
        y_actual_price = (y_actual_price * self.x_std) + self.x_mean

        price_scores = [price_accuracy_score(ap.flatten(), pp.flatten()) for ap, pp in zip(y_actual_price, y_pred_price)]
        updown_scores = [updown_accuracy_score(aud.flatten(), pud.flatten()) for aud, pud in zip(y_actual_updown, y_pred_updown)]

        # etf_scores = [etf_overall_score(price_actual=ap.flatten(),
        #                                 price_predict=pp.flatten(),
        #                                 updown_actual=aud.flatten(),
        #                                 updown_predict=pud.flatten()) for ap, pp, aud, pud in zip(y_actual_price, y_pred_price, y_actual_updown, y_pred_updown)]

        return OrderedDict(price_score=sum(price_scores)/len(price_scores), updown_score=sum(updown_scores)/len(updown_scores))

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch < self.skip or epoch % self.step != 0:
            return

        ckpt = OrderedDict(**self.tags, **dict(epoch=epoch), **self.val_score())

        with open(self.metrics_path, 'a') as f:
            f.write(self.delimiter.join([str(v) for v in ckpt.values()]) + '\n')

        self.epoch = epoch

    def on_train_end(self, log):
        if self.keep_model:
            name = '.'.join([f'{k}-{v}' for k, v in self.tags.items()] + [f'epoch-{self.epoch}'])
            self.model.save(PJ(self.ckpt_dir, f'{name}.h5'))

        del self.model
