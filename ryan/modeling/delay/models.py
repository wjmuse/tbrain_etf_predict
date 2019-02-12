




class Delay:

    def predict(self, X):
        return X.flatten()[-5:]


class DelayFactory:

    def __call__(self):

        return Delay()

build_fn = DelayFactory()


