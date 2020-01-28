from sklearn.linear_model import LinearRegression
import random

def run(output='model'):
    TRAIN_SET_LIMIT = 1000
    TRAIN_SET_COUNT = 100

    TRAIN_INPUT = list()
    TRAIN_OUTPUT = list()
    for i in range(TRAIN_SET_COUNT):
        a = random.randint(0, TRAIN_SET_LIMIT)
        b = random.randint(0, TRAIN_SET_LIMIT)
        c = random.randint(0, TRAIN_SET_LIMIT)
        op = a + (2*b) + (3*c)
        TRAIN_INPUT.append([a, b, c])
        TRAIN_OUTPUT.append(op)

    model = LinearRegression(n_jobs=-1)
    model.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)


run()