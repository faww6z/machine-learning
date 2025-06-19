import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/fawwaz/Downloads/studytime_scores.csv')

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m * x + b)) ** 2
    total_error /float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -(2/n) * x * (y-(m_now * x + b_now))
        b_gradient += -(2/n) * (y-(m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.0001
epochs = 300

for i in range(epochs):
    if i %50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, df, L)

print(m, b)

plt.scatter(df.studytime, df.score, color = 'black')
plt.plot(list(range(1, 10)), [m * x + b for x in range(1,10)], color = 'red')
plt.show()