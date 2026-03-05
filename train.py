import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    arr = np.loadtxt(filename, delimiter=",", dtype=np.float64)

    X_data = arr[:, :-1]
    y = arr[:, -1]
    
    mean = X_data[:,:].mean(axis=0)
    std = X_data[:,:].std(axis=0)

    X_data[:,:] = (X_data[:,:] - mean) / std
    X = np.column_stack([np.ones(X_data.shape[0]), X_data])
    
    return X, y, mean, std

def gradientDecent(X, y):
    alpha = 0.01
    iters = 15000
    theta = np.zeros(X.shape[1])
    
    history = []
    
    for t in range(iters):
        #Prediction
        prediction = X @ theta
        #Residuals
        residuals = prediction - y
        #Gradient
        gradient = (X.T @ residuals) / X.shape[0]
        #Update theta
        theta -= alpha * gradient
        
        if t % 100 == 0:
            cost = (residuals @ residuals) / (2*X.shape[0]) 
            history.append((t, cost))
    print(theta)     
    return history, theta 
        
def plotCurve(X):
    iterations = [x[0] for x in X]
    costs = [x[1] for x in X]

    plt.figure()
    plt.plot(iterations, costs)
    plt.xlabel('Iteration')
    plt.ylabel('J(θ)')
    plt.title('Loss Curve (Gradient Descent)')
    plt.savefig("loss_curve.png")
    plt.show()

X, y, mean, std = loadData("D3.csv")
X_history, theta = gradientDecent(X,y)

new_array = np.array([
            [1,1,1,1],
            [1,2,0,4],
            [1,3,2,1]])
new_array[:, 1:] = (new_array[:, 1:] - mean) / std
predictions = new_array @ theta 
print(predictions)

X = np.array([
    [1, 232, 33, 402],
    [1, 10, 22, 160],
    [1, 6437, 343, 231],
    [1, 512, 101, 17],
    [1, 441, 212, 55],
    [1, 453, 53, 99],
    [1, 2, 2, 10],
    [1, 332, 79, 154],
    [1, 182, 20, 89],
    [1, 123, 223, 12],
    [1, 424, 32, 15]])
y = np.array([
    [2201],
    [0],
    [7650],
    [5599],
    [8900],
    [1742],
    [0],
    [1215],
    [699],
    [2101],
    [8789]])
theta = np.linalg.pinv(X) @ y
print(theta)
plotCurve(X_history)
