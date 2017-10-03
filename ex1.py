import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def warnUpExercise():
    print("Running warmUpExercise...\n")
    print("5x5 Identity Matrix:")
    a=np.eye(5)
    print(a)

def computeCost(X,y,theta):
    m=X.shape[0]
    XMatrix=np.mat(X)
    yMatrix=np.mat(y)
    thetaMatrix=np.mat(theta)

    J=1/(2*float(m))*sum((np.array(XMatrix*thetaMatrix-yMatrix))**2)
    return J

def gradientDescent(X,y,theta,alpha,iterations):
    m=len(y)
    J_history=np.zeros((iterations,1))
    theta_s=theta.copy()
    for i in range(iterations):
        theta[0]=theta[0]-(alpha/m)*np.sum(np.mat(X)*np.mat(theta_s)-np.mat(y))
        p1=np.mat(X)*np.mat(theta_s)-np.mat(y)
        p2=X[:,1]*p1
        theta[1]=theta[1]-(alpha/m)*p2
        theta_s=theta.copy()
        J_history[i,:]=computeCost(X,y,theta)
    return theta

def drawJ_vals(theta0_vals,theta1_vals,J_vals):
    X,Y=np.meshgrid(theta0_vals,theta1_vals)
    fig=plt.figure(figsize=(8,6))
    ax=fig.gca(projection='3d')
    surf=ax.plot_surface(X, Y, J_vals, cmap=cm.coolwarm)
    plt.show()

def drawJ_valsContour(theta0_vals,theta1_vals,J_vals):
    X,Y=np.meshgrid(theta0_vals,theta1_vals)
    fig=plt.figure()
    plt.contour(X,Y,J_vals,np.logspace(-2,3,20))
    plt.show()

def Plotting(x,y,theta):
    f2 = plt.figure(2)
    p1 = plt.scatter(x, y, marker='x', color='r', label='Training Data', s=30)

    x1 = np.linspace(0, 25, 30)
    y1 = theta[0] + theta[1] * x1

    plt.plot(x1, y1, label="Test Data", color='b')

    plt.legend(loc='upper right')
    plt.show()

if __name__=="__main__":
    #=======================Part 1:Basic Function========================
    warnUpExercise()

    #=========================Part 2:Plotting============================
    print("Plotting Data...\n")
    fr=open('ex1data1.txt')
    arrayLines=fr.readlines()
    numberOfLines=len(arrayLines)
    x=np.zeros((numberOfLines,1))
    y=np.zeros((numberOfLines,1))
    index=0
    for line in arrayLines:
        line = line.strip()
        listFormLine = line.split(",")

        x[index, :] = listFormLine[:1]
        y[index] = listFormLine[-1]
        index += 1

    #====================Part 3:Gradient Descent=============================
    print("Running Gradient Descent...\n")
    columnOne=np.ones((numberOfLines,1))
    X=np.column_stack((columnOne,x))
    theta=np.zeros((2,1))
    print(theta.shape)

    iterations=1500
    alpha=0.01
    JInitialization=computeCost(X,y,theta)
    theta=gradientDescent(X,y,theta,alpha,iterations)

    print('Theta found by gradient descent: ')
    print(theta)
    print('Program paused.Press enter to continue.\n')
    predict1=np.mat([1,3.5])*theta
    print('For population 35000,we predict a profit of ',predict1*10000)
    predict2=np.mat([1,7])*theta
    print('For population 70000,wo predict a profit of ',predict2*10000)

    #===================Part 4:Visualizing J(theta_0,theta_1)=================
    print('Visualizing J(theta_0,theta_1)...')
    theta0_vals=np.linspace(-10,10,200)
    theta1_vals=np.linspace(-2,4,200)
    J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))

    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t=np.array([theta0_vals[i],theta1_vals[1]]).reshape(2,1)
            J_vals[i][j]=computeCost(X,y,t)
    drawJ_vals(theta1_vals,theta0_vals,J_vals)
    drawJ_valsContour(theta0_vals,theta1_vals,J_vals.T)
    Plotting(x,y,theta)