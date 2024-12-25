import numpy as np
class customLinearRegression:
    def __init__(self,alpha=0.0001,epoch = 10):
        self.w1 = np.random.random()
        self.w2 = np.random.random()
        self.b = np.random.random()
        self.alpha = alpha 
        self.epoch = epoch
    
    def fit(self,X,y):
        #X_trainshape->(331,5) row, col ,rocords = 331
        self.num_rec = X.shape[0]
        y_hat = self.predict(X)
        
        for i in range (self.epoch):
            y_hat = self.predict(X)
            
            loss = self.loss_mse(y,y_hat)
        
            diff = y - y_hat
            
            grad_w1 = (2/self.num_rec)*np.sum(diff*X['X2 house age'])
            grad_w2 = (2/self.num_rec)*np.sum(diff*X['X3 distance to the nearest MRT station'])
            grad_b = (2/self.num_rec)*np.sum(diff)
            
            self.w1 = self.w1 - self.alpha* grad_w1
            self.w2 = self.w2 - self.alpha* grad_w2
            self.b = self.b - self.alpha* grad_b
            
    def predict (self,X):
        return( self.w1*X['X2 house age'] 
            + self.w2*X['X3 distance to the nearest MRT station']
            + self.b )   #y=wx + b
    
    def loss_mse(self,y,y_hat):
        return np.sum((y- y_hat) ** 2)/self.num_rec
    
    