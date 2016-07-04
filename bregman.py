
# coding: utf-8

import numpy as np
from scipy import optimize


"""A dictionary of function and gradient oracles that can be fed as input 
    to the Bregman class

    Format
    --------

    <label> : [function lambda , gradient lambda] 

"""
losses = {
    "squared_error" : [lambda x: x**2 , lambda x: 2*x],
    "exponential" : [lambda x: np.exp(x), lambda x: np.exp(x)],
    "xlogx" : [lambda x: x* np.log(x), lambda x: np.log(x)+1]

}

class Bregman:
    """Class to experiment with different bregmann divergences 
    Parameters
    ----------

    low : float, lower limit of data
    high: float, upper limit of data
    points: int , number of points to be plotted between <low> and <high>
    data_fn: function of x. This is the curve the data points will follow.
    noise: float, The variance of the noraml distribution used as noise to the 
    signal.
    loss: string,The loss function label to be used from the 'losses' dictionary.
    """
    
    def __init__(self, low= 1, high = 20, points = 200, 
        data_fn = lambda x:2*x, loss="squared_error" , noise=0.1):
        self.xx, self.yy =self.get_data(low,high, points, data_fn, noise)
        self.w = np.random.rand(1,self.xx.shape[1]).transpose()
        self.intercept = np.zeros(self.xx.shape)
        self.function = losses[loss][0]
        self.gradient = losses[loss][1]

    def get_data(self, low, high, points, fn, noise=0.1):
        """Generate a linear regression problem. Only called by init."""
        xx  = np.linspace(low, high, points)
        xx = xx.reshape(xx.shape[0],1)
        yy= fn(xx)
        yy = np.random.normal(yy,noise)
        # yy = 2 * np.random.normal(xx)
        return xx,yy

    def bregman(self,x,y):
        """
        Computes the bregman divergence dfor 2 given data points.

        Parameters
        ---------
        x,y : Vectors, 1d array
        Data points

        Returns
        -------
        div: float,
        The divergence between the two points.
        """
        div = self.function(y) - self.function (x) - \
            np.dot(self.gradient(x).transpose(), (y-x))
        return div
    
    def loss(self, w):
        """Compute a loss over a list of data points, given a weight vector
        
        Parameters
        ---------
        w: vector, 1d array
        The weight vector

        Returns
        ------

        total: float
        The loss over the data points given a bregman divergenc
        """

        total = 1.0/self.xx.shape[0] * sum([self.bregman(np.dot(w,x),y) for x,y 
            in zip(self.xx,self.yy) ]) 
        return total

    def minimize(self):

        """Minimizes the  loss function based on the Bregman Divergence, and 
        updates the 'w' (weight) and intercept variables of the imstamce.

        """

        lin =  optimize.minimize(self.loss, self.w, method = 'Nelder-Mead')
        if lin.success:
            self.w = lin.x 
            self.intercept =  np.mean(self.yy) - (np.mean(self.xx)*self.w)
        else:
            print "Nelder- Mead failed. Switching to BGFS"
            print "Message:", lin.message 
            lin =  optimize.minimize(self.loss, self.w)
            if lin.success:
                self.w = lin.x 
                self.intercept =  np.mean(self.yy) - (np.mean(self.xx)*self.w)
            else:                
                print "The minimization failed. Please check your parameters."
                print "Message:", lin.message 

        return lin.success



    def get_weight_values(self, limit):
        """Returns a list of points that show the behaviour of the loss 
        function around  the mimimum. Call this function after calling
        minimize

        Parameters
        ---------
        limit: float,
        The area of around the minimum to be plotted.

        Returns
        --------
        ww: List of weight values
        ll : Corresponding value for the loss function
        List of points.
        """
        ww = np.linspace(self.w - limit, self.w + limit, 200)
        ll  = [self.loss(w) for w in ww]
        return ww,ll




