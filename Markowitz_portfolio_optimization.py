%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt as opt
import scipy.optimize as sco

np.random.seed(123)

# Turn off progress printing 
# opt.solvers.options['show_progress'] = False

dft_n_port = 500
dft_n_ass = 3
dft_n_obs = 500

class MarkowitzPortfolio():
    def __init__(self, returns=None, n_assets=None, n_obs=None):
        #(n_assets, n_obs) for simulation, or (returns) for backtest
        self.returns = returns
        self.n_assets = n_assets
        self.n_obs = n_obs
        self.opt_w = [[] for _ in range(4)]
        self.opt_r = np.zeros(4)
        self.opt_vol = np.zeros(4)
        
        if returns: #not None, backtest
            self.n_obs = returns.shape[1]
            self.n_assets =  returns.shape[0]
        else:
            self.returns = self.rand_returns(self.n_assets, self.n_obs)
    
    def rand_returns(self, n_assets, n_obs):
        return np.random.randn(n_assets, n_obs)
    
    def plot_returns(self, returns=None): 
        #use self.returns or give (returns) 
        if returns == None:
            returns = self.returns
               
        plt.figure(figsize=(10,10))
        plt.style.use('seaborn')
        plt.grid(True)
        plt.plot(returns.T, alpha=.4);
        plt.xlabel('time')
        plt.ylabel('returns')
    
    def rand_a_weights(self):
        # Produces n random weights that sum to 1 
        k = np.random.randn(self.n_assets)
        return k / sum(k)
    
    def rand_a_portfolio(self):     
        # Return the mean and standard deviation of returns for a random portfolio
        p = np.asmatrix(np.mean(self.returns, axis=1))
        w = np.asmatrix(self.rand_a_weights())
        C = np.asmatrix(np.cov(self.returns))
    
        mu = w * p.T
        sigma = np.sqrt(w * C * w.T)
    
        # Leave outliers for cleaner plots
        if sigma > 2:
            return self.rand_a_portfolio()
        return mu, sigma
    
    def rand_portfolios(self, n_portfolios = dft_n_port):    
        self.n_portfolios = n_portfolios
        means, vol = np.column_stack([self.rand_a_portfolio() for _ in range(n_portfolios)])
        self.portfolios_means = means
        self.portfolios_vol = vol
      
    def plot_portfolios(self):
        plt.figure(figsize=(10,7))
        plt.style.use('seaborn')
        plt.grid(True)
        
        plt.plot(self.portfolios_vol, self.portfolios_means, 'o', markersize=5, alpha = 0.7)
        plt.xlabel('vol')
        plt.ylabel('mean')
        plt.title('Mean and standard deviation of returns of portfolios')

    def optimal_portfolios(self):
        n = self.n_assets
        returns = np.asmatrix(self.returns)
    
        N = 100
        mus = [10**(2 * i/N - 1.0) for i in range(N)]
        
        # Convert to cvxopt matrices
        Cov = opt.matrix(np.cov(returns))
        r = opt.matrix(np.mean(returns, axis=1))
    
        # Create constraint matrices
        G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
        h = opt.matrix(0.0, (n ,1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
    
        # Calculate efficient frontier weights using quadratic programming
        optimal_portfolios_w = [opt.solvers.qp(Cov, -r*mu, G, h, A, b)['x'] for mu in mus]
        self.optimal_portfolios_w = optimal_portfolios_w
        
        # Calculate risk and returns for frontier
        optimal_returns = [opt.blas.dot(r, w) for w in optimal_portfolios_w]
        optimal_risks = [np.sqrt(opt.blas.dot(w, Cov*w)) for w in optimal_portfolios_w]
        self.optimal_returns = optimal_returns
        self.optimal_risks = optimal_risks

    def plot_efficient_frontier(self):
        plt.figure(figsize=(10,7))
        plt.style.use('seaborn')
        plt.grid(True)
        plt.xlim(0, 2.5)
        plt.ylim(-0.15, 0.2)
        plt.plot(self.portfolios_vol, self.portfolios_means, 'o', markersize=5, alpha=0.7)
        plt.plot(self.optimal_risks, self.optimal_returns, 'lightseagreen')
        plt.xlabel('vol')
        plt.ylabel('return')
        
    def plot_best(self, case, rf):
        plt.figure(figsize=(10,7))
        plt.style.use('seaborn')
        plt.grid(True)
        
        plt.xlim(0, 2.5)
        plt.xlabel('vol')
        plt.ylabel('return')
        
        plt.plot(self.portfolios_vol, self.portfolios_means, 'o', markersize=5, alpha = 0.7)
        plt.plot(self.optimal_risks, self.optimal_returns, 'lightseagreen')
        plt.scatter(self.opt_vol[case], self.opt_r[case], color='tomato',marker='o', s=50)
        
        plt.plot([0,self.opt_vol[case]*2], [rf, self.opt_r[case]*2 - rf], color='tomato', alpha=0.4, animated=True)
              
    def min_variance_portfolio(self): #nocash_noshort
        #Convert to cvxopt matrices
        n = self.n_assets
        returns = np.asmatrix(self,returns)
            
        Cov = opt.matrix(np.cov(returns))
        r = opt.matrix(np.mean(returns, axis=1))
    
        # Create constraint matrices
        G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
        h = opt.matrix(0.0, (n ,1))
        A = opt.matrix(1.0, (1, n))
        b = opt.matrix(1.0)
    
        #Calculate weights using quadratic programming
        w = opt.solvers.qp(Cov, -r*0, G, h, A, b)['x']
        self.min_var_w = w
        
        ## CALCULATE RISKS AND RETURNS
        self.min_var_r = opt.blas.dot(r, w)
        self.min_var_vol = np.sqrt(opt.blas.dot(w, Cov*w))
  
    def sharpe(self, weights, Cov, r_mean, rf):
        w = np.array(weights)
        r = np.dot(w, r_mean)
        vol = np.sqrt(np.dot(w, np.dot(Cov, w)))
        return -(r - rf)/vol
    
    def which_case(sefl, rf, short):
        return (rf == 0 and short == 0)*0 + (rf == 0 and short == 1)*1 + (rf > 0 and short == 0)*2 + (rf > 0 and short == 1)*3
        
    def get_parameters(self, case, rf):
        if case == 0:
            return self.n_assets, self.returns, tuple((0, 1) for _ in range(self.n_assets))
        elif case == 1:
            return self.n_assets, self.returns, tuple((-1, 1) for _ in range(self.n_assets))
        elif case == 2:
            return self.n_assets+1, np.vstack((self.returns, rf*np.ones((1,self.n_obs)))),                   
                                               tuple((0, 1) for _ in range(self.n_assets+1))
        elif case == 3:
            return self.n_assets+1, np.vstack((self.returns, rf*np.ones((1,self.n_obs)))),
                                              tuple((-1, 1) for _ in range(self.n_assets+1))
        
        
    def market_portfolio(self, rf = 0, short = 0, plot = True):
        case = self.which_case(rf, short)
        n, returns, bnds = self.get_parameters(case, rf)
        
        Cov = np.cov(returns)
        r = np.mean(returns, axis=1)
                
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        w = sco.minimize(self.sharpe, n*[1./n], args = (Cov, r, rf),
                         method = 'SLSQP', bounds = bnds, constraints = cons)['x']
        
        self.opt_w[case] = w
        self.opt_r[case] = np.dot(w, r)
        self.opt_vol[case] = np.sqrt(np.dot(w, np.dot(Cov,w)))
        
        print('%-18s'%'risk_free rate: ', rf)
        print('%-18s'%'short allowed: ', short == 1)
        print('%-18s'%'weights:', np.round(w, 5))
        print('%-18s'%'vol:', '%.5f'%self.opt_vol[case])
        print('%-18s'%'return:','%.5f'%self.opt_r[case])
        if plot:
            self.plot_best(case, rf)
