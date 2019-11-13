import time

def timing(my_function):
    def wrapper():
        t1 = time.time()
        my_function()
        t2 = time.time()
        print('\n{} takes {:.4f} s'.format(my_function.__name__, t2 - t1))
    return wrapper

if __name__=='__main__':
    
    @timing
    def Test():
        s = 0
        for i in range(100000):
            s = i
        return i
    
    test()