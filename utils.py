import time
import os
import argparse

def timing(my_function):
    def wrapper():
        t1 = time.time()
        my_function()
        t2 = time.time()
        print('\n{} takes {:.4f} s'.format(my_function.__name__, t2 - t1))
    return wrapper

def full_path(path):
    return os.path.normpath(os.path.realpath(os.path.expanduser(path)))

def get_parser(inputfolder="InputData"):
    parser = argparse.ArgumentParser(description="Set path")
    
    curr_path = full_path(__file__)
    input_dir = os.path.join(os.path.dirname(curr_path), inputfolder)
    
    parser.add_argument(
            "-i", "--input_dir",
            default = input_dir,
            help    = "Path of input data"
    )
    parser.add_argument(
            "-o", "--output_dir", 
            default = os.path.join(os.path.dirname(curr_path), "output"),
            help    = "Path of output directory"
    )
    parser.add_argument(
            "-w", "--work_dir", 
            default = os.path.dirname(curr_path),
            help    = "Working directory"
    )
    return parser

if __name__=='__main__':
    
    @timing
    def Test():
        s = 0
        for i in range(100000):
            s = i
        return i
    
    Test()
    
    print("This file: {}".format(full_path(__file__)))
        
    parser   = get_parser()
    args     = vars(parser.parse_args())
    work_dir = args.get('work_dir', 'error')
    print("Current working directory is {}".format(work_dir))
