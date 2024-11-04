import helper_functions as hf
import argparse

def start():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_file', type=str, nargs=1)
    # Read file
    args = parser.parse_args()
    A = hf.read_matrix_file(args.path_file[0])
    
    hf.print_matrix_info(A)

start()



