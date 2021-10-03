import os, sys, yaml
from shutil import copyfile
run_dir = os.getcwd()
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append(run_dir+'/case_builder')

import numpy as np
import matplotlib.pyplot as plt
from Case_builder import num_modes_list, input_len_list, output_len_list, method_list, var_num_list


os.mkdir(run_dir+'/results/comparisons/')

iter_num = 0
for num_modes in num_modes_list:
    
    try:
        os.mkdir(run_dir+'/results/comparisons/modes_'+str(num_modes))
    except:
        print(run_dir+'/results/comparisons/modes_'+str(num_modes)+' already exists. Moving on.')

    mode_path = run_dir+'/results/comparisons/modes_'+str(num_modes)
    
    for input_len in input_len_list:

        try:
            os.mkdir(mode_path+'/input_len_'+str(input_len))
        except:
            print(mode_path+'/input_len_'+str(input_len)+' already exists. Moving on.')

        input_path = mode_path+'/input_len_'+str(input_len)
        
        for output_len in output_len_list:

            try:
                os.mkdir(input_path+'/output_len_'+str(output_len))
            except:
                print(input_path+'/output_len_'+str(output_len)+' already exists. Moving on.')

            output_path = input_path+'/output_len_'+str(output_len)

            for method in method_list:

                try:
                    os.mkdir(output_path+'/method_'+str(method))
                except:
                    print(output_path+'/method_'+str(method)+' already exists. Moving on.')

                
                copyfile(run_dir+'/results/Experiment_'+str(iter_num)+'/everything.png',output_path+'/method_'+str(method)+'/MAE.png')

                iter_num = iter_num + 1