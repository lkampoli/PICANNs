To run experiments in Table 1 : Run the bash script "Run_Tbl1_Exp.sh" with the appropriate parameters. The script assumes there are 2 GPUs to run 10 experiments each.

To generate Fig 3 and Fig 4 : Run the python script "PICANN_Example_Script.py" with appropriate dataset. After the training is done. Use another script "Make_Results.py" to generate all the graphs.

Once we have a forward model trained on any dataset, use the script "Dual_Driver.py" with appropriate parametes to train the inverse network.