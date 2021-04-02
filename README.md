# PICANNs
Physics Informed Convex Artificial Neural Networks

# Description

# References

# Dependencies

# Usage
To run experiments in Table 1 : Run the bash script "Run_Tbl1_Exp.sh" with the appropriate parameters. The script assumes there are 2 GPUs to run 10 experiments each. If not change the parameters in the script.

To generate Fig 3 and Fig 4 : Run the python script "PICANN_Example_Script.py" with appropriate dataset. After the training is done. Use another script "Make_Results.py" to generate all the graphs.

Once we have a forward model trained on any dataset, use the script "Dual_Driver.py" with appropriate parametes to train the inverse network.

# Licence

Copyright 2020 Amanpreet Singh, 
               Martin Bauer,
               Sarang Joshi

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



# Contacts
* Amanpreet Singh (u1209323 at umail dot utah dot edu)
* Martin Bauer (bauer at math dot fsu dot edu)
* Sarang Joshi (sjoshi at sci dot utah dot edu)
