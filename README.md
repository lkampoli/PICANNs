# PICANNs
Physics Informed Convex Artificial Neural Networks

# Description
This repository contains the source code used for the work in \cite{}. It allows to train a PICANN network to learn the optimal transport map between a unknown target distribution and a reference distribution. The transport map then can be applied on the reference distribution to get the density estimation of the unknown distribution. The inverse map can be used to transform the samples from reference distribution to generate new samples from the target distribution.

More details can be found in the article below.

# References
Will be added

# Dependencies
Python 3.6
Numpy >= 1.19.1
Pytorch >= 1.6.0
Scipy >= 1.5.0
Matplotlib >= 3.2.2

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
