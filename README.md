# Monte-Carlo simulation on GPU     
The goal is to compare 3 MC_methods aplied on pricing applications.  
Price is modeled as the end of a stochastic trajectory.  
MC_Euler Method try to follow this trajectory using Euler Explicit method.  
MC_exact use the law of the variance which can be computed explicitly and compute exactly the price.
MC_almost compute all the price's trajectory and use the law of the variance. 

# GPU techniques  
We compute moment and second moment on GPU by Reduction.  

# Take-home techniques    
Gamma simulation is implemented using reject method, the proxy used is some gaussian cube.  
To check the reject inequality we need to compute two log, to make it faster we find an inequality that implies the previous and is faster to check.  

# Benchmark's metric
MC_exact output is taken as the true price.  
- Running time execution
- Accuracy : \frac{\left| P_{\text{approx}} - P_{\text{true}} \right|}{P_{\text{true}}}
- Sensibility : Boxplot accuracy shows the sensibility to model's parameters. 

# How to run the repo ?   
Open the notebook in colab or kaggle for GPU availability, then run the notebook.  
