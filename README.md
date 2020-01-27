### Generalized Linear Model (GLM) for Classification

#### Task :
• Performed Linear, Poisson, Ordinal (for 5 levels) regression on given dataset.

• Showed that all 3 decrease in Error rate from approx. 45% to 30% as number of iterations increases.

• Used Bayesian approach as model selection to find the unknown parameter which was converged after 100 loops for time
efficiency which showed better results with respect to error rate.

#### Run the Code :
Give the following parameters in the command line with the file name: pp3.py
```
\python3 pp3.py A.csv labels-A.csv bayesian ten

\python3 pp3.py usps.csv labels-usps.csv bayesian ten

\python3 pp3.py AP.csv labels-AP.csv poisson ten

\python3 pp3.py AO.csv labels-AO.csv ordinal ten
```
#### Expected results:
Graph 1 : Error rate graph for given dataset

Graph 2 : Run time per iteration

Output: Total run time for the algorithm etc.

#### For Model Selection run this:
```
\python3 pp3.py A.csv labels-A.csv bayesian bms

\python3 pp3.py usps.csv labels-usps.csv bayesian bms

\python3 pp3.py AP.csv labels-AP.csv poisson bms

\python3 pp3.py AO.csv labels-AO.csv ordinal bms
```

###### Refer report for more details
