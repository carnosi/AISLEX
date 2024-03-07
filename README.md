# AISLEX
[![Project Status: Active - The project has reached a stable, usable
state and is being actively
developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction
 Welcome to **AISLEX**, an Python package for Approximate Individual Sample Learning Entropy anomaly detection technieque.

 AISLEX provides an easy way to implement anomaly detection to your neural units implementations. The Learning Entropy (LE) evaluates, how the weights of the neural network model are adapting during training. This evaluation allows us to detect anomalies or novelty even in systems, those can not be accurately approximated. When an anomally occures, the weights of the neural network have to compensate for this sudden change which can be interpreted as a novelty.

 In this repository we provide original, pythonic, implementation which is easy to understand and JAX accelerated evalution which can evaluate even larger windows or higher weight counts more efficiently.

## Features
The **AIXLES** package allows the user to perform the following tasks:
* Use pythonic implementation of AISLE
* Use JAX acelereated implementation of AISLE
* A sample use with linear and quadratic neural unit
* Easy start sample usage examples in `.ipynb`

## Examples
In examples folder we provide code for sample usage of AISLEX for anomaly detection on artificial and real data. Our examples include:

* [Example 0 Artificial Signal](./examples/Example_0_Artificial_Signal.ipynb)
* [Example 1 Dynamic System](./examples/Example_1_Dynamic_System.ipynb)
* [Example 2 ECG Ventricual Tachycardia](./examples/Example_2_ECG_Ventricual_Tachycardia.ipynb)
* [Example 3 Performance Benchmark](./examples/Example_3_Performance_comparison.ipynb)

## Appliaction Examples
This repository is builing on top of many published papers, for sample usage you can explore any of the below:

[1] I. Bukovsky, M. Cejnek, J. Vrba, and N. Homma, “Study of Learning Entropy for Onset Detection of Epileptic Seizures in EEG Time Series,” in 2016 International Joint Conference on  Neural Networks  (IJCNN), Vancouver: IEEE, Jul. 2016.

[2] I. Bukovsky, N. Homma, M. Cejnek, and K. Ichiji, “Study of Learning Entropy for Novelty Detection in lung tumor motion prediction for target tracking radiation therapy,” in 2014 International Joint Conference on Neural Networks (IJCNN), Jul. 2014, pp. 3124–3129. doi: 10.1109/IJCNN.2014.6889834.

[3] I. Bukovsky, G. Dohnal, P. Steinbauer, O. Budik, K. Ichiji, and H. Noriyasu, “Learning Entropy of Adaptive Filters via Clustering Techniques,” in 2020 Sensor Signal Processing for Defence Conference (SSPD), Edinburgh, Scotland, UK: IEEE, Sep. 2020, pp. 1–5. doi: 10.1109/SSPD47486.2020.9272138.

[4] I. Bukovsky and C. Oswald, “Case Study of Learning Entropy for Adaptive Novelty Detection in Solid-Fuel Combustion Control,” in Intelligent Systems in Cybernetics and Automation Theory, vol. 348, R. Silhavy, R. Senkerik, Z. K. Oplatkova, Z. Prokopova, and P. Silhavy, Eds., Cham: Springer International Publishing, 2015, pp. 247–257. Accessed: Jan. 18, 2016. [Online]. Available: http://link.springer.com/10.1007/978-3-319-18503-3_25


## Requirements
The code has been tested with JAX 0.4.23, rest of the requirements is available in [requirements.txt](requirements.txt).


## How To Install
Currently we provide only Github repo, pipy support will be released later.

### From source/GitHub
```
git clone https://github.com/carnosi/AISLEX.git
```

## References
The algorithm foundation builds upon following papers:

[1] I. Bukovsky, “Learning Entropy: Multiscale Measure for Incremental Learning,” Entropy, vol. 15, no. 10, pp. 4159–4187, Sep. 2013, doi: 10.3390/e15104159.

[2] I. Bukovsky, J. Vrba, and M. Cejnek, “Learning Entropy: A Direct Approach,” in IEEE International Joint Conference on Neural Networks, Vancouver: IEEE, Feb. 2016.

[3] I. Bukovsky, W. Kinsner, and N. Homma, “Learning Entropy as a Learning-Based Information Concept,” Entropy, vol. 21, no. 2, p. 166, Feb. 2019, doi: 10.3390/e21020166.



## How To Cite
If AISLEX has been useful in your research, please consider citing our article:
```
[TBD]
```
