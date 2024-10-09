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
* Use pure NumPy implementation of AISLE
* Use JAX acelereated implementation of AISLE
* A sample use with linear and quadratic neural unit
* Easy start sample usage examples in `.ipynb`

## How To Install
Currently we provide only Github repo, PiPy distribution will be released later.

### From source / GitHub
```bash
git clone https://github.com/carnosi/AISLEX.git
```
## Example Usage
AISLEX provides two different backends for LE evaluation. Where the functions with **NumPy** backend can be imported as
```python
from src.aisle import aisle, aisle_window
```
and the functions with **JAX** backend can be imported as
```python
from src.aisle_jax import aisle, aisle_window
```
For single LE we can invoke The `aisle` function which returns the values in the shape of $[1, N_{OLE}]$, where $N_{OLE}$ is the amount of Orders of Learning Entropy. For evaluation we need to set hyper-parameters `oles` and `alphas` accordingly with monitored system.

* `oles` can be viewed as derivative values of weights to $n^{th}$ derivation.
* `alphas` set sensitivity to behavior which is already considered anomalous compared to the current normal. Resulting LE value is the mean from all alphas.

#### Single LE evaluation:
```python
# Generate random weights for example
n_update_steps = 100 # Number of recorded weight updates
weight_count = 100
weights = np.random.rand((n_update_steps, weight_count))

oles = (1, 2) # Orders of Learning Entropy
alphas = (6, 12, 13.5, 16.87) # Sensitivity of anomaly detection, can be int or float

aisle_values = aisle(weights, alphas, oles)
```
For sliding window evaluation of historical data the LE can be evaluated in a batched manner by invoking the `aisle_window` function, which returns the values in the shape of $[n_{windows}, N_{OLE}]$, where $n_{windows}$ is the amount of individual sliding windows and for each of them is evaluated one LE for $N_{OLE}$ Orders of Learning Entropy.

In **JAX** implementation the batches are mapped by `jax.vmap` function, which can be memory heavy, as each batch is saved individually for further parallelized processing.

#### Window LE evaluation:
```python
# Generate random weights for example
n_update_steps = 1000 # Number of recorded weight updates
weight_count = 100
weights = np.random.rand((n_update_steps, weight_count))

oles = (1, 2) # Orders of Learning Entropy
alphas = (6, 12, 13.5, 16.87) # Sensitivity of anomaly detection, can be int or float
window_size = 100

aisle_values = aisle_window(window_size, weights, alphas, oles)
```

##### Experimental features
Experimental features are not fully tested and may sometimes even show reduced performance compared to the fully released one. Currently under experimental features you can find the following:
* _Window chunk processing (JAX)_ - Allows users to find equillibrium between memory usage and execution speed by processing n windows at a given moment.

_Window chunk processing_
```python
from src.aisle_jax import aisle_window_chunked

# Generate random weights for example
n_update_steps = 10000 # Number of recorded weight updates
weight_count = 100
weights = np.random.rand((n_update_steps, weight_count))
window_chunk = 1000

oles = (1, 2) # Orders of Learning Entropy
alphas = (6, 12, 13.5, 16.87) # Sensitivity of anomaly detection, can be int or float
aisle_values = aisle_window_chunked(window_size, weights, alphas, oles, window_chunk)
```

## Jupyter Examples
In examples folder we provide code for sample usage of AISLEX for anomaly detection on artificial and real data and performance benchmark between our JAX and NumPy implementation. The examples include:

* [Example 0 Artificial Signal](./examples/Example_0_Artificial_Signal.ipynb)
* [Example 1 Dynamic System](./examples/Example_1_Dynamic_System.ipynb)
* [Example 2 ECG Ventricual Tachycardia](./examples/Example_2_ECG_Ventricual_Tachycardia.ipynb)
* [Example 3 Polarization Anomaly Detection](./examples/Example_3_Polarization_Anomaly_Detection.ipynb)
* [Performance Benchmark](./examples/Performance_comparison.ipynb)

## Requirements
The code has been tested with **JAX 0.4.23**, rest of the requirements is available in [requirements.txt](requirements.txt).

## Tips & Tricks
AISLEX relies on three hyperparameters for its evaluation. The optimal settings for these hyperparameters are always use-case specific, and there is no one-size-fits-all approach. Here are some tips to help you set up your anomaly detection pipeline effectively.

### Window size
The window size determines what the learning entropy will consider as normal behavior. Generally, you should set the window size longer than the longest expected anomaly. For example, if window=20 and your target anomaly usually lasts ~15 samples, a short window may bias AISLEX's detection, reducing the accuracy of anomaly identification.

We generally recommend setting the window size to be at least 2$\times$ longer than the expected anomaly length.

### Alpha sensitivity
The alpha parameter ($\alpha$) determines what weight changes will be considered anomalous and can take any real number. The sensitivity of $\alpha$ can be understood in several ways:

* The larger the value of $\alpha$, the larger the magnitudes of weight increment considered unusual.
* The larger the value of $\alpha$, the less sensitive Learning Entropy (LE) is to data that do not correspond to the current dynamics learned by the model.
* The larger the $\alpha$, the more unusual data samples in the signal will be detected.

In general, we recommend using between 2-6 different alpha sensitivities to approximate the optimal alpha for your specific system and desired anomaly sensitivity.

For initial setup, it is beneficial to have both clean and anomalous data available, allowing you to iterate toward the desired LE sensitivity.

### Orders of Learning Entropy
Orders of Learning Entropy (OLE) refer to the differential order of update changes, such as acceleration. Depending on the anomaly type and system response behavior, you should select the appropriate order or orders.

For data from mechanical systems, we generally expect that orders 1-3 will capture most anomalous behavior. However, for complex systems, such as those involving polarization changes, higher orders between 6-8 may be more beneficial.

You can experimentally determine the relevant OLEs relatively quickly, as the resulting LE will typically remain around 0 if the data do not contain relationships in higher orders.

## Application Examples
This repository is builing on top of many published papers, for sample usage you can explore any of the below:

[1] I. Bukovsky, M. Cejnek, J. Vrba, and N. Homma, “Study of Learning Entropy for Onset Detection of Epileptic Seizures in EEG Time Series,” in 2016 International Joint Conference on  Neural Networks  (IJCNN), Vancouver: IEEE, Jul. 2016.

[2] I. Bukovsky, N. Homma, M. Cejnek, and K. Ichiji, “Study of Learning Entropy for Novelty Detection in lung tumor motion prediction for target tracking radiation therapy,” in 2014 International Joint Conference on Neural Networks (IJCNN), Jul. 2014, pp. 3124–3129. doi: 10.1109/IJCNN.2014.6889834.

[3] I. Bukovsky, G. Dohnal, P. Steinbauer, O. Budik, K. Ichiji, and H. Noriyasu, “Learning Entropy of Adaptive Filters via Clustering Techniques,” in 2020 Sensor Signal Processing for Defence Conference (SSPD), Edinburgh, Scotland, UK: IEEE, Sep. 2020, pp. 1–5. doi: 10.1109/SSPD47486.2020.9272138.

[4] I. Bukovsky and C. Oswald, “Case Study of Learning Entropy for Adaptive Novelty Detection in Solid-Fuel Combustion Control,” in Intelligent Systems in Cybernetics and Automation Theory, vol. 348, R. Silhavy, R. Senkerik, Z. K. Oplatkova, Z. Prokopova, and P. Silhavy, Eds., Cham: Springer International Publishing, 2015, pp. 247–257. Accessed: Jan. 18, 2016. [Online]. Available: http://link.springer.com/10.1007/978-3-319-18503-3_25

## References
The algorithm foundation builds upon following papers:

[1] I. Bukovsky, “Learning Entropy: Multiscale Measure for Incremental Learning,” Entropy, vol. 15, no. 10, pp. 4159–4187, Sep. 2013, doi: 10.3390/e15104159.

[2] I. Bukovsky, J. Vrba, and M. Cejnek, “Learning Entropy: A Direct Approach,” in IEEE International Joint Conference on Neural Networks, Vancouver: IEEE, Feb. 2016.

[3] I. Bukovsky, W. Kinsner, and N. Homma, “Learning Entropy as a Learning-Based Information Concept,” Entropy, vol. 21, no. 2, p. 166, Feb. 2019, doi: 10.3390/e21020166.

## How To Cite
If AISLEX has been useful in your research, please consider citing our article:
IEEE:
```
[1] O. Budik, M. Novak, F. Sobieczky, and I. Bukovsky, “AISLEX: Approximate individual sample learning entropy with JAX,” SoftwareX, vol. 28, p. 101915, Dec. 2024, doi: 10.1016/j.softx.2024.101915.
```
BibTex:
```
@article{BUDIK2024101915,
title = {AISLEX: Approximate individual sample learning entropy with JAX},
journal = {SoftwareX},
volume = {28},
pages = {6},
year = {2024},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2024.101915},
url = {https://www.sciencedirect.com/science/article/pii/S2352711024002851},
author = {Ondrej Budik and Milan Novak and Florian Sobieczky and Ivo Bukovsky},
keywords = {Anomaly detection, Learning entropy, JAX, Machine learning weight evaluation},
abstract = {We present AISLEX, an online anomaly detection module based on the Learning Entropy algorithm, a novel machine learning-based information measure that quantifies the learning effort of neural networks. AISLEX detects anomalous data samples when the learning entropy value is high. The module is designed to be readily usable, with both NumPy and JAX backends, making it suitable for various application fields. The NumPy backend is optimized for devices running Python3, prioritizing limited memory and CPU usage. In contrast, the JAX backend is optimized for fast execution on CPUs, GPUs, and TPUs but requires more computational resources. AISLEX also provides extensive implementation examples in Jupyter notebooks, utilizing in-parameter-linear-nonlinear neural architectures selected for their low data requirements, computational simplicity, convergence analyzability, and dynamical stability.}
}
```
