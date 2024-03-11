# STA-250
STA250 (Deep Learning Theory) Final Project About StableRank 

based on python 3.9+ cuda 12.3

Autism dataset and Neur dataset are from 

## MNIST

- **Activation function:** ReLU (or Leaky-ReLU, with slope $\gamma = 0.01$).
- **Width of the network's hidden layer:** $m = 1000$.
- **Gaussian initialization:** Applied to the input-hidden layer weights, with a mean of $0$ and standard deviation defined by $\sigma \in \{0.00001,0.00005,0.0001,0.0005,0.001\}$.
- **Input size:** With a default value suitable for flattened 28x28 images, i.e., $784$.
- **Output size:** With a default setting of $10$, suitable for classification tasks like digit recognition.
- **Bias Initialization:** The biases of the first linear layer are initialized to $0$ if the weight variance is specified.
- **Training:** We train the NN with stochastic gradient descent with batch size 64 and learning rate 0.1 for 10 epochs.


## CIFAR 10

- **Activation function:** ReLU.
- **Width of the network's hidden layer:** $m = 512$.
- **Gaussian initialization:** Applied to the input-hidden layer weights, with a mean of $0$ and standard deviation defined by $\sigma \in \{0.00001,0.00005,0.0001,0.0005,0.001\}$.
- **Input size:** With a default value suitable for flattened 3x32x32 images, i.e., $3072$.
- **Output size:** With a default setting of $10$, suitable for classification tasks like digit recognition.
- **Bias Initialization:** The biases of the first linear layer are initialized to $0$ if the weight variance is specified.
- **Training:** We train the NN with stochastic gradient descent with batch size 128 and learning rate 0.01 for 100 epochs.

## Autism

Autism Dataset are measured among 104 samples: 47 autisms and 57 healthy controls, along with gender, brain region, age, and sites. Expressions of top 5 differently expressed genes and other variables are Orthogonal. Predict Autism or not (0-1).

- **Activation function:** ReLU (or Leaky-ReLU, with slope $\gamma = 0.01$).
- **Width of the network's hidden layer:** $m = 128$.
- **Gaussian initialization:** Applied to the input-hidden layer weights, with a mean of $0$ and standard deviation defined by $\sigma \in \{0.00001,0.00005,0.0001,0.0005,0.001\}$.
- **Bias Initialization:** The biases of the first linear layer are initialized to $0$ if the weight variance is specified.
- **Training:** We train the NN with gradient descent using full batch size 104 (number of samples) and learning rate 0.01 for 5000 epochs.

## Neuroblastoma

251 patients of the German Neuroblastoma Trials NB90-NB2004, diagnosed between 1989 and 2004, aged from 0 to 296 months (median 15 months). Oligonucleotide microarray with $p=10,707$. Predict Event-free (0-1).

- **Activation function:** ReLU (or Leaky-ReLU, with slope $\gamma = 0.01$).
- **Width of the network's hidden layer:** $m = 1000$.
- **Gaussian initialization:** Applied to the input-hidden layer weights, with a mean of $0$ and standard deviation defined by $\sigma \in \{0.00001,0.00005,0.0001,0.0005,0.001\}$.
- **Bias Initialization:** The biases of the first linear layer are initialized to $0$ if the weight variance is specified.
- **Training:** We train the NN with gradient descent using full batch size 246 (number of samples) and learning rate 0.22 for 50 epochs. (500 epochs is on working)
