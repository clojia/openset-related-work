[Home](https://clojia.github.io/)

## Problem
Open set recognition: Using K classes labels identify K+1 classes (1 as unknown).
## Solutions

### Selective Data

### No Additional Data

Classification. No "unknown" samples generated.
#### [OpenMax](https://clojia.github.io/independent-research/2018-08-IR-Open-Max)
OpenMax adapted EVT meta-recognition calibration in the penulimite layer of deep neural networks. For each instance, activation vector is revised to the sum of the product of its distance to the mean activation vectors (MAV) of each class. Then sent to softmax layer, which computes:

<img src="../../independent-research/images/openmax.png" width="250"> 

#### [II-loss](https://clojia.github.io/independent-research/2018-08-IR-Open-Set-Recognition)
ii-loss function was propsed in order to maximize the distance between different classes (inter class separation) and minimize distance of an instance from its class mean (intra class spread).

- intra class spread

<img src="../../independent-research/images/iiloss_intra.png" width="250"> 

- inter class spread

<img src="../../independent-research/images/iiloss_inter.png" width="250"> 

- ii-loss

<img src="../../independent-research/images/iiloss.png" width="300"> 

#### [OSNN](https://clojia.github.io/independent-research/2018-10-IR-NNDR)

The paper introduced two open-set extentions for the NN classifier: Class Verification (CV), as OSNN_cv and Nearest Neighbour Distance Ratio (OSNN). The difference is OSNN is able to classify test samples faraway from training ones while OSNN_cv does not.

### Generated Additional Data
Generate samples for "unknown" class first, then do classification.
#### [G-OpenMax](https://clojia.github.io/independent-research/2018-10-IR-G-OpenMax)
Based on OpenMax and GANs, objective function as

<img src="../../independent-research/images/G-OpenMax-Obj.png" width="500"> 

The function has an extra variable C than standard GANs, which denotes categories, which will be random selected as well: D(data, category). And there are K pre-trained classifier. For each generated sample, if the class with the highest value is different from the pre-trained classifier, it will be marked as "unknown". 

#### [Adversarial Sample Generation](https://clojia.github.io/independent-research/2018-10-IR-Open-Set-by-Adversarial-Sample-Generation)
ASG also generates "unkown" samples, which are close to "known" samples, different from GANs minmax strategy, ASG generated samples who are:
1.close to the seen class data

<img src="../../independent-research/images/ASG_P1.png" width="400"> 

2.scattered around the seen/unseen boundary

<img src="../../independent-research/images/ASG_P2.png" width="400"> 

Hence in general,

<img src="../../independent-research/images/argmin_P1_P2.png" width="400"> 

In the prediction stage, for a test instance x, if all "known" classifier are negative, x is predicted as "unkown". Otherwise, x is predicted as the class with highest confidence.

#### [Counterfactual Image Generation](https://clojia.github.io/independent-research/2018-10-IR-Open-Set-Learning-with-Counterfactual-Images)
Counterfactual image generation generates "unknown" samples around "known" samples. Different with standard GANs, the model uses an interpolated gradient penalty term P(D) to generate samples.

- minimize discriminator error

<img src="../../independent-research/images/Counterfactual-LD.png" width="300"> 

- minimize generator error

<img src="../../independent-research/images/Counterfactual-LG.png" width="300"> 

The architecture consists of three components: an encoder network E(x), a generator G(z) and a discriminator D. 
And the goal is to generate synthetic images closed to the real image but not in any k classes:

<img src="../../independent-research/images/Counterfactual-Z*.png" width="400"> 

where z is the encoding of fake image, E(X) is the encoding of real image.

### Introduced Additional Data
#### [Open Set Domain Adaptation by Backpropagation](https://clojia.github.io/independent-research/2018-11-IR-Open-Set-Domain-Adaptation-by-Backpropagation)
The paper marked unlabeled target samples as unknown, then mixed them with labeled source samples together to train a feature generator and a classifier. The objective functions look like:

<img src="../../independent-research/images/OSDAB-classifier.png" width="200"> 

<img src="../../independent-research/images/OSDAB-generator.png" width="200"> 

The classifier attempts to minimize both loss function whereas the generator attempts to maximize the value of L_adv(x_t) to deceive the classifier, such that it can generator better features which would recognize "known" samples from unlabeled target samples.
