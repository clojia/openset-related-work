[Home](https://clojia.github.io/)

## Problem
Open set recognition: Using K classes labels identify K+1 classes (1 as unknown).
## Solutions

### non-Generative Model
Classification. No "unknown" samples generated.
#### OpenMax
#### II-loss
#### OSNN



### Generative Model
Generate samples for "unknown" class first, then do classification.
#### [G-OpenMax](https://clojia.github.io/independent-research/2018-10-IR-G-OpenMax)
Based on OpenMax and GANs, objective function as

<img src="../../independent-research/images/G-OpenMax-Obj.png" width="500"> 

The function has an extra variable C than standard GANs, which denotes categories, which will be random selected as well: D(data, category). And there are K pre-trained classifier. For each generated sample, if the class with the highest value is different from the pre-trained classifier, it will be marked as "unknown". 

#### [Adversarial Sample Generation](https://clojia.github.io/independent-research/2018-10-IR-Open-Set-by-Adversarial-Sample-Generation)
ASG also generates "unkown" samples, which are close to "known" samples, different from GANs minmax strategy, ASG generated samples who are:
1.close to the seen class data

<img src="../../independent-research/images/ASG_P1.png" width="500"> 

2.scattered around the seen/unseen boundary

<img src="../../independent-research/images/ASG_P2.png" width="500"> 

Hence in general,

<img src="../../independent-research/images/argmin_P1_P2.png" width="500"> 

In the prediction stage, for a test instance x, if all "known" classifier are negative, x is predicted as "unkown". Otherwise, x is predicted as the class with highest confidence.

#### [Counterfactual Image Generation](https://clojia.github.io/independent-research/2018-10-IR-Open-Set-Learning-with-Counterfactual-Images)
Counterfactual image generation generates "unknown" samples around "known" samples. Different with standard GANs, the model uses an interpolated gradient penalty term P(D) to generate samples.

- minimize discriminator error

<img src="../../independent-research/images/Counterfactual-LD.png" width="500"> 

- minimize generator error

<img src="../../independent-research/images/Counterfactual-LG.png" width="500"> 

The architecture consists of three components: an encoder network E(x), a generator G(z) and a discriminator D. 
And the goal is to generate synthetic images closed to the real image but not in any k classes:

<img src="../../independent-research/images/Counterfactual-Z*.png" width="500"> 

where z is the encoding of fake image, E(X) is the encoding of real image.
