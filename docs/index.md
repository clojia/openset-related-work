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

<img src="https://github.com/clojia/independent-research/blob/master/images/G-OpenMax-Obj.png" width="500"> 

The function has an extra variable C than standard GANs, which denotes categories, which will be random selected as well: D(data, category). And there are K pre-trained classifier. For each generated sample, if the class with the highest value is different from the pre-trained classifier, it will be marked as "unknown". 

#### [Adversarial Sample Generation](https://clojia.github.io/independent-research/2018-10-IR-Open-Set-by-Adversarial-Sample-Generation)



#### Counterfactual Image Generation
