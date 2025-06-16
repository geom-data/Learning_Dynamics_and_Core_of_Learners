# '25. Feburary Experiment note

# Experiment 1. Direction from 'untargeted' to '2nd or least likely'.

## experiment 1-1. 
To see if different direction of adversarial attack yields different prediction.

It seems that 'untargeted' and '2nd targeted' are predicted similarly, while 'targeted to least likely' doesn't.

## experiment 1-2. 
fuzziness by 'targeted to 2nd' seems greater than that by 'targeted to least'.

Since PGD method increases the loss between $\hat{y}$ and $y$ using the gradient, that will decrease the fuzziness of original prediction and increase others as they are added up to 1. And 'untargeted' method yields similar result with '2nd targeted'. That would imply that to decrease $P_{c}$ is to increase $P_{c_2}$ where $c_2$ is the target class of 2nd highest fuzziness. It may imply that $f^{-1}(c)$ and $f^{-1}(c_2)$ are close.

## experiment 1-3.

### experiment 1-3-1
Base + adversarial experts specialized from Base + filter specialized from Base

Validation dataset for the validation history : original dataset + adversarial dataset (1:1)

Result

Testing dataset | accuracy using validation history|
---|---|
original| 0.8785|
Original + adversarial (1:1) | 0.8964|
Adversarial only, untargeted, PGD |0.9142 |
Adversarial only, targeted to 2nd, PGD | 0.8798|
Adversarial only, targeted to least, PGD | 0.8598|
Adversarial only, untargeted, CWL2 | 0.0092 |

### experiment 1-3-2

Full and Fine model trained on original and adversarial attacks with 1:1 ratio having 20 targets, which will be called 'Full and Fine'.

'Full and Fine' only.

Validation dataset for the validation history : original dataset + adversarial dataset (1:1)

Result

Testing dataset | accuracy using validation history|
---|---|
original| 0.8155|
Original + adversarial (1:1) | 0.8269|
Adversarial only, untargeted, PGD |0.8383 |
Adversarial only, targeted to 2nd, PGD | 0.8088|
Adversarial only, targeted to least, PGD | 0.7756|
Adversarial only, untargeted, CWL2 | 0.0668 |

Remark : If I cwl2 is given with the original label, then accuracy is 0.7421

### experiment 1-3-3

original experts specialized from 'Full and Fine' + adversarial experts specialized from 'Full and Fine' + filter specialized from 'Full and Fine'.

Validation dataset for the validation history : original dataset + adversarial dataset (1:1)

Result

Testing dataset | accuracy using validation history|
---|---|
original| 0.9065|
Original + adversarial (1:1) | 0.9179|
Adversarial only, untargeted, PGD |0.9292 |
Adversarial only, targeted to 2nd, PGD | 0.8757|
Adversarial only, targeted to least, PGD | 0.8645|
Adversarial only, untargeted, CWL2 | 0.039 |

As expected.

Remark : If I cwl2 is given with the original label, then accuracy is 0.8517

-> Trained on PGD adversarial examples can defend CWL2, with the original label


# Experiment 2. Specialization and Migration. Testing on original, PGD type of adversarial, and CWL2 type of adversarial along with using validation history.

Migration is slightly modified in the note 202502.ipynb

-> refers turned into specialist.

 logifold has been upgraded with
+ base -> full and fine 
+ base -> full and fine -> filter

and migrations.

Migration, if possible, is applied with CWL2 dataset. It has 10k number of training and validation dataset. I tried two version of migration. First one is only CWL2, and the other one is CWL2 + original.

+ (base -> full and fine)
+ (base -> full and fine -> filter)
+ (base -> expert on adv)
+ (base -> filter)
+ (trained on original + adv of CWL2 type, full and fine, 20 targets)
+ (trained on original + adv of CWL2 type, full and fine, 20 targets -> expert on adv)
+ (trained on original + adv of CWL2 type, full and fine, 20 targets -> filter)

Result

**Committe**
Base, Expert from base(migrated on CWL2 only), filter from base(migrated on CWL2 only)
Testing dataset | accuracy using validation history|
---|---|
original| 0|
Original + adversarial (1:1) | 0.4315|
Adversarial only, untargeted, PGD |0.863 |
Adversarial only, targeted to 2nd, PGD | 0.8503|
Adversarial only, targeted to least, PGD | 0.8389|
Adversarial only, untargeted, CWL2 | 0.8681   |
Adversarial only, untargeted, CWL2, labeled to original | 0 |

**Committe** : Base, Expert from base, filter from base(migrated on original+ CWL2)

Testing dataset | accuracy using validation history|
---|---|
original| 0.4489|
Original + adversarial (1:1) | 0.6523|
Adversarial only, untargeted, PGD |0.8557 |
Adversarial only, targeted to 2nd, PGD | 0.8438|
Adversarial only, targeted to least, PGD | 0.826|
Adversarial only, untargeted, CWL2 | 0.8515 |
Adversarial only, untargeted, CWL2, labeled to original | 0.0063 |

**Committe** : Full and fine trained on PGD + original, migrated once on CWL2

Testing dataset | accuracy using validation history|
---|---|
original| 0.|
Original + adversarial (1:1) | 0.4388|
Adversarial only, untargeted, PGD |0.8775 |
Adversarial only, targeted to 2nd, PGD | 0.8517|
Adversarial only, targeted to least, PGD | 0.8237|
Adversarial only, untargeted, CWL2 | 0.8907 |
Adversarial only, untargeted, CWL2, labeled to original | 0 |

**Committe** : Full and fine trained on PGD + original, migrated on original+ CWL2

Testing dataset | accuracy using validation history|
---|---|
original| 0.8705|
Original + adversarial (1:1) | 0.8253|
Adversarial only, untargeted, PGD |0.7801 |
Adversarial only, targeted to 2nd, PGD | 0.8067|
Adversarial only, targeted to least, PGD | 0.8034|
Adversarial only, untargeted, CWL2 | 0.8519 |
Adversarial only, untargeted, CWL2, labeled to original | 0.0423 |


**Committe** : Based on Full and fine trained on PGD + original, specialized (original + adv expert + filter). Adv expert and filter are migrated on CWL2 only.

Testing dataset | accuracy using validation history|
---|---|
original| 0.0|
Original + adversarial (1:1) | 0.442|
Adversarial only, untargeted, PGD |0.8841 |
Adversarial only, targeted to 2nd, PGD | 0.8471|
Adversarial only, targeted to least, PGD | 0.825|
Adversarial only, untargeted, CWL2 | 0.8748 |
Adversarial only, untargeted, CWL2, labeled to original | 0 |

**Committe** : Based on Full and fine trained on PGD + original, specialized (original + adv expert + filter). Filter is migrated on(migrated on original+ CWL2)

Testing dataset | accuracy using validation history|
---|---|
original| 0.8944|
Original + adversarial (1:1) | 0.8832|
Adversarial only, untargeted, PGD |0.872 |
Adversarial only, targeted to 2nd, PGD | 0.8345|
Adversarial only, targeted to least, PGD | 0.7942|
Adversarial only, untargeted, CWL2 | 0.8305 |
Adversarial only, untargeted, CWL2, labeled to original | 0.0454 |

# Experiment 3. Logifold.

Target : 0 - 9 original, 10 - 19 PGD type adversarial (Num of classes)

Committee

Filter (Full and Fine trained on 20 targets -> Specialized -> migrated on original + CWL2, 4:1 ratio )

Original Experts 
1. ResNet56v1, Trained on original
2. Full and Fine trained on 20 targets -> Specialized to 0-9

Adversarial Experts
1. Full and Fine trained on 20 targets -> Specialized to 10-19 -> migrated on CWL2
2. FandF trained on 20 targets -> specialized to 10-19

Full and Fine
1. FandF trained on 20 targets
2. FandF trained on 20 targets -> Migrated on original + CWL2

**Committee** All
Testing dataset | accuracy using validation history|
---|---|
Validation dataset | 0.9092
original| 0.8339|
Original + adversarial (1:1) | 0.842|
Adversarial only, untargeted, PGD |0.8501 |
Adversarial only, targeted to least, PGD | 0.7888|
Adversarial only, untargeted, CWL2 | 0.0671|

**Committee** Without Full and Fine models
Testing dataset | accuracy using validation history|
---|---|
original| 0.8972|
Original + adversarial (1:1) | 0.9102|
Adversarial only, untargeted, PGD |0.9232 |
Adversarial only, targeted to least, PGD | 0.8514|
Adversarial only, untargeted, CWL2 | 0.8441|
Original + PGD untargeted gen by VGG16 | 0.4562|
PGD untargeted gen by VGG16 | 0.0152|
PGD untargeted gen by VGG16 with label 0 - 9 | 0.8158 |
PGD targeted to least likely class By VGG16 | 0.0116|
PGD targeted to least likely class By VGG16 with label 0-9 | 0.6903|

# Experiment 4. ResNet four models and VGG four models. Do they show different behaviour on adversarial attacks?

Hypothesis.

 Models with different structures are different chart. Therefore...
 
 1. They work in the similar manner for given 'unpolluted' sample with high certainty.
 2. They work in different manner for given 'adversarial' attack.






