# MNIST Experiment note(From Siu-Cheong)

We show that adversarial attack to mnist can be detected by checking mean and variance of simplified input.
By taking atlas of models that have suitable restricted domains, we can maintain high accuracy on union of original and adversarial samples.

### Preprocessing
1. Load MNIST dataset and scale the pixel values to the range $[0, 1]$.
2. Add a channel dimension to the images.

3. one-hot encode our labels and split.
```
Python

# 1

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 3

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_val = x_test[:8000]
y_val = y_test[:8000]
x_test = x_test[8000:]
y_test = y_test[8000:]

```

### Create Logifold

Adversarial Mnist data is concatenated with original data, where adversarials are labeled in $[10,20)$
```
fold = lgd.Logifold(20,name='advMNISTfold',x_tr=x_train20,
                y_tr=y_train20,x_v=x_val20,y_v=y_val20,
                path='advMNISTfold/',storyFile="story")
```

### Added models
|Key |Target | Description |
| --- | --- | ---:|
| $(0,)$|$[0, \ldots, 9]$ | |
| $(1,)$ | $[0, \ldots, 9]$||
|$(2,)$| $[10, \ldots, 19]$|By Turn Specialist from $(0,)$|
|$(3,)$| $[10, \ldots, 19]$|By Turn Specialist from $(1,)$|
|$(4,)$| $[0, \ldots, 19]$|Newly Trained|
|$(5,)$| $[0, \ldots, 9]$|By Turn Specialist from $(4,)$|
|$(6,)$| $[ [0,\ldots, 9], [10, \ldots, 19]]$|By Turn Specialist from $(4,)$|

'Best' Committee : 2,3,5,6

Using Validation History, tested on (original test + Adversarial test), 20k in total, it achieves 98.8% accuracy rate.

## Conclusion
1. Learning from adversarial attack (20 classes, key 4 above) and then specialized to original classes (10 classes, key 5 above) already improves acc of single model.
2. Logifold structure produces even further improvements

# CIFAR10 Experiment note(From Siu-Cheong)

model trained on polluted samples with 20 classes evaluated on clean samples,
by taking shortened answer np.argmax(ans[:10] + ans[10:],axis=1).
Result is slightly worse than 10-class model in the same size. 

Classifier distinguishes clean and adv samples with extremely high acc.
even for adv samples generated from a different model.
But perhaps it is because these models are similar?
Also the noise is small in this trial. (But still can distinguish!)
