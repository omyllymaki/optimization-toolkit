# Optimization toolkit

Home for different numerical optimization algos written in Python from scratch.

## Requirements

- Python 3.6
- Python libraries: numpy

Install requirements by running

```
pip3 install -r requirements-core.txt
```

To run samples, install also additional requirements:

```
pip3 install -r requirements.txt
```

## What does this do?

Given independent variables x, dependent variables y, and functions (feval, ferr, fcost), 

- y_estimate = feval(x, parameters)
- errors = ferr(y_estimate, y)
- cost = fcost(errors)

Solve optimal parameters that will minimize the cost, using selected optimization method.

## Usage example

```
# Specify function you want to fit
def feval(x, param):
    ...

optimizer = get_optimizer(method=Method.GN, feval=feval)
param, costs, _ = optimizer.fit(x, y, init_guess)
y_estimate = feval(x, param)
```

Samples folder contains multiple samples. Run all samples by typing:

```
./run_samples.sh
```