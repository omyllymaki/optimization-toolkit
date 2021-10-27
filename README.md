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

Given independent variables x, dependent variables y, termination criteria, and functions (f_eval, f_err, f_cost), 

- y_estimate = f_eval(x, parameters)
- errors = f_err(y_estimate, y)
- cost = f_cost(errors, parameters)

Solve optimal parameters that will minimize the cost, using selected optimization method.

## Usage example

```
# Specify function you want to fit
def f_eval(x, param):
    ...

optimizer = GaussNewton(f_eval=f_eval)
param, costs, _ = optimizer.fit(x, y, init_guess)
y_estimate = f_eval(x, param)
```

Samples folder contains multiple samples. Run all samples by typing:

```
./run_samples.sh
```