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

Minimal example:
```
def f_eval(x, param):
    return param[0] * x + param[1]


x = np.arange(1, 100)
param_true = np.array([1.0, 2.5])
y = f_eval(x, param_true)
param, costs, _ = GaussNewton(f_eval=f_eval).run(x, y, np.random.randn(2))
print(f"Param: {param}")
print(f"Costs: {costs}")
```

Samples folder contains multiple samples. Run all samples by typing:

```
./run_samples.sh
```