# pyGCO: a python wrapper for the graph cuts

**The original wrapper is [pygco](https://github.com/yujiali/pygco)**

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gco-wrapper)](https://pypi.org/project/gco-wrapper/)
[![PyPI Status](https://badge.fury.io/py/gco-wrapper.svg)](https://badge.fury.io/py/gco-wrapper)
[![PyPI Status](https://pepy.tech/badge/gco-wrapper)](https://pepy.tech/project/gco-wrapper)

![CI testing](https://github.com/Borda/pyGCO/workflows/CI%20testing/badge.svg?branch=master&event=push)
[![codecov](https://codecov.io/gh/Borda/pyGCO/branch/master/graph/badge.svg)](https://codecov.io/gh/Borda/pyGCO)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/4b875e6fbf3349e18a139a2a005736a4)](https://www.codacy.com/app/Borda/pyGCO?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Borda/pyGCO&amp;utm_campaign=Badge_Grade)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Borda/pyGCO/master.svg)](https://results.pre-commit.ci/latest/github/Borda/pyGCO/master)

<!--
[![Maintainability](https://api.codeclimate.com/v1/badges/9c11485e2f4f23189069/maintainability)](https://codeclimate.com/github/Borda/pyGCO/maintainability)
[![Coverage Badge](https://api.shippable.com/projects/5883a5e12a5d900f00b8b9ff/coverageBadge?branch=master)](https://app.shippable.com/github/Borda/pyGCO)
-->

---

This is a python wrapper for [gco-v3.0 package](http://vision.csd.uwo.ca/code/), which implements a graph cuts based move-making algorithm for optimization in Markov Random Fields.

It contains a copy of the **gco-v3.0 package**.  Some of the design were borrowed from the [gco_python](https://github.com/amueller/gco_python) package. However, compared to gco_python:
* This package does not depend on Cython. Instead it is implemented using the ctypes library and a C wrapper of the C++ code.
* This package is an almost complete wrapper for gco-v3.0, which supports more direct low level control over GCoptimization objects.
* This package supports graphs with edges weighted differently.

This wrapper is composed of two parts, a C wrapper and a python wrapper.

## Implemented functions
 * **cut_general_graph**(...)
 * **cut_grid_graph**(...)
 * **cut_grid_graph_simple**(...)

## Building wrapper

1. download the last version of [gco-v3.0](http://vision.csd.uwo.ca/code/gco-v3.0.zip) to the _gco_source_
1. compile gco-v3.0 and the C wrapper using `make`
1. compile test_wrapper using `make test_wrapper`
1. run the C test code `./test_wrapper` (now you have the C wrapper ready)
```bash
make download
make all
make test_wrapper
./test_wrapper
```

The successful run should return:
```bash
labels = [ 0 2 2 1 ], energy=19
data energy=15, smooth energy=4
```

Next test the python wrapper using `python test_examples.py`, if it works fine you are ready to use pygco.

To include pygco in your code, simply import pygco module. See the documentation inside code for more details.

## Install wrapper

Clone repository and enter folder, then

```bash
pip install -r requirements.txt
python setup.py install
```

Now it can be also installed from PyPi
```bash
pip install gco-wrapper
```

## Show test results

Visualisation of the unary terns for **binary segmentation**

![unary terms](./images/binary_unary.png)

**4-connected** components with the initial labeling (left) and estimated labeling with regularisation **1** (middle) and **0** (right)

![labelling](./images/binary_labels-4conn.png)

**8-connected** components with the initial labeling (left) and estimated labeling with regularisation **1** (middle) and **0** (right)

![labelling](./images/binary_labels-8conn.png)

Visualisation of the unary terns for **3 labels segmentation**

![unary terms](./images/grid_unary.png)

with the __initial__ labeling (left) and __estimated__ labeling (right)

![labelling](./images/grid_labels.png)
