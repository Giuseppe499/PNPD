# PNPD (Preconditioned Nested Primal-Dual)

This repository implements the **Preconditioned Nested Primal-Dual (PNPD)**
method. It includes both the core PNPD package and some examples for
testing its performance.

## Repository Structure

```
├── examples
│   ├── example_1
│   ├── example_2
│   └── example_3
└── PNPD
    ├── math_extras.py
    ├── plot_extras.py
    ├── schedulers.py
    ├── solvers.py
    └── utilities.py
```

- `PNPD`: python package and supporting utilities.
   - `solvers.py`: implementation of the PNPD method and other solvers used for the examples.
   - `math_extras.py`: mathematical functions and utilities.
   - `plot_extras.py`: utilities for generating plots of results.
   - `schedulers.py`: scheduling functions for the non-stationary
     version of PNPD.
   - `utilities.py`: other generic utilities.
- `examples`: test scripts and examples for evaluating the PNPD method's
  performance.
   - `example_*`: each folder contains a suite of tests for a specific problem.

## Running Examples

1. **(Optional, recommended)** Create a virtual environment to isolate dependencies (see [python
   docs](https://docs.python.org/3/library/venv.html))
2. Move to the root folder of this repository:

   ```
   cd path/to/repo
   ```

3. Install requirements with

   ```
   pip install .
   ```
4. Move to an example folder (e.g., `example_1`):  

   ```
   cd ./examples/example_1
   ```

5. Generate the blurred image and *point spread function (PSF)* for the example

   ```
   python init.py
   ```

6. Run a test, (e.g., `PNPD_comparison.py`):

   ```
   python PNPD_comparison.py 
   ```

7. Check the generated plots and results in `./examples/plots/` directory.