# PNPD (Preconditioned Nested Primal-Dual)

## How to run examples

1. (Optional, recommended) Create a virtual environment and activate it
   (see [python docs](https://docs.python.org/3/library/venv.html))
2. Install requirements with `pip install .`
3. Move to an example folder e.g.:  

   ```
   cd ./examples/example_1
   ```

4. Generate the blurred image and the psf for the example

   ```
   python init.py
   ```

5. Run a test e.g.:

   ```
   python PNPD_comparison.py 
   ```

6. Check the results in `./examples/plots/`
