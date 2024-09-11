"""
PNPD implementation

Copyright (C) 2024 Giuseppe Scarlato

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from context import PNPD, examples

from examples import PNPD_bootstrap as test
from init import DATA_SAVE_PATH as EXAMPLE_DATA_PATH
from init import EXAMPLE_NAME
from examples.constants import *

SAVE_PATH = "." + PICKLE_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME

parameters = test.Parameters(
    nu=[1e-2, 1e-1, 1e-3, 1e-2, 1e-2],
    lam=2e-4,
    iterations=151,
    bootstrap_iterations=[20,20,20,5,50],
    k_max= None
    )
parameters.k_max = [3] * len(parameters.nu)

def compute(*args, **kwargs):
    return test.compute(parameters=parameters, *args, **kwargs)

def plot(*args, **kwargs):
    return test.plot(*args, **kwargs)

if __name__ == "__main__":
    from PNPD.utilities import load_data
    
    PICKLE_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME
    data = load_data(EXAMPLE_DATA_PATH)
    test_data = compute(data=data, save_path=SAVE_PATH)
    plot(test_data, ".." + PLOTS_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME + "/")
