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

from examples import PNPD_proposed as test

parameters = test.Parameters(
    nu=[1e-1,4e-2,4e-2,4e-2],
    lam_PNPD=[6e-3,1e-2,1e-2],
    lam_NPD=7e-4,
    k_max= [5,1,5,10],
    iterations=51,
    bootstrap_iterations=30
    )

def compute(*args, **kwargs):
    return test.compute(parameters=parameters, *args, *kwargs)

def plot(*args, **kwargs):
    return test.plot(*args, **kwargs)

if __name__ == "__main__":
    from init import DATA_SAVE_PATH as EXAMPLE_DATA_PATH
    from init import EXAMPLE_NAME
    from PNPD.utilities import load_data
    from examples.constants import *
    data = load_data(EXAMPLE_DATA_PATH)
    test_data = compute(data)
    plot(test_data, ".." + PLOTS_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME + "/")
