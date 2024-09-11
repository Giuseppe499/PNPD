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

from examples import PNPD_NPDIT_k as test
from PNPD_comparison import parameters as PNPD_comparison_parameters
from numpy import mean

parameters = test.Parameters(**PNPD_comparison_parameters.__dict__)
parameters.k_max = [2**i for i in range(7)]
parameters.iterations = 50

def compute(*args, **kwargs):
    return test.compute(parameters=parameters, *args, *kwargs)

def plot(*args, **kwargs):
    return test.plot(*args, **kwargs)

def delta_table(data: test.TestData):
    table = [["$k_{{max}}$", "PNPD", "NPDIT", "$\Delta$", "NPDIT/PNPD"]]
    for k_max in parameters.k_max:
        row = []
        row.append(f"{k_max}")
        PNPD_average_timestep = mean(
            data.metrics_results[f"PNPD, $k_{{max}}={k_max}$"]["time"][1:])
        NPDIT_average_timestep = mean(
            data.metrics_results[f"NPDIT, $k_{{max}}={k_max}$"]["time"][1:])
        row.append(PNPD_average_timestep)
        row.append(NPDIT_average_timestep)
        delta = NPDIT_average_timestep-PNPD_average_timestep
        row.append(delta)
        row.append(NPDIT_average_timestep/PNPD_average_timestep)
        table.append(row)
    return table
        



if __name__ == "__main__":
    from init import DATA_SAVE_PATH as EXAMPLE_DATA_PATH
    from init import EXAMPLE_NAME
    from PNPD.utilities import load_data
    from examples.constants import *
    from tabulate import tabulate
    import numpy as np
    import os

    data = load_data(EXAMPLE_DATA_PATH)
    test_data = compute(data)
    table = delta_table(test_data)
    print(tabulate(table, headers="firstrow"))
    folder = ".." + PLOTS_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/"
    folder += test.TEST_NAME + "/"
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    with open(folder + "table.tex", "w") as file:
        file.write(tabulate(table, headers="firstrow", tablefmt="latex_raw"))
    np.savetxt(folder + "table.csv", table, delimiter=",", fmt='%s')
    plot(test_data, folder)
