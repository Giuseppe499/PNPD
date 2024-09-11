from context import PNPD, examples

from examples import PNPD_nu as test
from PNPD_nu import parameters
SUB_TEST_NAME = "PNPD_k=1"

parameters.k_max = [1 for nu in parameters.nu]
parameters.k_max_in_method_name = False

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
    plot(test_data, ".." + PLOTS_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME + "/" + SUB_TEST_NAME + "/")