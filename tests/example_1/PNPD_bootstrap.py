import tests.PNPD_bootstrap as test
from init import DATA_SAVE_PATH as EXAMPLE_DATA_PATH
from init import EXAMPLE_NAME
from tests.constants import *

SAVE_PATH = "." + PICKLE_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME

parameters = test.Parameters(
    nu=[1e-2, 1e-1, 1e-3, 1e-4, 1e-2, 1e-2],
    lam=2e-4,
    iterations=151,
    bootstrap_iterations=[20,20,20,20,5,50],
    k_max= None
    )
parameters.k_max = [3] * len(parameters.nu)

def compute(*args, **kwargs):
    return test.compute(parameters=parameters, *args, **kwargs)

def plot(*args, **kwargs):
    return test.plot(*args, **kwargs)

if __name__ == "__main__":
    from utilities import load_data
    
    PICKLE_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME
    data = load_data(EXAMPLE_DATA_PATH)
    test_data = compute(data=data, save_path=SAVE_PATH)
    plot(test_data, ".." + PLOTS_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME + "/")
