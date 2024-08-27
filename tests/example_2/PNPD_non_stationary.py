import tests.PNPD_non_stationary as test
from init import DATA_SAVE_PATH as EXAMPLE_DATA_PATH
from init import EXAMPLE_NAME
from tests.constants import *

SAVE_PATH = "." + PICKLE_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME

parameters = test.Parameters(
    nu=[1e-1, 1e-2, 1e-2, 1e-2],
    lam=[5e-3, 2e-4, 5e-4, 5e-4],
    iterations=150,
    k_max= [1,1,1,1]
    )

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
