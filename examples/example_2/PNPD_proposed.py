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
