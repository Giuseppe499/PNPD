import tests.PNPD_nu as test
SUB_TEST_NAME = "PNPD"

parameters = test.Parameters(iterations=150,
                             nu=[1,1e-1,1e-2],
                             lam_PNPD=[5e-4,5e-3,1.5e-2],
                             extrapolation=[True, True, True],
                             k_max=[1,1,5])

def compute(*args, **kwargs):
    return test.compute(parameters=parameters, *args, *kwargs)

def plot(*args, **kwargs):
    return test.plot(*args, **kwargs)

if __name__ == "__main__":
    from init import DATA_SAVE_PATH as EXAMPLE_DATA_PATH
    from init import EXAMPLE_NAME
    from utilities import load_data
    from tests.constants import *
    data = load_data(EXAMPLE_DATA_PATH)
    test_data = compute(data)
    plot(test_data, ".." + PLOTS_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME + "/" + SUB_TEST_NAME + "/")