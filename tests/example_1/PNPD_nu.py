import tests.PNPD_nu as test
SUB_TEST_NAME = "PNPD"

parameters = test.Parameters(iterations=80,
                             nu=[1,1e-1,1e-2],
                             lam_PNPD=[3e-4,2e-3,1e-2],
                             extrapolation=[True, True, True],
                             k_max=[1,3,10])

def compute(*args, **kwargs):
    return test.compute(parameters=parameters, *args, *kwargs)

def plot(*args, **kwargs):
    return test.plot(*args, **kwargs)

if __name__ == "__main__":
    from init import DATA_SAVE_PATH as EXAMPLE_DATA_PATH
    from init import EXAMPLE_NAME
    from utilities import load_data
    from tests.constants import *
    from matplotlib import pyplot as plt
    data = load_data(EXAMPLE_DATA_PATH)
    test_data = compute(data)
    folder =  ".." + PLOTS_SAVE_FOLDER + "/" + EXAMPLE_NAME + "/" + test.TEST_NAME + "/" + SUB_TEST_NAME + "/"
    plot(test_data, folder)