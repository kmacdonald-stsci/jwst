import numpy as np
import pytest
import random

from stcal.ramp_fitting import ramp_fit_class
from stcal.ramp_fitting.ramp_fit import ramp_fit
from stcal.ramp_fitting.ols_fit import calc_num_seg
from stcal.ramp_fitting.ols_fit import rows_per_slice

from jwst.datamodels import dqflags
from jwst.datamodels import RampModel

test_dq_flags = dqflags.pixel

GOOD = test_dq_flags["GOOD"]
DO_NOT_USE = test_dq_flags["DO_NOT_USE"]
SATURATED = test_dq_flags["SATURATED"]
JUMP_DET = test_dq_flags["JUMP_DET"]

DELIM = "-" * 70


def test_row_slices():
    # Ensure the rows get sliced as expected
    nslices = random.randint(3, 32)
    nrows = random.randint(100, 1024)
    rslices = rows_per_slice(nslices, nrows)

    assert(len(rslices) == nslices)  # Check for the correct number of slices
    assert(sum(rslices) == nrows)    # Check slices have all the rows


def test_basic():
    """
    Run the same data through single processor and multiprocessing, then
    compare the results to make sure they are the same.
    """

    # Single processor
    ramp_model, rnoise, gain, dims = base_data()
    image_info_s, integ_info_s, opt_info_s, gls_dummy_s = ramp_fit(
        ramp_model, 512, True, rnoise, gain, 'OLS', 'optimal', 'none', test_dq_flags)

    # Multi-processor
    ramp_model, rnoise, gain, dims = base_data()
    image_info_n, integ_info_n, opt_info_n, gls_dummy_n = ramp_fit(
        ramp_model, 512, True, rnoise, gain, 'OLS', 'optimal', 'half', test_dq_flags)

    assert_image_close(image_info_s, image_info_n)
    assert_integ_close(integ_info_s, integ_info_n)
    assert_opt_close(opt_info_s, opt_info_n)


def test_do_not_use():
    """
    Run the same data through single processor and multiprocessing, then
    compare the results to make sure they are the same.
    """

    # Single processor
    ramp_model, rnoise, gain, dims = base_data()
    ngroups = dims[1]
    bad_group = ngroups // 2
    ramp_model.groupdq[0, bad_group, 0, 0] = DO_NOT_USE

    image_info_s, integ_info_s, opt_info_s, gls_dummy_s = ramp_fit(
        ramp_model, 512, True, rnoise, gain, 'OLS', 'optimal', 'none', test_dq_flags)

    # Multi-processor
    ramp_model, rnoise, gain, dims = base_data()
    ngroups = dims[1]
    bad_group = ngroups // 2
    ramp_model.groupdq[0, bad_group, 0, 0] = DO_NOT_USE

    image_info_n, integ_info_n, opt_info_n, gls_dummy_n = ramp_fit(
        ramp_model, 512, True, rnoise, gain, 'OLS', 'optimal', 'half', test_dq_flags)

    assert_image_close(image_info_s, image_info_n)
    assert_integ_close(integ_info_s, integ_info_n)
    assert_opt_close(opt_info_s, opt_info_n)


# ------------------------------------------------------------------------------

def base_data():
    dims = (2, 10, 33, 32)
    rnoise, gain, nframes, group_time, frame_time = 10.34, 5.5, 1, 1.0, 10.0

    # Create basic models
    ramp_model, rnoise, gain = setup_inputs(
        dims, rnoise, gain, nframes, group_time, frame_time)

    # Set up basic data array to be used for ramp fitting
    base = 0.1
    current = 0.0
    base_arr = np.array([k+1 for k in range(dims[1])], dtype=np.float32)
    for nint in range(dims[0]):
        for row in range(dims[2]):
            for col in range(dims[3]):
                ramp_arr = base_arr * current
                ramp_model.data[nint, :, row, col] = ramp_arr
                current = current + base

    # Add a jump detection
    ramp_model.groupdq[0, dims[1] // 3, 1, 1] = JUMP_DET

    return ramp_model, rnoise, gain, dims


def assert_image_close(image_s, image_n):

    # Get arrays from single and multiprocessing tuples
    sdata, sdq, svar_poisson, svar_rnoise, serr = image_s
    ndata, ndq, nvar_poisson, nvar_rnoise, nerr = image_n

    tol = 2e-5  # Set tolerance

    # Check single and mulitprocessing gets the same answer
    np.allclose(sdata, ndata, rtol=tol, atol=tol)
    np.allclose(sdq, ndq, rtol=tol, atol=tol)
    np.allclose(svar_poisson, nvar_poisson, rtol=tol, atol=tol)
    np.allclose(svar_rnoise, nvar_rnoise, rtol=tol, atol=tol)
    np.allclose(serr, nerr, rtol=tol, atol=tol)


def assert_integ_close(integ_s, integ_n):

    # Get arrays from single and multiprocessing tuples
    sdata, sdq, svar_poisson, svar_rnoise, sint_times, serr = integ_s
    ndata, ndq, nvar_poisson, nvar_rnoise, nint_times, nerr = integ_n

    tol = 2e-5  # Set tolerance

    # Check single and mulitprocessing gets the same answer
    np.allclose(sdata, ndata, rtol=tol, atol=tol)
    np.allclose(sdq, ndq, rtol=tol, atol=tol)
    np.allclose(svar_poisson, nvar_poisson, rtol=tol, atol=tol)
    np.allclose(svar_rnoise, nvar_rnoise, rtol=tol, atol=tol)
    np.allclose(serr, nerr, rtol=tol, atol=tol)

    print(f"sint_times = {sint_times}")
    print(f"nint_times = {nint_times}")


def assert_opt_close(opt_s, opt_n):

    # Get arrays from single and multiprocessing tuples
    (sslope, ssigslope, svar_poisson, svar_rnoise,
        syint, ssigyint, spedestal, sweights, scrmag) = opt_s
    (nslope, nsigslope, nvar_poisson, nvar_rnoise,
        nyint, nsigyint, npedestal, nweights, ncrmag) = opt_n

    tol = 2e-5  # Set tolerance

    # Check single and mulitprocessing gets the same answer
    np.allclose(sslope, nslope, rtol=tol, atol=tol)
    np.allclose(ssigslope, nsigslope, rtol=tol, atol=tol)
    np.allclose(svar_poisson, nvar_poisson, rtol=tol, atol=tol)
    np.allclose(svar_rnoise, nvar_rnoise, rtol=tol, atol=tol)
    np.allclose(syint, nyint, rtol=tol, atol=tol)
    np.allclose(ssigyint, nsigyint, rtol=tol, atol=tol)
    np.allclose(spedestal, npedestal, rtol=tol, atol=tol)
    np.allclose(sweights, nweights, rtol=tol, atol=tol)
    np.allclose(scrmag, ncrmag, rtol=tol, atol=tol)


def print_total_information(image_info, integ_info, opt_info):
    print(DELIM)
    image_information(image_info)
    print(DELIM)
    integ_information(integ_info)
    print(DELIM)
    opt_infornmation(opt_info)
    print(DELIM)


def image_information(image_info):
    data, dq, var_poisson, var_rnoise, err = image_info

    print("    --> Primary product")
    print(f"data.shape = {data.shape}")
    print(f"dq.shape = {dq.shape}")
    print(f"var_poisson.shape = {var_poisson.shape}")
    print(f"var_rnoise.shape = {var_rnoise.shape}")
    print(f"err.shape = {err.shape}")



def integ_information(integ_info):
    data, dq, var_poisson, var_rnoise, int_times, err = integ_info

    print("    --> Integration product")
    print(f"data.shape = {data.shape}")
    print(f"dq.shape = {dq.shape}")
    print(f"var_poisson.shape = {var_poisson.shape}")
    print(f"var_rnoise.shape = {var_rnoise.shape}")
    print(f"int_times = {int_times}")
    print(f"err.shape = {err.shape}")


def opt_infornmation(opt_info):
    (slope, sigslope, var_poisson, var_rnoise,
        yint, sigyint, pedestal, weights, crmag) = opt_info

    print("    --> Optional product")
    print(f"slope.shape = {slope.shape}")
    print(f"sigslope.shape = {sigslope.shape}")
    print(f"var_poisson.shape = {var_poisson.shape}")
    print(f"var_rnoise.shape = {var_rnoise.shape}")
    print(f"yint.shape = {yint.shape}")
    print(f"sigyint.shape = {sigyint.shape}")
    print(f"pedestal.shape = {pedestal.shape}")
    print(f"weights.shape = {weights.shape}") 
    print(f"crmag.shape = {crmag.shape}")



# Need test for multi-ints near zero with positive and negative slopes
def setup_inputs(
        dims, rnoise, gain, nframes, group_time, frame_time):

    # Get dimensions
    nints, ngroups, nrows, ncols = dims

    # Create model arrays
    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    gdq = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8) * GOOD
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    int_times = np.zeros((nints,))

    # Create noise arrays
    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
    rnoise = np.full((nrows, ncols), rnoise, dtype=np.float32)

    # Create RampModel
    model1 = RampModel(
        data=data, err=err, pixeldq=pixdq, groupdq=gdq, int_times=int_times)
    model1.meta.instrument.name = 'MIRI'
    model1.meta.instrument.detector = 'MIRIMAGE'
    model1.meta.instrument.filter = 'F480M'

    model1.meta.observation.date = '2015-10-13'

    model1.meta.exposure.type = 'MIR_IMAGE'
    model1.meta.exposure.group_time = group_time
    model1.meta.exposure.frame_time = frame_time
    model1.meta.exposure.ngroups = ngroups
    model1.meta.exposure.nframes = nframes
    model1.meta.exposure.groupgap = 0
    model1.meta.exposure.drop_frames1 = 0

    model1.meta.subarray.name = 'FULL'
    model1.meta.subarray.xstart = 1
    model1.meta.subarray.ystart = 1
    model1.meta.subarray.xsize = ncols
    model1.meta.subarray.ysize = nrows

    return model1, rnoise, gain
