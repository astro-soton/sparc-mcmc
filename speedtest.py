###############################################################################
# SPARC Rotation Curve Analysis #
###############################################################################

from __future__ import print_function
from scipy.integrate import odeint
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy import integrate
from scipy import stats
import math
import scipy.optimize as op
import matplotlib.pyplot as plt
import numpy as np
import corner
import emcee
from scipy.integrate import quad
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.interpolate import interp1d
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
plt.rc('font', family='serif')
from scipy.optimize import brentq, minimize
from multiprocessing import Pool
from pandas import DataFrame
import pandas as pd

from pathlib import Path

from typing import Optional, TextIO
from numpy.typing import NDArray
from numpy import float32, float64


###############################################################################
## DATA ##
###############################################################################
SAMPLE_GALAXY_NAMES_PATH: Path = Path("data/FinalGalaxySample.txt")
SPARC_ROTATION_CURVES_PATH: Path = Path("data/rotationCurvesSPARC.txt")
SPARC_LUMINOSITIES_PATH: Path = Path("data/centralLumSPARC.txt")
TULLY_FISHER_PATH: Path = Path("data/baryonic Tully-Fisher relation.txt")

# Read in SPARC rotation curve data
names = np.loadtxt(
    SPARC_ROTATION_CURVES_PATH,
    dtype=str,
    unpack=True,
    usecols=(0),
)
R, Vobs, e_Vobs, Vg, Vs, Vb = np.loadtxt(
    SPARC_ROTATION_CURVES_PATH,
    dtype=float,
    unpack=True,
    usecols=(2, 3, 4, 5, 6, 7),
)  # Radial points, and individual rotational velocity components


df_sample_galaxies: DataFrame = pd.read_csv(
    SAMPLE_GALAXY_NAMES_PATH,
    names=(
        "name",
    )
)

df_sparc_rotation_curves: DataFrame = pd.read_csv(
    SPARC_ROTATION_CURVES_PATH,
    sep=r'\s+',
    usecols=(0, 2, 3, 4, 5, 6, 7),
    names=(
        "name",
        "radius",
        "vel_obs",
        "vel_obs_err",
        "vel_gas",
        "vel_stellar",
        "vel_bulge",
    )
)

df_sparc_luminosities: DataFrame = pd.read_csv(
    SPARC_LUMINOSITIES_PATH,
    sep=r'\s+',
    usecols=(0, 5, 7, 8, 11, 12, 13, 17),
    names=(
        "name",
        "inclination",
        "lum",  # 3.6 micron band luminosity
        "lum_err",  # Luminosity error
        "r_0",
        "surf_brightness",  # Surface brightness
        "h_ion_mass",  # H-I Ion mass
        "qual_factor",  # Qualify factor
    )
)

df_baryon_tf_relation: DataFrame = pd.read_csv(
    TULLY_FISHER_PATH,
    sep=r'\s+',
    usecols=(0, 1, 11),
    names=(
        "name",
        "log_mass",
        "vel",
    )
)


def SPARC1(sample_galaxy_names):
    # extract photometry data for a given galaxy in the SPARC sample
    # returns:
    # Inc = inclination, Lum = total 3.6 micron luminosity, e_Lum = Luminosity error, SBright = surface brightness, HImass = Ionised hydrogen mass, Q = quality factor
    sparc_galaxy_names = np.loadtxt(
        SPARC_LUMINOSITIES_PATH, 
        dtype=str, unpack=True, usecols=(0)
    )
    I, L, e_L, r, SB, MHI, QF = np.loadtxt(
        SPARC_LUMINOSITIES_PATH, 
        dtype = float, unpack = True, usecols = (5, 7, 8, 11, 12, 13, 17)
    )
    Lum = np.array([])
    e_Lum = np.array([])
    SBright = np.array([])
    HImass = np.array([])
    Q = np.array([])
    Inc = np.array([])
    r0 = np.array([])

    for sample_galaxy_name in sample_galaxy_names:
        for idx, sparc_galaxy_name in enumerate(sparc_galaxy_names):
            if sparc_galaxy_name == sample_galaxy_name:
                Inc = np.append(Inc, I[idx])
                Lum = np.append(Lum, L[idx])
                e_Lum = np.append(e_Lum, e_L[idx])
                SBright = np.append(SBright, np.log10(SB[idx]))
                HImass = np.append(HImass, MHI[idx])
                Q = np.append(Q, QF[idx])
                r0 = np.append(r0, (r[idx]))
    
    return Inc, Lum, e_Lum, r0, SBright, HImass, Q


def SPARC2(sample_galaxy_name):
    # extract SPARC rotation curve data
    # returns:
    # r_list = radial points, Vtot_data = total velocity, Verr = tota velocity uncertainty, Vgas = gas velocity, Vstar = stellar velocity, Vbulge = bulge velocity
    #     
    r_list = np.array([])
    Vtot_data = np.array([])
    Verr = np.array([])
    Vgas = np.array([])
    Vstar = np.array([])
    Vbulge = np.array([])

    for idx, sparc_galaxy_name in enumerate(names):
        if sparc_galaxy_name == sample_galaxy_name:
            r_list = np.append(r_list, R[idx])
            Vtot_data = np.append(Vtot_data, Vobs[idx])
            Verr = np.append(Verr, e_Vobs[idx])
            Vgas = np.append(Vgas, Vg[idx])
            Vstar = np.append(Vstar, Vs[idx])
            Vbulge = np.append(Vbulge, Vb[idx])

    return r_list, Vtot_data, Verr, Vgas, Vstar, Vbulge



def RC(sample_galaxy_names):
    # extract SPARC rotation curve data for a given galaxy
    # returns:
    # r_list = radial points, Vtot_data = total velocity, Verr = tota velocity uncertainty, Vgas = gas velocity, Vstar = stellar velocity, Vbulge = bulge velocity

    num_galaxies: int = len(sample_galaxy_names)
    r_list = [[]] * num_galaxies
    Vtot_data = [[]] * num_galaxies
    Verr = [[]] * num_galaxies
    Vgas = [[]] * num_galaxies
    Vstar = [[]] * num_galaxies
    Vbulge = [[]] * num_galaxies

    for idx, sample_galaxy_name in enumerate(sample_galaxy_names):
        r, Vdata, Ve, Vg, Vs, Vb = SPARC2(sample_galaxy_name)
        r_list[idx] = [r]
        Vtot_data[idx] = [Vdata]
        Verr[idx] = [Ve]
        Vgas[idx] = [Vg]
        Vstar[idx] = [Vs]
        Vbulge[idx] = [Vb]

    return r_list, Vtot_data, Verr, Vgas, Vstar, Vbulge


def TF():
    # Extract Tully-Fisher Data
    # returns:
    # TF_names = names of galaxies in TF data, TF_M = log total baryonic mass, TF_V = flat rotational velocity
    TF_names = np.loadtxt(
        TULLY_FISHER_PATH,
        dtype=str, unpack=True, usecols=(0)
    )
    TF_M, TF_V = np.loadtxt(
        TULLY_FISHER_PATH,
        dtype=float, unpack=True, usecols=(1, 11)
    )
    return TF_names, TF_M, TF_V


###############################################################################
## GLOBAL VALUES ##
###############################################################################

galaxy_names = np.loadtxt(SAMPLE_GALAXY_NAMES_PATH, dtype=str, ndmin=1)

Inc, Lum, e_Lum, r0, SBright, HImass, Q = SPARC1(galaxy_names)
r_list, Vtot_data, Verr, Vgas, Vstar, Vbulge = RC(galaxy_names)
TF_names, TF_M, TF_V = TF()
cosmology.setCosmology('planck18')
HH = cosmology.getCurrent().H0
h = HH/100
G = 4.302*(10**(-6))  # The gravitational constant in Kpc/Msun(km/s)^2
redshift = 0


###############################################################################
## MODELS ##
###############################################################################
"""
'NFW' - NFW
'cNFW' - Cored - NFW (Newman et al)
'DC14' - Di Cinto et al. 2014 (Feedback model
'Einasto' - Einasto profile
'Burkert' - Burkert 1995 profile
'MOND' - Standard MOND interpolation function
'SIMPLE MOND' - Simplified MOND interpolation function
"""


###############################################################################
## LIMITS ##
###############################################################################
"""
Limits for optimal parameters, ready for least squares fitting and priors"
"Variable_lim = np.array([lower bound,upper bound]):
"""

# SPARC main papers specifies the Mass-to-Light ratio ranges (Katz et al)
MLs_lim = np.array([0.1, 10])
Mhalo_log_lim = np.array([8, 14])  # Halo mass prior
c_lim = np.array([1, 100])  # concentration prior


###############################################################################
## THEORETICAL MASS AND VELOCITY CURVES ##
###############################################################################

# SWM 5/11/24: Take the constant terms and calculations out of the function,
# and do them once so we can refer to them later to save time.
# Not sure if this is necessary -  won't really be if this isn't called in the MCMC or a fitter
CONST_HH: float = 0.7  # Scaled Hubble constant in (km/s)/kpc
CONST_RHO_C: float = (3 * (CONST_HH**2)) / (8 * np.pi * G)  # Critical density
CONST_K: float = (4 * np.pi) / 3
CONST_DELTA: float = 18 * np.pi**2  # redshift dependent critical density for z=0 (Bryan+Norman 1997 et al)

CONST_VIRIAL_RADIUS_DENOMINATOR: float = CONST_RHO_C * CONST_DELTA * CONST_K


def calculate_virial_radius_NFW_model(Mhalo_log):
    """
    Virial radius of the galaxy for a NFW profile.

    :param Mhalo_log: The log mass of the halo (in M_☉?)
    :return: The virial radius of the galaxy in kiloparsecs
    """
    return np.cbrt(((10 ** Mhalo_log) / CONST_VIRIAL_RADIUS_DENOMINATOR))


def Mdm_DC14_calc(galaxy_name, r_input, Mhalo_log, c, MLs):
    # Calculation for the log of the dark matter halo mass out to the last radial point for the DC14 model
    
    Mdm_result: NDArray = np.zeros(len(r_input))
    Rvir = calculate_virial_radius_NFW_model(Mhalo_log)  # Virial radius
    
    X = np.log10(MLs*L*(10**9)) - Mhalo_log  # Extrapolation factor
    if X < -4.1:
        X = -4.1  # extrapolation beyond accurate DC14 range
    if X > -1.3:
        X = -1.3
    
    # transition parameter between the inner slope and outer slope
    alpha = 2.94 - np.log10((10**(X+2.33))**(-1.08) + (10**(X+2.33))**(2.29))
    beta = 4.23 + (1.34 * X) + (0.26 * (X**2))  # outer slope
    gamma = -0.06 + np.log10((10**(X+2.56))**(-0.68) + (10**(X+2.56)))  # inner slope
    
    # Katz say 0.00001 is a better fit than DC's 0.00003
    c_sph = c * (1 + (0.00001 * np.exp(3.4 * (X + 4.5))))
    rm_2 = Rvir / c_sph
    r_s = rm_2 / (((2 - gamma) / (beta - 2))**(1 / alpha))  # Scale radius
    temp1 = quad(lambda x: ((x**2) / (((x / r_s)**gamma) *
                 (1+(x / r_s)**alpha)**((beta - gamma) / alpha))), 0, Rvir)
    
    # Ps (scale density) where M(Rvir) = Mhalo. From Di Cintio et al 2014.
    Ps = (10 ** Mhalo_log) / (4 * math.pi * temp1[0])

    for idx in range(len(r_input)):
        r = r_input[idx]
        temp1 = quad(lambda x: ((x**2) / (((x / r_s)**gamma) * (1+(x / r_s)**alpha)**((beta - gamma) / alpha))), 0, r)
        Mdm_result[idx] = np.log10(4 * math.pi * Ps * temp1[0])

    return Mdm_result


def Vdm_calc(galaxy_name, r_input, Mhalo_log, c, MLs, r_c, n, alpha):
    Mdm = Mdm_DC14_calc(galaxy_name, r_input, Mhalo_log, c, MLs)
    Vdm: NDArray = np.zeros(len(r_input))

    for idx in range(len(Mdm)):
        Vdm[idx] = np.sqrt((G * 10**Mdm[idx]) / r_input[idx])  # Equation 1, H. Katz et al. 2017
    
    return Vdm


def Vtot_calc(r_input, Mhalo_log, c, MLs):
    # Calculates the total rotation curve for NFW and DC14
    Vdm = (Vdm_calc(galaxy_name, r_input, Mhalo_log, c, MLs, 0,  0, 0))
    Vtot: NDArray = np.zeros(len(Vdm))

    for i in range(len(Vdm)):
        Vtot[i] = np.sqrt(Vdm[i]**2 + MLs*Vstar[i]**2 + np.sign(Vgas[i]) * Vgas[i]**2)

    return Vtot


def get_velocity_residuals(
        velocity_observed: NDArray, 
        velocity_calculated: NDArray
) -> NDArray:
    """
    Residuals in velocity calculations at each radii

    :param velocity_observed: Observed velocities at each radius (units ???)
    :param velocity_calculated: Calculated best fit model for velocities at each radius (units ???)
    :return: The residuals of the model fit to the data (units ???)
    """
    # SWM 5/11/24: Assuming these are numpy arrays, we can directly subtract them elementwise
    return velocity_calculated - velocity_observed


###############################################################################
## LEAST SQUARES FITTING ##
###############################################################################

def opline(galaxy_name, plot=True):
    # Returns optimal parameter values and covariances and maximum veocity for a given density profile by fitting theoretical velocities to the observed total velocity data from SPARC
    global r_list
    global Vtot_data
    global Verr
    global Vgas
    global Vstar
    global Vbulge
    global L
    #r_list1 = r_list
    i = np.where(galaxy_name == galaxy_names)
    # Luminosity of given galaxy as a global variable in DC14 halo mass calculation
    L = Lum[i]

    if model == 'NFW' or model == 'DC14':
        popt, pcov = op.curve_fit(
            Vtot_calc, r_list, Vtot_data, 
            sigma=Verr, 
            p0=(11, 10, 0.5), 
            bounds=(
                [Mhalo_log_lim[0], c_lim[0], MLs_lim[0]],
                [Mhalo_log_lim[1], c_lim[1], MLs_lim[1]],
            ), 
            max_nfev=20000, 
            method='trf',
        )
        
        Res = get_velocity_residuals(
            Vtot_data, Vtot_calc(r_list, *popt)
        )

    # plots rotation curves if plot=True:
    if plot:
        if model == 'NFW' or model == 'DC14':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                           gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
            fig.suptitle(model+' '+galaxy_name+' Rotation Curve', fontsize=30)
            ax1.plot(r_list, Vtot_calc(r_list, *popt), c='darkorange',
                     label="Total modelled velocity", linewidth=3)
            ax1.plot(r_list, Vdm_calc(galaxy_name, r_list, *popt, 0, 0, 0), c='hotpink',
                     linestyle='solid', label="Modelled DM velocity", linewidth=3)
            ax1.plot(r_list, Vtot_data, c='blue', linestyle='dotted',
                     label="Total velocity (data)", linewidth=3)
            # stellar velocity curve scaled by mass-to-ight ratio
            ax1.plot(r_list, Vstar*np.sqrt(popt[2]), c='crimson',
                     linestyle='dashed', label="Disk velocity", linewidth=3)
            ax1.plot(r_list, np.absolute(Vgas), c='yellowgreen',
                     linestyle='dashdot', label="Gas velocity", linewidth=3)
            ax1.errorbar(r_list, Vtot_data, yerr=Verr, fmt=".k")
            ax1.legend(loc='lower right', borderaxespad=0, fontsize=10)
            ax1.set_xlabel("Radius - kpc", fontsize=15)
            ax1.set_ylabel("Rotational Velocity - km/s", fontsize=15)
            ax2.plot(r_list, Res, 'o-', linewidth=3)
            ax2.plot(r_list, np.zeros(len(r_list)),
                     color="gray", linestyle='--')
            ax2.set_xlabel("Radius - kpc", fontsize=15)
            ax2.set_ylabel("Residuals - km/s", fontsize=15)
            fig.savefig(
                Path("results") / f"{model}_{galaxy_name}_Rotation Curve LSF"
            )
            plt.show()

    if model in ('NFW', 'DC14'):
        return popt, max(Vtot_calc(r_list, *popt)), np.sqrt(np.diag(pcov))


###############################################################################
## MCMC ##
###############################################################################

def get_log_likeihood(theta):
    """
    Log likelihood function

    :param theta: ???
    :param r_input: ???
    :param Vtot_data: ???
    :param Verr: ???
    :returns: ???
    """
    global r_list, Vtot_data, Verr

    # SWM 5/11/25: If the function should be different for different models, 
    # then we probably want to have different *functions*, 
    # not a switch within the same function that we keep checking every call.
    # if model == 'NFW' or model == 'DC14':
    
    Mhalo_log, c, MLs = theta

    modelV = Vtot_calc(r_list, Mhalo_log, c, MLs)
    return -0.5*(
        np.sum(
            np.power(Vtot_data-modelV, 2) * np.divide(1.0, np.power(Verr, 2))
        )
    )


# SWM 5/11/24: Logs and square roots are expensive, 
# so we take the ones that don't change out, do them once, and keep them as a constant
CONST_MU: float = np.log(0.5**2 / np.sqrt(0.5**2 + (0.11)**2))
CONST_SIGMA: float = np.sqrt(np.log(1 + ((0.11)**2)/(0.5)**2))  # McGaugh 2016

CONST_PRIOR_LOG_DENOMINATOR: float = np.sqrt(2 * np.pi) * CONST_SIGMA  # SWM 5/11/24: I'd rename this to something physically or mathematically meaningful if possible
CONST_PRIOR_SUBTRACT_DENOMINATOR: float = 2 * CONST_SIGMA**2           # SWM 5/11/24: Same as above!


def get_log_prior(theta):
    """
    Log probability for the priors

    :param theta: The MCMC sample for this step, which unpacks to the log halo mass, c??? and MLs???

    :returns: ??? if the log halo mass, ??? and ??? are between the global limits, or -∞ if not.
    """
    # Flat priors on all parameters with a log normal prior on mass-to-light ratio given by P.Li et al 2019
    Mhalo_log, c, MLs = theta

    if Mhalo_log_lim[0] < Mhalo_log < Mhalo_log_lim[1] and MLs_lim[0] < MLs < MLs_lim[1] and c_lim[0] < c < c_lim[1]:
        # wiki log normal distribution page
        return np.log(
            1.0 / (CONST_PRIOR_LOG_DENOMINATOR * MLs)
        ) - (
            np.log(MLs) - CONST_MU
        )**2 / CONST_PRIOR_SUBTRACT_DENOMINATOR

    return -np.inf


def get_log_probability(theta):
    """
    Log probability function

    :param theta: The MCMC sample for this step, which unpacks to the log halo mass, c??? and MLs???

    :return: ??? if finite, or -∞ if not.
    """
    global r_list, Vtot_data, Verr

    # SWM 4/11/25: If theta is already a numpy array, this should avoid converting it to a list and back
    # Annoyingly, the [X:Y] notation *includes* element X, but only up to element Y-1 - so [0:3] is [0, 1, 2]
    log_prior = get_log_prior(
        np.array(theta[0:3])
    )

    if not np.isfinite(log_prior):
        return -np.inf
    
    return log_prior + get_log_likeihood(theta)


def perform_MCMC_fit(
        output_figure_name: str,
        num_threads: Optional[int] = None
):  
    """
    Performs MCMC fit of the galaxy data, and plots corner plots and chains for NFW and DC14 profile

    :param output_figure_name: Modifier to add to the name of the output figure
    :param num_threads: If provided, the number of threads to parallelise over
    :return: The MCMC parameter estimates for ???, ??? and ???
    """
    ndim, nwalkers = 3, 200
    popt = opline(galaxy_name, False)[0]
    pos = [popt + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    if num_threads:
        # If we're parallelising, set up the pool of threads
        with Pool(num_threads) as pool:
            print("perform_MCMC_fit: Declaring sampler")
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, 
                get_log_probability,
                pool=pool,
                a=3,
            )
            print("perform_MCMC_fit: Declared sampler")
            
            samples = sampler.run_mcmc(
                initial_state=pos, nsteps=1000, progress=True
                # initial_state=pos, nsteps=10000, progress=True
            )

    else:
        # Run on a single thread
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim,
            get_log_probability,
            a=3,
        )

        samples = sampler.run_mcmc(
            initial_state=pos, nsteps=1000, progress=True
            # initial_state=pos, nsteps=10000, progress=True
        )

    # samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))
    # sampler_chain = sampler.get_chain(discard=1000)
    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
    sampler_chain = sampler.get_chain(discard=100)
    labels = ["$\\log_{10}(M_{halo})$", "$c_{vir}$", "$\\Upsilon_{*}$"]
    
    plt.clf()
    fig, axes = plt.subplots(ndim, sharex=True, figsize=(8, 9))
    
    for i in range(ndim):
        ax = axes[i]
        ax.plot(sampler_chain[:, :, i], "k", alpha=0.4)
        ax.set_xlim(0, len(sampler_chain))
        ax.set_ylabel(labels[i])
        ax.set_xlabel("step number")
        ax.yaxis.set_label_coords(-0.1, 0.5)

    fig.tight_layout(h_pad=0.0)
    fig = corner.corner(samples, labels=["$\\log_{10}(M_{halo})$", "$c_{vir}$", "$\\Upsilon_{*}$"], show_titles=True, quantiles=[ 0.16, 0.5, 0.84], title_quantiles=[0.16, 0.5, 0.84])  # median and 1 sigma percentiles
    fig.savefig(
        Path("results") / f"{model}_{galaxy_name}_{output_figure_name}"
    )
    plt.show()
    
    logMh, c, ML = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(sampler.flatchain, [16, 50, 84], axis=0)))
    return logMh, c, ML  # MCMC parameter estimates


###############################################################################
# MCMC RESULTS #
###############################################################################
df_galaxy_rotation: DataFrame = None
df_galaxy_luminosity: DataFrame = None


def generate_MCMC_files(
        galaxy_names, 
        output_file_name: str, 
        num_threads: Optional[int] = None
):
    """
    Performs MCMC and saves results to txt files to be extracted later for plotting
    
    :param galaxy_names: List of galaxy files
    :param output_file_name: Name added to the model name for the output file
    :param num_threads: If provided, the number of threads to parallelise over
    """
    global df_galaxy_luminosity
    global df_galaxy_rotation_curves

    global galaxy_name
    global r_list
    global Vtot_data
    global Verr
    global Vgas
    global Vstar
    global H1M
    global L

    # SWM 4/11/24: Using `np.append` actually creates a new array every
    # time you call it, so it's really slow.
    # Replaced it with fixed-length arrays we access using indexes.       
    num_galaxies: int = len(galaxy_names)

    df_results: DataFrame = DataFrame(
        {
            "name": df_sample_galaxies['name'],
            "halo_mass": None,
            "halo_mass_min": None,
            "halo_mass_max": None,
            "c": None,
            "c_min": None,
            "c_max": None,
            "ML": None,
            "ML_min": None,
            "ML_max": None,
        }
    )

    Mh: NDArray = np.zeros(num_galaxies)
    H_max: NDArray = np.zeros_like(Mh) 
    H_min: NDArray = np.zeros_like(Mh)
    
    c: NDArray = np.zeros(num_galaxies)
    c_max: NDArray = np.zeros_like(c)
    c_min: NDArray = np.zeros_like(c)
    
    ML: NDArray = np.zeros(num_galaxies)
    ML_max: NDArray = np.zeros_like(ML)
    ML_min: NDArray = np.zeros_like(ML)

    r, Vtot, Ve, Vg, Vs, Vb = RC(galaxy_names)

    # SWM 5/11/24: Instead of re-saving everything every cycle, we'll open the file once
    # and just append each new entry to it
    output_file: TextIO = open(
        Path("results") / f"{model}_{output_file_name}.txt", "w"
    )

    for idx, galaxy_name in enumerate(galaxy_names):
        df_galaxy_rotation_curve = df_sparc_rotation_curves[
            df_sparc_rotation_curves['name'] == galaxy_name
        ]
        df_galaxy_luminosity = df_sparc_luminosities[
            df_sparc_luminosities['name'] == galaxy_name
        ]

        r_list = r[idx][0]
        Vtot_data = Vtot[idx][0]
        Verr = Ve[idx][0]
        Vgas = Vg[idx][0]
        Vstar = Vs[idx][0]
        H1M = HImass[idx]
        L = Lum[idx]

        Mh_SPARC, c_SPARC, ML_SPARC = perform_MCMC_fit(
            output_figure_name=output_file_name,
            num_threads=num_threads,
        )

        Mh[idx] = Mh_SPARC[0]
        H_max[idx] = Mh_SPARC[1]
        H_min[idx] = Mh_SPARC[2]
        
        c[idx] = c_SPARC[0]
        c_max[idx] = c_SPARC[1]
        c_min[idx] = c_SPARC[2]

        ML[idx] = ML_SPARC[0]
        ML_max[idx] = ML_SPARC[1]
        ML_min[idx] = ML_SPARC[2]

        content = np.array(
            [
                Mh[idx], H_max[idx], H_min[idx],
                c[idx], c_max[idx], c_min[idx],
                ML[idx], ML_max[idx], ML_min[idx],
            ]
        )
    
        # SWM 4/11/24: Only writing the new line to file
        np.savetxt(
            output_file, content, fmt='%.18e', delimiter=' '
        )


# model = "NFW"
# output_file_name = "MCMC_Katz"

model: str = "DC14"
OUTPUT_FILE_NAME: str = "MCMC_speedtest_v2"

NUM_THREADS: int = 8

generate_MCMC_files(
    galaxy_names=galaxy_names, 
    output_file_name=OUTPUT_FILE_NAME,
    num_threads=NUM_THREADS,
)
