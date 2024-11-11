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
from scipy.optimize import brentq,minimize
from multiprocessing import Pool

from pathlib import Path

###############################################################################
## DATA ##
###############################################################################

# Read in SPARC rotation curve data
names = np.loadtxt(
    Path("data/rotationCurvesSPARC.txt"),  # Without 'Path' directories work differently on Windows/Linux
    dtype=str, unpack=True, usecols=(0)
)
R, Vobs, e_Vobs, Vg, Vs, Vb = np.loadtxt(
    str(Path("data/rotationCurvesSPARC.txt")),
    dtype=float, unpack=True, usecols=(2, 3, 4, 5, 6, 7)
)  # Radial points, and individual rotational velocity components


def SPARC1(galaxy_names):
    # extract photometry data for a given galaxy in the SPARC sample
    # returns:
    # Inc = inclination, Lum = total 3.6 micron luminosity, e_Lum = Luminosity error, SBright = surface brightness, HImass = Ionised hydrogen mass, Q = quality factor
    names = np.loadtxt(
        Path("data/centralLumSPARC.txt"), 
        dtype=str, unpack=True, usecols=(0)
    )
    I, L, e_L, r, SB, MHI, QF = np.loadtxt(
        Path("data/centralLumSPARC.txt"), 
        dtype=float, unpack=True, usecols=(5, 7, 8, 11, 12, 13, 17)
    )
    Lum = np.array([])
    e_Lum = np.array([])
    SBright = np.array([])
    HImass = np.array([])
    Q = np.array([])
    Inc = np.array([])
    r0 = np.array([])
    for i in range(len(galaxy_names)):
        galaxy_name = galaxy_names[i]
        for j in range(len(names)):
            if galaxy_name == names[j]:
                Inc = np.append(Inc, I[j])
                Lum = np.append(Lum, L[j])
                e_Lum = np.append(e_Lum, e_L[j])
                SBright = np.append(SBright, np.log10(SB[j]))
                HImass = np.append(HImass, MHI[j])
                Q = np.append(Q, QF[j])
                r0 = np.append(r0, (r[j]))
    return Inc, Lum, e_Lum, r0, SBright, HImass, Q


def SPARC2(galaxy_name):
    # extract SPARC rotation curve data
    # returns:
    # r_list = radial points, Vtot_data = total velocity, Verr = tota velocity uncertainty, Vgas = gas velocity, Vstar = stellar velocity, Vbulge = bulge velocity
    r_list = np.array([])
    Vtot_data = np.array([])
    Verr = np.array([])
    Vgas = np.array([])
    Vstar = np.array([])
    Vbulge = np.array([])
    for i in range(len(names)):
        if str(names[i]) == galaxy_name:
            r_list = np.append(r_list, R[i])
            Vtot_data = np.append(Vtot_data, Vobs[i])
            Verr = np.append(Verr, e_Vobs[i])
            Vgas = np.append(Vgas, Vg[i])
            Vstar = np.append(Vstar, Vs[i])
            Vbulge = np.append(Vbulge, Vb[i])
    return r_list, Vtot_data, Verr, Vgas, Vstar, Vbulge


def RC(galaxy_names):
    # extract SPARC rotation curve data for a given galaxy
    # returns:
    # r_list = radial points, Vtot_data = total velocity, Verr = tota velocity uncertainty, Vgas = gas velocity, Vstar = stellar velocity, Vbulge = bulge velocity
    r_list = [[] for i in range(len(galaxy_names))]
    Vtot_data = [[] for i in range(len(galaxy_names))]
    Verr = [[] for i in range(len(galaxy_names))]
    Vgas = [[] for i in range(len(galaxy_names))]
    Vstar = [[] for i in range(len(galaxy_names))]
    Vbulge = [[] for i in range(len(galaxy_names))]
    for i in range(len(galaxy_names)):
        galaxy_name = galaxy_names[i]
        r, Vdata, Ve, Vg, Vs, Vb = SPARC2(galaxy_name)
        r_list[i].append(r)
        Vtot_data[i].append(Vdata)
        Verr[i].append(Ve)
        Vgas[i].append(Vg)
        Vstar[i].append(Vs)
        Vbulge[i].append(Vb)
    return r_list, Vtot_data, Verr, Vgas, Vstar, Vbulge


def TF():
    # Extract Tully-Fisher Data
    # returns:
    # TF_names = names of galaxies in TF data, TF_M = log total baryonic mass, TF_V = flat rotational velocity
    TF_names = np.loadtxt(
        Path('data/baryonic Tully-Fisher relation.txt'),
        dtype=str, unpack=True, usecols=(0)
    )
    TF_M, TF_V = np.loadtxt(
        Path('data/baryonic Tully-Fisher relation.txt'), 
        dtype=float, unpack=True, usecols=(1, 11)
    )
    return TF_names, TF_M, TF_V

###############################################################################
## GLOBAL VALUES ##
###############################################################################

# SWM 11/11/24: Added `ndmin` or it thinks 1-long lists of names are 0-dimensional
galaxy_names = np.loadtxt(
    Path("data/FinalGalaxySample.txt"), dtype=str, ndmin=1
)
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

"Limits for optimal parameters, ready for least squares fitting and priors"
"Variable_lim = np.array([lower bound,upper bound]):"

# SPARC main papers specifies the Mass-to-Light ratio ranges (Katz et al)
MLs_lim = np.array([0.1, 10])
Mhalo_log_lim = np.array([8, 14])  # Halo mass prior
c_lim = np.array([1, 100])  # concentration prior

###############################################################################
## THEORETICAL MASS AND VELOCITY CURVES ##
###############################################################################


def Rvir_calc(Mhalo_log):
    # Virial radius of the galaxy for a NFW profile.
    HH = 0.7  # Scaled Hubble constant in (km/s)/kpc
    rho_c = (3 * (HH**2)) / (8 * np.pi * G)  # Critical density
    k = (4*np.pi) / 3
    delta = 18*np.pi**2  # redshipt dependent critical density for z=0 #Bryan+Norman 1997 et al
    # Virial radius in kpc
    Rvir_result = np.cbrt(((10 ** Mhalo_log) / (rho_c * delta * k)))
    return Rvir_result

def Mdm_DC14_calc(galaxy_name, r_input, Mhalo_log, c, MLs):
    # Calculation for the log of the dark matter halo mass out to the last radial point for the DC14 model
    Mdm_result = []
    Rvir = Rvir_calc(Mhalo_log)  # Virial radius
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
    for i in range(len(r_input)):
        r = r_input[i]
        temp1 = quad(lambda x: ((x**2) / (((x / r_s)**gamma) * (1+(x / r_s)**alpha)**((beta - gamma) / alpha))), 0, r)
        Mdm_result.append(np.log10(4 * math.pi * Ps * temp1[0]))
    return Mdm_result


def Vdm_calc(galaxy_name, r_input, Mhalo_log, c, MLs, r_c, n, alpha):
    Mdm = Mdm_DC14_calc(galaxy_name, r_input, Mhalo_log, c, MLs)
    Vdm = []
    for i in range(len(Mdm)):
        Vdm.append(np.sqrt((G * 10**Mdm[i]) / r_input[i]))  # Equation 1, H. Katz et al. 2017
    return Vdm

def Vtot_calc(r_input, Mhalo_log, c, MLs):
    # Calculates the total rotation curve for NFW and DC14
    Vtot = []
    Vdm = (Vdm_calc(galaxy_name, r_input, Mhalo_log, c, MLs, 0,  0, 0))
    for i in range(len(Vdm)):
        Vtot.append(np.sqrt(Vdm[i]**2 + MLs*Vstar[i]**2 + np.sign(Vgas[i])*Vgas[i]**2))
    return Vtot

def res(Vdata, Vcalc):
    # Residuals in velocity calculations at each radii
    res = []
    for i in range(len(Vdata)):
        res.append(Vcalc[i] - Vdata[i])
    return res

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
        popt, pcov = op.curve_fit(Vtot_calc, r_list, Vtot_data, sigma=Verr, p0=(11, 10, 0.5), bounds=(
            [Mhalo_log_lim[0], c_lim[0], MLs_lim[0]], [Mhalo_log_lim[1], c_lim[1], MLs_lim[1]]), max_nfev=20000, method='trf')
        Res = res(Vtot_data, Vtot_calc(r_list, *popt))
# plots rotation curves if plot=True:
    if plot == True:
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
                Path("results/" + model + galaxy_name + " Rotation Curve LSF")
            )
            plt.show()
    if model == 'NFW' or model == 'DC14':
        return popt, max(Vtot_calc(r_list, *popt)), np.sqrt(np.diag(pcov))

###############################################################################
## MCMC ##
###############################################################################


def loglike(theta, r_input, Vtot_data, Verr):
    # Log liklihood function
    if model == 'NFW' or model == 'DC14':
        Mhalo_log, c, MLs = theta[0], theta[1], theta[2]
        modelV = Vtot_calc(r_list, Mhalo_log, c, MLs)
    return -0.5*(np.sum(np.power(Vtot_data-modelV, 2) * np.divide(1.0, np.power(Verr, 2))))

'Select the correct set of priors:'

def logprior(theta):
    # Flat priors on all parameters with a log normal prior on mass-to-light ratio given by P.Li et al 2019
    if model == 'NFW' or model == 'DC14':
        Mhalo_log, c, MLs = theta
        if Mhalo_log_lim[0] < Mhalo_log < Mhalo_log_lim[1] and MLs_lim[0] < MLs < MLs_lim[1] and c_lim[0] < c < c_lim[1]:
            # wiki log normal distribution page
            mu = np.log(0.5**2 / np.sqrt(0.5**2 + (0.11)**2))
            sigma = np.sqrt(np.log(1 + ((0.11)**2)/(0.5)**2))  # McGaugh 2016
            return np.log(1.0/(np.sqrt(2*np.pi)*sigma*MLs))-0.5*(np.log(MLs)-mu)**2/(sigma**2)

    return -np.inf

def logprob(theta, r_input, Vtot_input, Verr):
    # log probability function
    if model == 'NFW' or model == 'DC14':
        p = theta[0], theta[1], theta[2]
        p_input = np.array([])
        p_input = np.append(p_input, p)
        lp = logprior(p_input)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglike(theta, r_input, Vtot_input, Verr)


def MCMC_Fit():
    # corner plots and chains for NFW and DC14 profile
    ndim, nwalkers = 3, 200
    popt = opline(galaxy_name, False)[0]
    pos = [popt + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    if __name__ == '__main__':
        with Pool(8) as pool:
            print("X")
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, pool = pool, args=(r_list, Vtot_data, Verr), a=3)#, threads=20))
            print("X")
            samples = sampler.run_mcmc(pos, 1000, progress=True)
            # samples = sampler.run_mcmc(pos, 10000, progress=True)

    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
    sampler_chain = sampler.get_chain(discard=100)
    # samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))
    # sampler_chain = sampler.get_chain(discard=1000)
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
        Path("results/" + model + galaxy_name + " MCMC - Speeadtest")
    )
    plt.show()
    logMh, c, ML = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(sampler.flatchain, [16, 50, 84], axis=0)))
    return logMh, c, ML  # MCMC parameter estimates

###############################################################################
# MCMC RESULTS #
###############################################################################

def MCMC_files(galaxy_names):
    # Saves MCMC results to txt files to be extracted later for plotting
    global galaxy_name
    global r_list
    global Vtot_data
    global Verr
    global Vgas
    global Vstar
    global H1M
    global L
    Mh, H_max, H_min = np.array([]), np.array([]), np.array([])
    c, c_max, c_min = np.array([]), np.array([]), np.array([])
    ML, ML_max, ML_min = np.array([]), np.array([]), np.array([])
    r, Vtot, Ve, Vg, Vs, Vb = RC(galaxy_names)
    for i in range(len(galaxy_names)):
        galaxy_name = galaxy_names[i]
        r_list = r[i][0]
        Vtot_data = Vtot[i][0]
        Verr = Ve[i][0]
        Vgas = Vg[i][0]
        Vstar = Vs[i][0]
        H1M = HImass[i]
        L = Lum[i]
        if model == "NFW":
            mcmc = MCMC_Fit()
            Mh_SPARC, c_SPARC, ML_SPARC = mcmc[0], mcmc[1], mcmc[2]
            Mh_est, Hmax, Hmin = Mh_SPARC[0], Mh_SPARC[1], Mh_SPARC[2]
            cest, cmax, cmin = c_SPARC[0], c_SPARC[1], c_SPARC[2]
            MLest, MLmax, MLmin = ML_SPARC[0], ML_SPARC[1], ML_SPARC[2]
            Mh, H_max, H_min = np.append(Mh, Mh_est), np.append(
                H_max, Hmax), np.append(H_min, Hmin)
            c, c_max, c_min = np.append(c, cest), np.append(
                c_max, cmax), np.append(c_min, cmin)
            ML, ML_max, ML_min = np.append(ML, MLest), np.append(
                ML_max, MLmax), np.append(ML_min, MLmin)
            file = open(
                Path("results/NFW_MCMC_Katz.txt"), "w"
            )
            content = np.column_stack(
                (Mh, H_max, H_min, c, c_max, c_min, ML, ML_max, ML_min))
            np.savetxt(
                Path("results/NFW_MCMC_Katz.txt"),
                content, fmt='%.18e', delimiter=' '
            )
            file.close()
        elif model == "DC14":
            mcmc = MCMC_Fit()
            Mh_SPARC, c_SPARC, ML_SPARC = mcmc[0], mcmc[1], mcmc[2]
            Mh_est, Hmax, Hmin = Mh_SPARC[0], Mh_SPARC[1], Mh_SPARC[2]
            cest, cmax, cmin = c_SPARC[0], c_SPARC[1], c_SPARC[2]
            MLest, MLmax, MLmin = ML_SPARC[0], ML_SPARC[1], ML_SPARC[2]
            Mh, H_max, H_min = np.append(Mh, Mh_est), np.append(
                H_max, Hmax), np.append(H_min, Hmin)
            c, c_max, c_min = np.append(c, cest), np.append(
                c_max, cmax), np.append(c_min, cmin)
            ML, ML_max, ML_min = np.append(ML, MLest), np.append(
                ML_max, MLmax), np.append(ML_min, MLmin)
            file = open(
                Path("results/DC14_MCMC_MCMC_speedtest.txt"), "w"
            )
            content = np.column_stack(
                (Mh, H_max, H_min, c, c_max, c_min, ML, ML_max, ML_min))
            np.savetxt(
                Path("results/DC14_MCMC_MCMC_speeadtest.txt"),
                content, fmt='%.18e', delimiter=' '
            )
            file.close()
    

model = "DC14"

MCMC_files(galaxy_names)


