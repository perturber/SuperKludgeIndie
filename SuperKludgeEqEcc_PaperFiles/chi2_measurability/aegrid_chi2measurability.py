import numpy as np
from itertools import product
import os
import h5py
from tqdm import tqdm

#import SEF
from stableemrifisher.fisher import StableEMRIFisher

#import FEW
from few.utils.utility import get_p_at_t #function to get p0 for a given inspiral time
from few.trajectory.inspiral import EMRIInspiral #trajectory generator
from few.trajectory.ode.flux import SuperKludgeFlux, KerrEccEqFlux #trajectory modules for SK and Kerr adiabatic equatorial models, respectively
from few.waveform import GenerateEMRIWaveform #few waveform generator
from few.waveform.waveform import SuperKludgeWaveform #waveform module for SK

#response wrapper
from fastlisaresponse import ResponseWrapper

#noise PSD
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens

use_gpu = False #True when running on the cluster

if not use_gpu:
    import few
    cfg_set = few.get_config_setter(reset=True)
    cfg_set.enable_backends('cpu')
    cfg_set.set_log_level('info')

#fixed EMRI parameters
T_LISA = 0.1 #observation time, years. Should be 2.0 for actual runs.
dt = 10.0 #sampling interval, seconds

m1 = 1e6 #MBH mass in solar masses (source frame)
m2 = 10.0 #secondary mass in solar masses (source frame)
x0 = 1.0 #inclination, must be = 1.0 for equatorial model
p0_buffer = 0.02 #smol buffer

# initial phases
Phi_phi0 = 0.1 #azimuthal phase
Phi_theta0 = 0.2 #polar phase
Phi_r0 = 0.3 #radial phase

# define the extrinsic parameters
qK = np.pi / 5  # polar spin angle
phiK = np.pi / 6  # azimuthal viewing angle
qS = np.pi / 3  # polar sky angle
phiS = np.pi / 4  # azimuthal viewing angle
dist = 1.0  # distance in Gpc. We'll adjust this later to fix the SNR as SNR_fixed.
SNR_fixed = 50.0 #desired SNR throughout the grid. 

filename = f'aegrid_chi2measurability' #filename
if not os.path.exists(filename):
    os.mkdir(filename)

filename_Fisher = f'Fishers' #subfolder where all the Fisher matrices will be stored.
filename_Fisher = os.path.join(filename,filename_Fisher)

SK_traj = EMRIInspiral(func=SuperKludgeFlux)
adiabatic_traj = EMRIInspiral(func=KerrEccEqFlux)

#set SK add_args here. This is where you remove/add pieces.
chi2 = 0.5 #dimensionless secondary spin
evolve_1PA = True #whether to include 1PA corrections.
evolve_primary = False #whether to include primary's evolution
evolve_2PA = True #whether to include 2PA corrections.
add_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA]

### generator a grid of parameters over (a and e0) ###

N = 10 #no. of points along each axis

try:
    with h5py.File(f"{filename}/data.h5", "r") as f:
        param_grid = f["gridpoints"][:]  # Read the dataset into a NumPy array
        p_range = f["p0"][:] + p0_buffer #buffer
    try:
        with h5py.File(f"{filename}/data.h5", "r") as f:
            dist_range = f["dists"][:]  # Read the dataset into a NumPy array
    except KeyError:
        pass #will be handled later
        
except FileNotFoundError:
    print("generating the a, e grid.")
    #choose parameter ranges here
    a_range = np.linspace(-0.9,0.9,N)
    e_range = np.linspace(0.1,0.5,N) #choose wisely: eccentricity above 0.5 is extremely expensive...
    
    # Generate the Cartesian product of all parameter values
    param_grid = np.array(list((product(a_range, e_range))))
    
    p_range = []
    for i in tqdm(range(len(param_grid))):
        a = param_grid[i,0]
        e0 = param_grid[i,1]

        #generate p0 using the adiabatic trajectory module instead of SK.
        p0 = get_p_at_t(traj_module=adiabatic_traj, t_out=T_LISA, traj_args=[m1, m2, a, e0, x0, Phi_phi0, Phi_theta0, Phi_r0])
        
        p_range.append(p0)

        #check p for plunge trajectories
        #t, p, e, x, pp, pt, pr = kerr_traj(m1, m2, a, p0, e0, x0, Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T_LISA, dt=dt)
    
        #print(t[-1] - (T_LISA*YRSID_SI))
        
    p_range = np.array(p_range)

    print(param_grid.shape, p_range.shape)
    
    #save as an h5py file
    with h5py.File(f"{filename}/data.h5", "w") as f:
        f.create_dataset("gridpoints", data=param_grid)
        f.create_dataset("p0", data=p_range) #save plunging traj p0
        
    p_range += p0_buffer #buffer

### initialize SEF ###

#SK waveform class and kwargs
waveform_class = SuperKludgeWaveform
waveform_class_kwargs =  dict(inspiral_kwargs=dict(err=1e-11,),
                             sum_kwargs=dict(pad_output=True),
                             mode_selector_kwargs=dict(mode_selection_threshold=1e-5))

#waveform generator class and kwargs
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(return_list = False)

#response wrapper class and kwargs
response_generator = ResponseWrapper

tdi_gen = "1st generation"
order = 20
tdi_kwargs_esa = dict(
    orbits=EqualArmlengthOrbits(use_gpu=use_gpu), order=order, tdi=tdi_gen, tdi_chan="AET",
)
index_lambda = 8
index_beta = 7
# with longer signals we care less about this
t0 = 10000.0  # throw away on both ends when our orbital information is weird

ResponseWrapper_kwargs = dict(
    #waveform_gen=waveform_generator, #SEF will automatically include this information
    Tobs = T_LISA,
    dt = dt,
    index_lambda = index_lambda,
    index_beta = index_beta,
    t0 = t0,
    flip_hx = True,
    use_gpu=use_gpu,
    is_ecliptic_latitude=False,
    remove_garbage="zero",
    **tdi_kwargs_esa
)

#noise setup
channels = [A1TDISens, E1TDISens, T1TDISens] #AET TDI channels
noise_model = get_sensitivity
noise_kwargs = [{"sens_fn": channel_i} for channel_i in channels] 

sef = StableEMRIFisher(
                    waveform_class=waveform_class, 
                    waveform_class_kwargs=waveform_class_kwargs,
                    waveform_generator=waveform_generator,
                    waveform_generator_kwargs=waveform_generator_kwargs,
                    ResponseWrapper=response_generator, 
                    ResponseWrapper_kwargs=ResponseWrapper_kwargs,
                    noise_model=noise_model, 
                    noise_kwargs=noise_kwargs, 
                    channels=channels,
                    stats_for_nerds = False, 
                    deriv_type='stable',
                    use_gpu = use_gpu,
                    )

param_names = ['m1','m2','a','chi2','p0','e0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_r0'] #parameters to be varied at each grid point.

add_param_args = {'chi2':chi2,
                  '1PA':evolve_1PA,
                  'evolve_primary':evolve_primary,
                  '2PA':evolve_2PA} #dict of parameters NOT included by default in SEF. This contains the additional SK parameters.  

sef_kwargs = dict(
        param_names = param_names, #name of parameters for which to calculate the Fishers
        add_param_args = add_param_args, #additional parameters in the model
        der_order = 4, #order of finite difference
        Ndelta = 12, #how many grid points for estimating Fisher stability
        stability_plot = False, #whether to plot stability surfaces for each parameter
        filename = filename_Fisher, #folder where to save the Fishers
        plunge_check = False, #whether to check if the Fisher is plunging.
 ) #arguments for SEF at the time of call

emri_kwargs = dict(
    T = T_LISA, dt = dt,
)

### Calculate luminosity distances to ensure SNR = SNR_fixed for all sources.###
try: #dist ranges already available?
    with h5py.File(f"{filename}/data.h5", "r") as f:
        dist_range = f["dists"][:]  # Read the dataset into a NumPy array

except:
    print("scaling the luminosity distances to achieve an SNR of ", SNR_fixed)
    dist_range = []
    for i in tqdm(range(len(param_grid))):
        a = param_grid[i][0]
        e0 = param_grid[i][1]
        p0 = p_range[i]
    
        param_list = [m1, m2, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

        param_list_snr_in = param_list + list(add_param_args.values()) #param_list as you would supply to the waveform generator

        #calculate Fisher
        SNR_before = sef.SNRcalc_SEF(*param_list_snr_in, **emri_kwargs, use_gpu = use_gpu)
        dist_fact = SNR_before/SNR_fixed #adjust distance such that SNR = SNR_fixed
        
        param_list[6] *= dist_fact
        param_list_snr_in[6] = param_list[6]

        dist_range.append(dist * dist_fact)
    
        SNR_after = sef.SNRcalc_SEF(*param_list_snr_in, **emri_kwargs, use_gpu = use_gpu)
    
        print("SNR before: ", SNR_before, "SNR_after: ", SNR_after)
    
    #save as an h5py file
    with h5py.File(f"{filename}/data.h5", "a") as f: #"a" flag for appending existing data file.
        f.create_dataset("dists", data=np.array(dist_range))

### Calculating and saving Fisher matrices along the grid ###
def logmasstransform(Fisher, m1, m2, index_of_m1 = 0, index_of_m2 = 1):
    """ transform m1, m2 -> lnm1, lnm2. Fisher transformation: https://en.wikipedia.org/wiki/Fisher_information """
    
    J = np.eye(len(Fisher))
    J[index_of_m1,index_of_m1] = m1
    J[index_of_m2,index_of_m2] = m2

    return J.T@Fisher@J

#evaluating and saving the Fisher matrices along the grid
for i in tqdm(range(len(param_grid))):

    try:
        #skip Fisher calculation for ones already calculated
        with h5py.File(f"{filename_Fisher}/Fisher_{i}.h5", "r") as f:
            _ = f["Fisher"][:]
        continue

    except FileNotFoundError:

        a = param_grid[i][0]
        e0 = param_grid[i][1]
        p0 = p_range[i]
        dist = dist_range[i]

        sef_kwargs['suffix'] = i #add a suffix to the Fisher filename

        wave_params = dict(
            m1 = m1, 
            m2 = m2, 
            a = a, 
            p0 = p0, 
            e0 = e0, 
            xI0 = x0, 
            dist = dist, 
            qS = qS, 
            phiS = phiS, 
            qK = qK, 
            phiK = phiK, 
            Phi_phi0 = Phi_phi0, 
            Phi_theta0 = Phi_theta0, 
            Phi_r0 = Phi_r0
            )

        #calculate Fisher
        Fisher = sef(wave_params = wave_params, **sef_kwargs) #execute SEF for Fisher calculation

        Fisher_transformed = logmasstransform(Fisher, m1, m2)

        with h5py.File(f"{filename_Fisher}/Fisher_{i}.h5", "a") as f:
            f.create_dataset("Fisher_transformed", data = Fisher_transformed)