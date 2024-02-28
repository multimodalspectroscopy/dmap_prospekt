import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from functools import partial
from sklearn.linear_model import LinearRegression
from mms_nirs.UCLN import DefaultValues
from mms_nirs.utils import ExtinctionCoefficients, calc_dpf,calc_attenuation_slope
from scipy.interpolate import interp1d

def calculate_concentrations(spectra_file_path: str, 
                             wavelengths_file_path: str, 
                             optode_dist: float, 
                             dpf: float) -> pd.DataFrame:
    
    # Load spectra and wavelengths
    spectra = np.loadtxt(spectra_file_path, delimiter=',')
    spectra_wavelengths = np.loadtxt(wavelengths_file_path, delimiter='\t')

    # Load extinction coefficients
    epsilon_labview_771to906 = DefaultValues().extinction_coefficients
    extinction_coefficients_inv = np.linalg.pinv(epsilon_labview_771to906)
    wavelength_dependency = DefaultValues().wavelength_dependency
    print('epsilon_labview_771to906', epsilon_labview_771to906)
    print('wavelength_dependency', wavelength_dependency)
    # Interpolate values to extinction wavelengths
    interp_wavelengths = np.arange(780, 901)
    
    # Preallocate arrays
    n_spectra = spectra.shape[0]
    attenuation = np.zeros(spectra.shape)
    attenuation_interp = np.zeros((interp_wavelengths.size, n_spectra))
    
    for i in range(n_spectra):
        attenuation[i, :] = np.log10(spectra[0, :] / spectra[i, :])
        attenuation_interp[:, i] = interp1d(spectra_wavelengths, attenuation[i, :].T,
                                            kind="cubic", fill_value='extrapolate')(interp_wavelengths)
    
    # Reshape wavelength_dependency array
    wavelength_dependency_reshaped = wavelength_dependency.reshape(1, -1)
    
    # Perform the division
    attenuation_interp_wavelength_dependency = np.divide(attenuation_interp.T, wavelength_dependency_reshaped)
    
    # Calculate concentrations
    concentrations = np.transpose(np.matmul(extinction_coefficients_inv,
                                             attenuation_interp_wavelength_dependency.T) * (1 / (optode_dist * dpf)))
    
    # Convert concentrations array to a DataFrame
    concentrations_df = pd.DataFrame(concentrations, columns=['HbO2', 'HHb', 'CCO'])
    
    return concentrations_df

# Define file paths
spectra_file_path = "/Users/darshana/Desktop/Github_spectra.csv"
wavelengths_file_path = "/Users/darshana/Desktop/Github_wavelengths.csv"

# Define optode_dist and dpf
optode_dist = 3
dpf = 4.99

# Calculate concentrations
concentrations = []
conc = calculate_concentrations(spectra_file_path, wavelengths_file_path, optode_dist, dpf)
concentrations.append(conc * 1000)
print(concentrations)

# Convert concentrations to DataFrame
concentrations_df = pd.DataFrame(concentrations[0], columns=['HbO2', 'HHb', 'CCO'])

# Define the file path
file_path = '/Users/darshana/Desktop/concentrations_df.xlsx'

# Export concentrations DataFrame to Excel
concentrations_df.to_excel(file_path, index=False)
