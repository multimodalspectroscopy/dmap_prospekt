import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from numpy.linalg import pinv
import os
import glob
import pandas as pd
import itertools
from functools import partial
from sklearn.linear_model import LinearRegression
import scipy.stats
from mms_nirs.utils.attenuation import calc_attenuation_slope, calc_attenuation_spectra
from mms_nirs.utils.extinction_coefficients import ExtinctionCoefficients
import numpy as np
from functools import partial
from SRS import srs_values, calc_conc, calc_k_mua, calc_sto2
from scipy.stats import linregress
from scipy import interpolate


def Data_Pre_Processing(data_path,path_subject,plot_data,conf_det_merged):

    #remove pre-processed file
    if os.path.exists(path_subject+"/output/Input.npz"):
        os.remove(path_subject+"/output/Input.npz")

    #Check if output directory exists
    if not os.path.exists(path_subject+"/output/"):
        os.mkdir(path_subject+"/output")


    subject_name = os.path.basename(path_subject)
    id_subject = int(subject_name[3:7])

    #Check if the directory is non null
    list_csv = glob.glob(path_subject+"/Spectra/*.csv")
    if len(list_csv) ==0:
        return subject_name+"\tNo csv files", id_subject
    

    # Wavelength
    WAVELENGTHS = np.arange(710, 900, 1)
    CYRIL_wavelength = np.genfromtxt(data_path+"raw_wavelengths.csv", delimiter=",")

    #Source detector separation
    detector_id = np.array([1,8,2,7,3,6,4,5]) #disposition of the detectors for SD_separations
    SD_separations = np.arange(10, detector_id.shape[0]*10+1, 10)

    #Load intensity spectra (size: k, T, N), k: nb of detector, T: time, N: nb of wavelength
    Intensity_spectra = []
    Ref = []

    #Get temporal resolution
    file_path = glob.glob(path_subject+"/Spectra/*"+str(detector_id[0])+".csv")[0]
    I = np.genfromtxt(file_path, delimiter=",")
    if I.ndim == 1:
        return subject_name+"\tNo temporal resolution", id_subject

    T = I.shape[0]

    i = 0
    for det_id in detector_id:
        #Load Ref
        path_ref = glob.glob(data_path+ "/CYRILrefs/ref_det"+str(det_id)+"*")[0]
        file_path = glob.glob(path_ref+"/Spectra/*"+str(det_id)+".csv")[0]
        I = np.genfromtxt(file_path, delimiter=",")

        if I.ndim == 2:
            I = I.mean(axis=0)
        Ref.append(interpolate.interp1d(CYRIL_wavelength,I, kind='cubic')(WAVELENGTHS))

        #Load Intensity
        file_path = glob.glob(path_subject+"/Spectra/*"+str(det_id)+".csv")[0]
        if os.stat(file_path).st_size > 0:
            I = np.genfromtxt(file_path, delimiter=",")

            I_interp = []
            if I.ndim == 2:
                #Check temporal resolution
                if I.shape[0] != T:
                    return subject_name+"\tNot the same tempora resolution for det"+str(detector_id[0]), id_subject

                #Interpolate
                for t in range(I.shape[0]):
                    I_interp.append(interpolate.interp1d(CYRIL_wavelength,I[t,:], kind='cubic')(WAVELENGTHS))
            else:
                return subject_name+"\tNo temporal resolution", id_subject

                #I_interp.append(interpolate.interp1d(CYRIL_wavelength,I, kind='cubic')(WAVELENGTHS))

            Intensity_spectra.append(np.asarray(I_interp))
        else:
            return subject_name+"\tEmpty csv file", id_subject

        i=i+1

    #Convert list in ndarray
    Intensity_spectra = np.asarray(Intensity_spectra)
    Ref = np.asarray(Ref)

    # Calculate mean intensity
    Intensity_spectra_mean = Intensity_spectra.mean(axis=1)

    # Remove detectors that cannot be used
    keep_id = np.array([],dtype=int)
    merge_Flag = False
    merged_id = np.array([],dtype=int)

    for id in range(SD_separations.shape[0]):
        # Check if the measured intensity is greater than the ref
        if(np.any(Intensity_spectra_mean[id,:]>Ref[id,:])):
            continue

        # Check if the measured intensity is lower than 5% of the Ref mean signal
        if(np.all(Intensity_spectra_mean[id,:]<Ref[id,:].mean()*0.05)):
            if conf_det_merged:
                merged_id = np.append(merged_id,id)
                continue
            else:
                #Only keep the first detector (intensities for other detectors are merged together)
                if merge_Flag == True:
                    continue
                merge_Flag = True

        keep_id = np.append(keep_id,id)

    # Plot intensities

    # Colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.close('all')
    if plot_data:
        plt.figure()
        plt.suptitle("Intensities measurement "+subject_name+" and ref")
        plt.subplot(131)
        plt.title("Row data")
        for id in range(Intensity_spectra_mean.shape[0]):
            plt.plot(WAVELENGTHS,Intensity_spectra_mean[id,:],color=colors[id],label='SD separation '+str(SD_separations[id]))
            plt.plot(WAVELENGTHS,Ref[id,:],color=colors[id],linestyle=':')
        plt.xlabel("Wavelength (nm)")
        plt.legend(loc="best")

        plt.subplot(132)
        plt.title("After selection")
        for id in keep_id:
            plt.plot(WAVELENGTHS,Intensity_spectra_mean[id,:],color=colors[id],label='SD separation '+str(SD_separations[id]))
            plt.plot(WAVELENGTHS,Ref[id,:],color=colors[id],linestyle=':')
        plt.xlabel("Wavelength (nm)")
        plt.legend(loc="best")

        plt.subplot(133)
        plt.title("Intensity merged")
        for id in merged_id:
            plt.plot(WAVELENGTHS,Intensity_spectra_mean[id,:],color=colors[id],label='SD separation '+str(SD_separations[id]))
            plt.plot(WAVELENGTHS,Ref[id,:],color=colors[id],linestyle=':')
        plt.xlabel("Wavelength (nm)")
        plt.legend(loc="best")

        plt.show()


    # Check data
    if keep_id.size<2:
        return subject_name+" Less than 2 SD sep", id_subject

    # #if there is less than 2 detectors exit function
    # if keep_id.size<2 and merged_id.size ==0 :
    #     # print("Less than 2 SD sep (",SD_separations[keep_id],") ",subject_name)
    #     return


    # Merged intensity
    merged_I = np.array([])
    merged_SD = 0
    if conf_det_merged:
        merged_I = Intensity_spectra[merged_id,:,:].sum(axis=0)
        merged_Ref = Ref[merged_id,:].sum(axis=0)
        merged_SD =SD_separations[merged_id].mean()


    # Only keep selected intensity profiles
    Intensity_spectra = Intensity_spectra[keep_id,:,:]
    Ref = Ref[keep_id,:]
    SD_separations = SD_separations[keep_id]


    # #Add merged intensities (Change here add correction)
    # if conf_det_merged:
    #     SD_separations = np.append(SD_separations,merged_SD)
    #     Intensity_spectra = np.vstack((Intensity_spectra,merged_I))
    #     Ref = np.vstack((Ref,merged_Ref))


    #Save input Data
    np.savez(path_subject+"/output/Input",I=Intensity_spectra,
    Ref_I = Ref,
    SD_separations_in_mm = SD_separations,
    wavelength = WAVELENGTHS,
    merged_I = merged_I,
    merged_Ref = merged_Ref,
    merged_SD_in_mm = merged_SD)

    return subject_name+"\t"+str(T)+"s", id_subject
    # print(Attenuation_spectra.shape)

## Check pre-processed data

path_subject = "/Users/darshana/CYRIL_Charly_121923/PROSPEKT/PMS0002_35+6__020823_105226"

data = np.load(path_subject+"/output/Input.npz")
SD_separations = data['SD_separations_in_mm']
WAVELENGTHS = data['wavelength']
I = data['I']
Ref_I = data['Ref_I']
merged_I = data['merged_I']
merged_Ref = data['merged_Ref']



def compute_SRS(path_subject,chrom,wave_start,wave_end,plot_data):

    #remove output file
    if os.path.exists(path_subject+"/output/SRS.npz"):
        os.remove(path_subject+"/output/SRS.npz")

    #Load pre processed data
    if len(glob.glob(path_subject+"/output/Input.npz")) ==0:
        print("No Input file",path_subject)
        return

    data = np.load(path_subject+"/output/Input.npz")
    SD_separations = data['SD_separations_in_mm']
    WAVELENGTHS = data['wavelength']

    Intensity_spectra = data["I"].mean(axis=1)
    #Ref intensity
    Ref = data["Ref_I"]


    # Compute Attenuation spectra
    Attenuation_spectra = np.log10(np.divide(Ref, Intensity_spectra))


    #Select wavelength range
    id_start = np.where(WAVELENGTHS>=wave_start)[0][0]
    id_end = np.where(WAVELENGTHS<=wave_end)[0][-1]
    
    WAVELENGTHS = WAVELENGTHS[id_start:id_end+1]
    Attenuation_spectra_data = Attenuation_spectra[:,id_start:id_end+1]

    # Plot data
    if plot_data:
        plt.figure()
        for i in range(Attenuation_spectra_data.shape[0]):
            plt.plot(WAVELENGTHS,Attenuation_spectra_data[i,:],label=str(SD_separations[i]))
        plt.legend(loc="best")
        plt.show()

    # Get absorption and extinction coefficient
    extinction_coefficients = (
        ExtinctionCoefficients.set_index("wavelength")
        .loc[WAVELENGTHS]
        .reset_index()
        .values
    )
    #Get chromophore names
    chrom_name = np.asarray(ExtinctionCoefficients.columns, dtype='<U18')

    #Only keep selected chromophores
    keep_id = np.array([],dtype=int)
    for i in range(chrom.shape[0]):
        keep_id = np.append(keep_id,np.where(chrom[i]==chrom_name)[0][0])
    extinction_coefficients = extinction_coefficients[:,keep_id]

    # Get MBLL Matrix
    ext_coeffs_inv = np.linalg.pinv(extinction_coefficients)

    # Get all possible combination of 2 and more detectors
    id_comb = []
    for i in range(2,SD_separations.shape[0]+1):
        id_comb += list(itertools.combinations(np.arange(0,SD_separations.shape[0]), i))

    C = []
    StO2 = []
    k_mua = []
    SD = []
    r_squared = []


    for i in range(len(id_comb)):
        #Select a combinaison
        comb = np.asarray(id_comb[i])
        #Calculate attenuation slope
        Attenuation_slope = calc_attenuation_slope(np.expand_dims(Attenuation_spectra_data[comb,:],axis=1),SD_separations[comb])

        #Compute r square to test the linearity between SD and A
        if len(id_comb)>2:
            #Linear regression
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(SD_separations, Attenuation_spectra_data[:,0])

            #calculate R-squared
            r_squared.append(r_value**2)
        else:
            r_squared.append(np.nan)
    
        #Calculate Concentration and StO2 with SRS
        _C, _StO2, _k_mua = srs_values(np.squeeze(Attenuation_slope),
                                        WAVELENGTHS,
                                        ext_coeffs_inv,
                                        SD_separations[comb].min(),
                                        SD_separations[comb].max())
        C.append(_C)
        StO2.append(_StO2)
        k_mua.append(_k_mua)

        SD_txt = ""
        for j in range(len(SD_separations[comb])):
            SD_txt=SD_txt+str(int(SD_separations[comb][j]))+" "
        SD_txt = SD_txt[0:-1] #remove last space
        SD.append(SD_txt)
    

    # Save SRS values
    C = np.asarray(C)
    StO2 = np.asarray(StO2)
    SD = np.asarray(SD)
    np.savez(path_subject+"/output/SRS",
                    C=C,
                    StO2=StO2,
                    SD_separation_in_mm = SD,
                    k_mua=k_mua,
                    r_squared = r_squared)
    print('C', C)
    print('StO2', StO2)
    print('SD',SD)
    print('k_mua',k_mua)



    C_df = pd.DataFrame(C)
    StO2_df = pd.DataFrame(StO2)
    SD_df = pd.DataFrame(SD)
    k_mua_df = pd.DataFrame(k_mua)

        # Plot k_mua against wavelengths
    plt.figure()
    plt.plot(WAVELENGTHS, k_mua_df.T)  
    plt.title("Absorption coefficient (k_mua) vs Wavelength")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption coefficient (1/cm)")
    plt.grid(True)
    plt.show()

    excel_file = '/Users/darshana/Desktop/darshana_SRS_output.xlsx'

    with pd.ExcelWriter(excel_file) as writer:
        C_df.to_excel(writer, sheet_name='sheet_1', index=False)
        StO2_df.to_excel(writer, sheet_name='sheet_2', index=False)
        SD_df.to_excel(writer, sheet_name='sheet_3', index=False)
        k_mua_df.to_excel(writer, sheet_name='sheet_4', index=False)

# Process SRS
plt.close('all')

# data path
data_path = "/Users/darshana/CYRIL_Charly_121923/PROSPEKT/"
path_subject = "/Users/darshana/CYRIL_Charly_121923/PROSPEKT/PMS0002_35+6__020823_105226"

#absortion and extinction coeff for fitting
#chrom=np.array(["HHb", "HbO2", "water"])
chrom=np.array(["HHb", "HbO2", "water", "fat"])

#Process SRS
compute_SRS(path_subject,chrom,wave_start=780,wave_end=900,plot_data=False)