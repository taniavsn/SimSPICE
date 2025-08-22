from sunraster.instr.spice import read_spice_l2_fits # type: ignore
import os
import numpy as np
from astropy.io import fits
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings 
warnings.filterwarnings("ignore")


class Sprout_ML:
    ''' 
    The SPROUTS data consist of a file with 8 windows: 'O III 703 / Mg IX 706 (Merged)', 'O II 718 - Peak (spectral bin 2)', 'S IV 750/ Mg IX (spectral bin 2)', 'Ne VIII 770 (Merged)', 
    'S V 786 / O IV 787 (Merged)', 'Ly-gamma-CIII group (Merged)', 'N III 991 (Merged)' and 'O VI 1032 (Merged)'

    Each of those windows is a 'raster', i.e. a 3D data cube with 2 spatial dimensions and 1 spectral dimension (we disregard the time dimension). 
    In other words, each raster is a 2D image where each pixel is a spectrum.
    '''
    #min and max peak half width (sigma)
    MIN_WIDTH = 0.2 #Angstrom
    MAX_WIDTH = 0.8 #Angstrom # prev:2
    LINES = ['O III 702', 'O III 703', 'Mg IX 706', 'O II 718', 'Mg IX 749', 'S IV 750','Ne VIII 770', 'Mg VIII 772', 
        'S V 786', 'O IV 787', 'O IV 790', 'Ly-gamma 972', 'C III 977', 'O I / Na VI', 'N III 989', 'N III 991', 'O VI 1032']

    SIZE_CROPPED_MAP = 116160
    SHAPE_CROPPED_MAP = (605,192)


    def __init__(self, dir, filename, padding_size=10):
        self.filename = os.path.join(dir,filename)
        self.fitsfile = filename
        self.exposure = self.get_exposure(dir+filename)
        self.padding_size = padding_size        
        self.common_wvl = np.array([766.6 , 766.79,  766.98,  767.17,  
                                    767.36,  767.55,  767.74,  767.93,
        768.12,  768.31,  768.5 ,  768.69,  768.88,  769.07,  769.26,
        769.45,  769.64,  769.83,  770.02,  770.21,  770.4 ,  770.59,
        770.78,  770.97,  771.16,  771.35,  771.54,  771.73,  771.92,
        772.11,  772.3 ,  772.49,  772.68,  772.87,  773.06,  773.25,
        773.44,  773.63,  773.82,  774.01])
    
    def get_exposure(self, filename):
        '''Open the file and get the data'''
        return read_spice_l2_fits(filename, memmap=False)

    def get_keys(self):
        '''Prints the different windows of the Sprout file'''
        return list(self.exposure.keys())[3]
    
    def get_nbr_keys(self):
        '''Get the number of windows in the file'''
        return len([self.get_keys()])
    
    def get_rasters(self):
        '''Returns a list of the rasters present in the file'''
        rasters = [self.exposure['Ne VIII 770 (Merged)']]
        return rasters
    
    def get_wcs(self, bindown=False):
        ''' returns a list of wcs corresponding to the rasters present in the file'''
        rasters = self.get_rasters()
        wcss = [copy.deepcopy(r.wcs) for r in rasters]  # Make a deep copy to avoid modifying original wcs
        if bindown:
            for x in range(len(wcss)):
                wcss[x].wcs.crpix[1] = (wcss[x].wcs.crpix[1]/2)
                wcss[x].wcs.cdelt[1] = (wcss[x].wcs.cdelt[1]*2)
        return wcss
    
    def get_cubes(self, croplatbottom=725, croplattop=120):
        '''Returns a list of the actual numerical data of the rasters present in the file in W/m**2/nm/str'''
        cubes = []
        for key in [self.get_keys()]:
            cubes.append(np.nan_to_num(self.exposure[key][0,:,croplattop:croplatbottom,:].data))  # indexation starts at the top
        return cubes
    
    def get_cubes_uncropped(self):
        '''Returns a list of the actual numerical data of the rasters present in the file'''
        cubes = []
        for key in [self.get_keys()]:
            cubes.append(np.nan_to_num(self.exposure[key][0,:,:,:].data))
        return cubes
    
    def whatsgoingon(self, key='Ne VIII 770 (Merged)', slice=20, croplattop=None, croplatbottom=None):
        
        cube = self.exposure[key][0,:,croplattop:croplatbottom,:].data
        wcs = self.get_wcs()[6]
        ax = plt.subplot(projection=wcs, slices=('x', 'y', 20, 1))
        im = ax.imshow((cube[slice, :, :])**(1/2.2), aspect=1/4, cmap='gist_heat', vmin=0)
        plt.colorbar(im, ax=ax)
        
        # Remove axis labels and ticks for the "Time" axis
        ax.coords[3].set_ticks_visible(False)  # Hide ticks on the "time" axis
        ax.coords[3].set_ticklabel_visible(False)  # Hide tick labels on the "time" axis
        ax.coords[3].set_axislabel(' ')  # Remove the "time" axis label
        
        # Optionally, remove x and y axis labels if needed
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')

        datetime_str = (self.fitsfile).split('_')[3]
        date = datetime_str[:8]  
        time = datetime_str[9:] 
        plt.title(f"{date[:4]}-{date[4:6]}-{date[6:]} T {time[:2]}:{time[2:4]}:{time[4:]}")
        #plt.show()

    def get_file_info(self):
        '''Get info from the header of the file'''
        hdul = fits.open(self.filename)
        print(hdul.info(),'\n----------------------------------------')
        rast = self.get_rasters()[0]
        print(rast,'\n----------------------------------------')
        print(rast.wcs)
        hdul.close()

    def get_raster_dimensions(self):
        print(self.filename, ' : ', self.exposure[[self.get_keys()][0]].data.shape)
    
    def get_file_time(self):
        print(self.get_rasters()[0])

    def get_wvl_arrays_A(self):
        wavelengths_arrays = []
        rasters = self.get_rasters()
        for raster in rasters:
            wavelengths_arrays.append(raster.spectral_axis.value*(10**10)) # turn to Angstrom
        return wavelengths_arrays

    def pad_wvl_arrays(self):
        ''' 
        Resolution of each spectral window : 0.19 A
        Pad each wavelength array by 20 values left and right, i.e. by about 3.8 Angstrom

        Returns a list (usually len 8) of arrays containing the padded wvl arrays
        '''
        wvlarrays_padded = []
        wavelength_arrays = self.get_wvl_arrays_A()
        for array in wavelength_arrays:
            step_size = array[11]-array[10]
            left_pad_wvl = array[0] - step_size * np.arange(self.padding_size, 0, -1) 
            right_pad_wvl = array[-1] + step_size * np.arange(1, self.padding_size+1) 
            padded_wavelengths = np.concatenate([left_pad_wvl, array, right_pad_wvl])
            wvlarrays_padded.append(padded_wavelengths)
            
        return wvlarrays_padded

    def pad_flux_array(self, method='zeros'):
        ''' 
        flux_arrays = list
                list of datacubes containing the spectra
        method: str or None
                Way to fill the flux values. 
                If median, the median value of the spectrum is used, NaNs if None.
        size = int
                number of added data points on each side of the spectrum

        Returns flux_arrays_padded, masks where:
            - flux_arrays_padded is a list of all cubes with each spectrum padded.
            - masks is a list of arrays with the same shape as flux_arrays_padded, 
            containing 1 for original values and 0 for padding.
        '''
        flux_arrays_padded = []
        masks = []
        flux_arrays = self.get_cubes()
        
        for flux in flux_arrays:  # flux shape should be (x, 610, 192)
            padded_spectra = np.zeros((flux.shape[0] + 2 * self.padding_size, flux.shape[1], flux.shape[2]))
            mask = np.zeros_like(padded_spectra, dtype=int)  # Mask array with 0 for padding
            
            for i in range(flux.shape[1]):
                for j in range(flux.shape[2]):
                    # Create padding based on the method
                    tab = np.linspace(0, 1, self.padding_size)
                    if method == 'median':
                        pad_flux = np.full_like(tab, np.nanmedian(flux[:, i, j]))
                    elif method == 'zeros':
                        pad_flux = np.zeros_like(tab)
                    else:
                        pad_flux = np.full_like(tab, np.nan)
                    
                    # Concatenate the padded values with the actual spectrum
                    padded_spectrum = np.concatenate([pad_flux, flux[:, i, j], pad_flux])
                    padded_spectra[:, i, j] = padded_spectrum
                    
                    # Set the mask to 1 for the actual data range
                    mask[self.padding_size:self.padding_size + flux.shape[0], i, j] = 1

            flux_arrays_padded.append(padded_spectra)
            masks.append(mask)
        
        return flux_arrays_padded, masks


def interpolate_arrays(common_wvl, wvl_old, flux_array, mask_array=None):
    '''
    Returns a new array of interpolated values of flux on a common wavelength array.
    '''
    if mask_array is not None:
        f = interp1d(wvl_old, mask_array)
        mask_new = f(common_wvl)
        f = interp1d(wvl_old, flux_array)
        flux_new = f(common_wvl)
        return flux_new, mask_new
    else:    
        f = interp1d(wvl_old, flux_array)
        flux_new = f(common_wvl)
        return flux_new
    