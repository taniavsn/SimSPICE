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
        self.common_wvl = np.array([700.3833769, 700.5784109, 700.7734449, 700.9684789, 701.1635129, 
                                    701.3585469, 701.5535809, 701.7486149, 701.9436489, 702.1386829, 702.3337169, 
                                    702.5287509, 702.7237849, 702.9188189, 703.1138529, 703.3088869, 703.5039209, 
                                    703.6989549, 703.8939889, 704.0890229, 704.2840569, 704.4790909, 704.6741249, 
                                    704.8691589, 705.0641929, 705.2592269, 705.4542609, 705.6492949, 705.8443289, 
                                    706.0393629, 706.2343969, 706.4294309, 706.6244649, 706.8194989, 707.0145329, 
                                    707.2095669, 707.4046009, 707.5996349, 707.7946689, 707.9897029, 708.1847369, 
                                    783.4678609, 783.6628949, 783.8579289, 784.0529629, 784.2479969, 784.4430309, 
                                    784.6380649, 784.8330989, 785.0281329, 785.2231669, 785.4182009, 785.6132349, 
                                    785.8082689, 786.0033029, 786.1983369, 786.3933709, 786.5884049, 786.7834389, 
                                    786.9784729, 787.1735069, 787.3685409, 787.5635749, 787.7586089, 787.9536429, 
                                    788.1486769, 788.3437109, 788.5387449, 788.7337789, 788.9288129, 789.1238469, 
                                    789.3188809, 789.5139149, 789.7089489, 789.9039829, 790.0990169, 790.2940509, 
                                    790.4890849, 790.6841189, 790.8791529, 791.0741869, 766.5974199, 766.7924539, 
                                    766.9874879, 767.1825219, 767.3775559, 767.5725899, 767.7676239, 767.9626579, 
                                    768.1576919, 768.3527259, 768.5477599, 768.7427939, 768.9378279, 769.1328619, 
                                    769.3278959, 769.5229299, 769.7179639, 769.9129979, 770.1080319, 770.3030659, 
                                    770.4980999, 770.6931339, 770.8881679, 771.0832019, 771.2782359, 771.4732699, 
                                    771.6683039, 771.8633379, 772.0583719, 772.2534059, 772.4484399, 772.6434739, 
                                    772.8385079, 773.0335419, 773.2285759, 773.4236099, 773.6186439, 773.8136779, 
                                    774.0087119, 774.2037459]
                            )
    
    def get_exposure(self, filename):
        '''Open the file and get the data'''
        return read_spice_l2_fits(filename, memmap=False)

    def get_keys(self):
        '''Prints the different windows of the Sprout file'''
        keys = list(self.exposure.keys())
        return [keys[i] for i in [0,3,4]]
    
    def get_nbr_keys(self):
        '''Get the number of windows in the file'''
        return len([self.get_keys()])
    
    def get_rasters(self):
        '''Returns a list of the rasters present in the file'''
        rasters = [self.exposure['O III 703 / Mg IX 706 (Merged)'], self.exposure['Ne VIII 770 (Merged)'], self.exposure['S V 786 / O IV 787 (Merged)']]
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
        for key in self.get_keys():
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
    