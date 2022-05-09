import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from skimage import exposure

os.environ['KMP_DUPLICATE_LIB_OK']='True'

mask_palette = {1 : (0,153,0),
                2 : (0,153,0),
                3 : (0,153,0),
              0 : (255,255,255)} 

def convert_to_binary(arr_2d, palette= mask_palette):
    """ Numeric labels to RGB-color encoding.
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def get_channels(im, all_channels, select_channels):
    """
    Filters the channels for a given set of channels. Used to extract RGB channels from full-spectral S2 image.
    """

    all_ch = all_channels
    final_ch = select_channels

    chns = [all_ch.index(final_ch) if isinstance(final_ch, str) else final_ch for final_ch in final_ch]
    
    return im[chns]

def transpose(im):
    """
    Transpose operation. Used to straighten the rasterio image.
    """
    
    channels = im.shape[0]
    
    if channels == 1:
        im = im[0]  
    else:
        im = im.transpose(1, 2, 0)  
    return im

def norm_(data):
    """
    Normalization. ( Data - min_value ) / (max_value - min_value). 
    """

    min_, max_ = data.min(), data.max()
    n_data = (data - min_) / (max_ - min_)
    return n_data
        
def stretch(im):
    """
    Strectch using 1st and 99th percentile. 
    """

    p1, p99 = np.percentile(im, (1, 99))
    J = exposure.rescale_intensity(im, in_range=(p1, p99))
    J = J/J.max()
    return J

def visualize(export = False , WDPA_ID = None, **images):
    """PLot images in one row.
    """

    n = len(images)
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(32, 20), num=1, clear=True)
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        
        string = '--' # --WDPA_ID:'
        plt.title(' '.join(name.upper()) + string.upper() + '{}'.format(WDPA_ID).title())
        plt.imshow(image)
        
        if export is not False:
            plt.savefig(r'C:\Users\burak\Desktop\QA_Check_MapInWild\addition_100' + '\{}.png'.format(WDPA_ID))   
    if export is False:
        plt.show()
    
WC_palette = {10 : (0,160,0),    # "Tree cover" 00a000
              20 : (150,100,0), #"Shrubland" 966400
              30 : (255, 180, 0), #"Grassland" ffb400
              40 : (255,255,100), #"Cropland" ffff64
              50 : (195,20,0), #"Built-up" c31400
              60 : (255, 245, 215),  #"Bare / sparse vegetation" fff5d7
              70 : (255, 255, 255), #"Snow and ice" ffffff
              80 : (0, 70, 200), #"Permanent water bodies" 0046c8
              90 : (0, 220, 130), #"Herbaceous wetland" 00dc82
              95 : (0, 150, 120), # "Mangroves" 009678
              100 : (255, 235, 175)} #"Moss and lichen" ffebaf

cmappx = ["#00a000", "#966400", "#ffb400", "#ffff64","c31400","#fff5d7","#ffffff","#0046c8","#00dc82", "#009678", "#ffebaf"]
rgb = [(0, 160, 0),(150, 100, 0), (255, 180, 0),  (255, 255, 100),(195, 20, 0),(255, 245, 215),(255, 255, 255),(0, 70, 200),(0, 220, 130),(0, 150, 120),(255, 235, 175)]

def convert_to_color(arr_2d, palette= WC_palette):
    """ Numeric labels to RGB-color encoding.
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


class MapInWild_S2(): 
    """Data Module for Sentinel subset of MapInWild.
    """

    def __init__(
            self,
            root: str = "data"):
        
        self.root = root
        
        mods = os.listdir(root)
        self.mod_paths = [os.path.join(root, mod) for mod in mods]
        self.im_filenames = os.listdir(self.mod_paths[0])

    @staticmethod
    def _read_img(image_path):
        with rasterio.open(image_path) as data:
            bands = data.descriptions
            image = data.read()
        return image, bands
        
    def __len__(self):
        return len(self.im_filenames)
    
    def __getitem__(self, idx):   
        
        im_name = self.im_filenames[idx]
        s2_autumn_indexer = os.path.join(self.mod_paths[4], im_name)
        s2_spring_indexer = os.path.join(self.mod_paths[2], im_name)
        s2_summer_indexer = os.path.join(self.mod_paths[3], im_name)
        s2_winter_indexer = os.path.join(self.mod_paths[1], im_name)

        s2_autumn = self._read_img(s2_autumn_indexer)
        s2_spring = self._read_img(s2_spring_indexer)
        s2_summer = self._read_img(s2_summer_indexer)
        s2_winter = self._read_img(s2_winter_indexer)

        WDPA_ID = im_name.split('.')[0] 
        
        return s2_autumn, s2_spring, s2_summer, s2_winter, WDPA_ID


class MapInWild_AUX(): #dataset
    """
    Data Module for viz MapInWild.
    """

    def __init__(
            self,
            root: str = "data",
            plot = False):
        
        self.root = root
        
        mods = os.listdir(root)
        self.mod_paths = [os.path.join(root, mod) for mod in mods]
        self.im_filenames = os.listdir(self.mod_paths[0])

    @staticmethod
    def _read_img(image_path):
        with rasterio.open(image_path) as data:
            bands = data.descriptions
            image = data.read()
        return image
        
    def __len__(self):
        return len(self.im_filenames)
    
    def __getitem__(self, idx):   
        im_name = self.im_filenames[idx]
        
        s1_indexer = os.path.join(self.mod_paths[1], im_name) 
        WC_indexer = os.path.join(self.mod_paths[2], im_name)
        NIIR_indexer = os.path.join(self.mod_paths[3], im_name)
        mask_indexer = os.path.join(self.mod_paths[4], im_name)
       
        s1_ = self._read_img(s1_indexer)
        WC_ = self._read_img(WC_indexer)
        NIIR_ = self._read_img(NIIR_indexer)
        mask_ = self._read_img(mask_indexer)
        
        WDPA_ID = im_name.split('.')[0] 
        
        return s1_, WC_, NIIR_, mask_, WDPA_ID
    
    

def viz_single_id(root, im_name):
    """
    Outputs a 8-pair of WDPA patches. 
    """
    
    def _read_img(image_path):
        with rasterio.open(image_path) as data:
            bands = data.descriptions
            image = data.read()
        return image, bands

    def visualize_(export = False , WDPA_ID = None, **images):
        """PLot images in one row."""
        n = len(images)
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(32,32), num=1, clear=True)
        
        gs1 = gridspec.GridSpec(4, 4)
        gs1.update(wspace=0.02, hspace=0.02) # set the spacing between axes. 

        for i, (name, image) in enumerate(images.items()):
            plt.subplot(gs1[i])
            plt.xticks([])
            plt.yticks([])
 
            string = '--WDPA_ID:'
            plt.imshow(image)
            
            if export is not False:
                plt.savefig(r'C:\Users\burak\Desktop\Neuer Ordner' + '\{}'.format(WDPA_ID))   
        if export is False:
            plt.show()
    
    mods = os.listdir(root)
    mod_paths = [os.path.join(root, mod) for mod in mods]

    im_name = str(im_name) + '.tif'
    s2_autumn_indexer = os.path.join(mod_paths[3], im_name)
    s2_spring_indexer = os.path.join(mod_paths[4], im_name)
    s2_summer_indexer = os.path.join(mod_paths[5], im_name)
    s2_winter_indexer = os.path.join(mod_paths[6], im_name)

    s1_indexer = os.path.join(mod_paths[2], im_name) 
    WC_indexer = os.path.join(mod_paths[0], im_name)
    VIIRS_indexer = os.path.join(mod_paths[7], im_name)
    mask_indexer = os.path.join(mod_paths[1], im_name)

    s2_autumn = _read_img(s2_autumn_indexer)
    s2_spring = _read_img(s2_spring_indexer)
    s2_summer = _read_img(s2_summer_indexer)
    s2_winter = _read_img(s2_winter_indexer)

    s1_ = _read_img(s1_indexer)
    WC_ = _read_img(WC_indexer)
    VIIRS_ = _read_img(VIIRS_indexer)
    mask_ = _read_img(mask_indexer)
    s1 = transpose(s1_[0])
    s1 = stretch(s1)
    
    VIIRS = transpose(VIIRS_[0])
    mask = transpose(mask_[0])
    mask_binary = convert_to_binary(mask)

    WC_color = convert_to_color(WC_[0].squeeze())
    S2_auutmn_rgb = get_channels(im = s2_autumn[0], all_channels = s2_autumn[1], select_channels = ['B4', 'B3', 'B2'])
    S2_autumn_rgb_tr = transpose(S2_auutmn_rgb)
    S2_autumn_rgb_tr_st = stretch(S2_autumn_rgb_tr)
    
    S2_spring_rgb = get_channels(im = s2_spring[0], all_channels = s2_spring[1], select_channels = ['B4', 'B3', 'B2'])
    S2_spring_rgb_tr = transpose(S2_spring_rgb)
    S2_spring_rgb_tr_st = stretch(S2_spring_rgb_tr)

    S2_summer_rgb = get_channels(im = s2_summer[0], all_channels = s2_summer[1], select_channels = ['B4', 'B3', 'B2'])
    S2_summer_rgb_tr = transpose(S2_summer_rgb)
    S2_summer_rgb_tr_st = stretch(S2_summer_rgb_tr)

    S2_winter_rgb = get_channels(im = s2_winter[0], all_channels = s2_winter[1], select_channels = ['B4', 'B3', 'B2'])
    S2_winter_rgb_tr = transpose(S2_winter_rgb)
    S2_winter_rgb_tr_st = stretch(S2_winter_rgb_tr)

    visualize_(WDPA_ID = im_name,
              Spring = S2_spring_rgb_tr_st,
              Summer = S2_summer_rgb_tr_st,
              Autumn = S2_autumn_rgb_tr_st,
              Winter = S2_winter_rgb_tr_st,
              s1 = s1[:,:,0],
              WC = WC_color,
              NIIR = VIIRS,
              mask = mask_binary,
              export = False)
