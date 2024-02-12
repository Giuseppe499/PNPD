from tests.grayscale import grayGenerateBlurredImage
from tests.grayscale import grayNPD, grayNPD_kMax,\
    grayPNPD, grayPNPD_kMax,\
    grayNPDIT
from tests.grayscale import grayPlotImages, grayPlotAllMethods
from tests.grayscale import grayPlotNPD, grayPlotPNPD, grayPlotNPDIT

from PIL import Image
import numpy as np
import scipy.io as sio
from mathExtras import gaussianPSF, outOfFocusPSF
import grayConfig

Compute = True
Plot = True

for i in range(1,5):
    grayConfig.prefix = f"grayEx{i}"

    grayConfig.lam =  1e-3
    grayConfig.lamValues = [1e-2, 1e-3]
    grayConfig.lamPNPD = 1e-2
    grayConfig.lamValuesPNPD = [1e-1, 1e-2, 1e-3] 
    grayConfig.nu = 1e-1
    
    if i == 1:
        IMGPATH = "cameraman.tif"
        grayConfig.noisePercent = 0.01 

        grayConfig.lam =  1e-4
        grayConfig.lamValues = [1e-3, 1e-4]
        grayConfig.lamPNPD = 1e-3
        grayConfig.lamValuesPNPD = [1e-2, 1e-3, 1e-4]     

        image = Image.open(IMGPATH)
        image = np.asarray(image) / 255
        image = image[::2, ::2]
        n = image.shape[0]
        print(image.shape)

        # Generate PSF
        psf = gaussianPSF(n, 2)
    elif i == 2:
        IMGPATH = "cameraman.tif"
        grayConfig.noisePercent = 0.02 

        grayConfig.lam =  1e-3
        grayConfig.lamValues = [1e-2, 1e-3]
        grayConfig.lamPNPD = 1e-2
        grayConfig.lamValuesPNPD = [1e-1, 1e-2, 1e-3]     

        image = Image.open(IMGPATH)
        image = np.asarray(image) / 255
        image = image[::2, ::2]
        n = image.shape[0]
        print(image.shape)

        # Generate PSF
        psf = gaussianPSF(n, 2)
    elif i == 3:
        IMGPATH = "peppers.tiff"
        grayConfig.noisePercent = 0.01

        grayConfig.lam =  1e-4
        grayConfig.lamPNPD = 1e-3

        image = Image.open(IMGPATH).convert('L')
        image = np.asarray(image) / 255
        image = image[::2, ::2]
        n = image.shape[0]
        print(image.shape)

        # Generate PSF
        psf = outOfFocusPSF(n, 8)
    elif i ==4:
        IMGPATH = "peppers.tiff"
        grayConfig.noisePercent = 0.01

        grayConfig.lam =  5e-4
        grayConfig.lamPNPD = 5e-3

        image = Image.open(IMGPATH).convert('L')
        image = np.asarray(image) / 255
        image = image[::2, ::2]
        n = image.shape[0]
        print(image.shape)

        # Generate PSF
        psf = sio.loadmat('kernel.mat')['motion8']
        m = psf.shape[0]
        psf = np.pad(psf, ((n-m)//2 + 1, (n-m)//2) , 'constant')


    
    # Center PSF
    psfBT = psf.copy()
    psf = np.roll(psf, (-psf.shape[0] // 2, -psf.shape[0] // 2), axis=(0, 1))

    grayConfig.image = image
    grayConfig.psf = psf
    grayConfig.psfBT = psfBT

    # Generate blurred image
    grayGenerateBlurredImage.main()

    if Compute:    
        # NPD
        grayNPD.main()
        grayNPD_kMax.main()

        # PNPD
        grayPNPD.main()
        grayPNPD_kMax.main()

        # NPDIT
        grayNPDIT.main()

    if Plot:
        grayPlotImages.main()
        grayPlotAllMethods.main()
        grayPlotNPD.main()
        grayPlotPNPD.main()
        grayPlotNPDIT.main()