from tests.grayscale import grayscaleGenerateBlurredImage
from tests.grayscale import grayscaleNPD, grayscaleNPD_kMax,\
    grayscalePNPD, grayscalePNPD_kMax,\
    grayscaleNPDIT
from tests.grayscale import grayscalePlotImages, grayscalePlotAllMethods
from tests.grayscale import grayscalePlotNPD, grayscalePlotPNPD, grayscalePlotNPDIT

from PIL import Image
import numpy as np
from mathExtras import generatePsfMatrix
import grayConfig

Compute = False
Plot = True

i = 1
grayConfig.prefix = f"grayEx{i}"

IMGPATH = "cameraman.tif"
grayConfig.noisePercent = 0.02
grayConfig.lam =  1e-3
grayConfig.lamValues = [1e-2, 1e-3]
grayConfig.lamPNPD = 1e-2
grayConfig.lamValuesPNPD = [1e-1, 1e-2, 1e-3]
grayConfig.nu = 1e-1

image = Image.open(IMGPATH)
image = np.asarray(image) / 255
image = image[::2, ::2]
grayConfig.image = image
n = image.shape[0]
print(image.shape)

# Generate PSF
psf = generatePsfMatrix(n, 1.6)
psfBT = psf.copy()
# Center PSF
psf = np.roll(psf, (-psf.shape[0] // 2, -psf.shape[0] // 2), axis=(0, 1))
grayConfig.psf = psf
grayConfig.psfBT = psfBT

# Generate blurred image
grayscaleGenerateBlurredImage.main()

if Compute:    
    # NPD
    grayscaleNPD.main()
    grayscaleNPD_kMax.main()

    # PNPD
    grayscalePNPD.main()
    grayscalePNPD_kMax.main()

    # NPDIT
    grayscaleNPDIT.main()

if Plot:
    grayscalePlotImages.main()
    grayscalePlotAllMethods.main()
    grayscalePlotNPD.main()
    grayscalePlotPNPD.main()
    grayscalePlotNPDIT.main()