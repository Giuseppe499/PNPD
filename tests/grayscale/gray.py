from tests.grayscale import grayGenerateBlurredImage, grayPlotMvsNM, grayPlotRRESSIM
from tests.grayscale import grayNPD, grayPNPD, grayNPDIT
from tests.grayscale import grayPlotImages

from PIL import Image
import numpy as np
import scipy.io as sio
from mathExtras import gaussianPSF, outOfFocusPSF
import grayConfig

Compute = True
Plot = True
recIndexes = [20, 150]
grayConfig.recIndexes = recIndexes

for i in range(2,3):
    grayConfig.prefix = f"grayEx{i}"
    grayConfig.maxIt = 150
    grayConfig.kMax = 1
    grayConfig.nu = None
    kMaxx = 1
    lamPNPD = None

    if i == 1:
        Comparison = True
    else:
        Comparison = False    
    
    if i == 1:
        IMGPATH = "cameraman.tif"
        grayConfig.noisePercent = 0.01 

        lamm =  1e-4
        nuu = 1e-1

        image = Image.open(IMGPATH)
        image = np.asarray(image) / 255
        image = image[::2, ::2]
        n = image.shape[0]
        print(image.shape)

        # Generate PSF
        psf = gaussianPSF(n, 2)
    elif i == 2:
        IMGPATH = "cameraman.tif"
        grayConfig.noisePercent = 0.05 

        lamm =  3e-3
        lamPNPD = 2e-2
        nuu = 1e-1
        kMaxx = 8

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

        lamm =  1e-4
        nuu = 1e-1

        image = Image.open(IMGPATH).convert('L')
        image = np.asarray(image) / 255
        image = image[::2, ::2]
        n = image.shape[0]
        print(image.shape)

        # Generate PSF
        psf = outOfFocusPSF(n, 7.5)
    elif i ==4:
        IMGPATH = "peppers.tiff"
        grayConfig.noisePercent = 0.01

        lamm =  5e-4
        nuu = 1e-1

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

    grayConfig.kMax = kMaxx

    grayConfig.momentum = True  
    # NPD
    grayConfig.lam =  lamm
    grayConfig.suffix = grayConfig.generateSuffix()
    suffixNPD = grayConfig.suffix
    if Compute:
        grayNPD.main()

    # NPD no momentum
    grayConfig.momentum = False
    grayConfig.suffix = grayConfig.generateSuffix()
    suffixNPD_NM = grayConfig.suffix
    if Compute:
        grayNPD.main()

    grayConfig.nu = nuu
    # NPDIT
    grayConfig.momentum = True
    grayConfig.suffix = grayConfig.generateSuffix()
    suffixNPDIT = grayConfig.suffix
    if Compute:
        grayNPDIT.main()

    if lamPNPD is None:
        grayConfig.lam = lamm/grayConfig.nu
    else:
        grayConfig.lam = lamPNPD
    # PNPD
    grayConfig.momentum = True
    grayConfig.suffix = grayConfig.generateSuffix()
    suffixPNPD = grayConfig.suffix
    if Compute and not Comparison:
        grayPNPD.main()

    # PNPD no momentum
    grayConfig.momentum = False
    grayConfig.suffix = grayConfig.generateSuffix()
    suffixPNPD_NM = grayConfig.suffix
    if Compute:
        grayPNPD.main()

    if Comparison:
        # PNPD nu Comparison
        nuList = [1e0, 5*1e-1, 1e-1, 5*1e-2, 1e-2, 5*1e-3]
        lamListNu = [lamm/nu for nu in nuList]
        PNPD_nu_suffix = []
        grayConfig.momentum = True
        for i, nu in enumerate(nuList):
            grayConfig.nu = nu
            grayConfig.lam = lamListNu[i]
            grayConfig.suffix = grayConfig.generateSuffix()
            PNPD_nu_suffix.append(grayConfig.suffix)
            if Compute:
                grayPNPD.main()

        # PNPD nu Comparison no momentum
        PNPD_nu_suffix_NM = []
        grayConfig.momentum = False
        for i, nu in enumerate(nuList):
            grayConfig.nu = nu
            grayConfig.lam = lamListNu[i]
            grayConfig.suffix = grayConfig.generateSuffix()
            PNPD_nu_suffix_NM.append(grayConfig.suffix)
            if Compute:
                grayPNPD.main()

        grayConfig.momentum = True
        grayConfig.nu = nuu
        lamList = [lamm*val for val in [1e1, 1e0]]
        # PNPD kMax Comparison
        kMaxList = [10,5,2,1]
        PNPD_Kmax_suffix_lam = []
        for lamb in lamList:
            grayConfig.lam = lamb
            PNPD_Kmax_suffix = []
            for kMax in kMaxList:
                grayConfig.kMax = kMax
                grayConfig.suffix = grayConfig.generateSuffix()
                PNPD_Kmax_suffix.append(grayConfig.suffix)
                if Compute:
                    grayPNPD.main()
            PNPD_Kmax_suffix_lam.append(PNPD_Kmax_suffix)

        #NPDIT kMax Comparison
        NPDIT_Kmax_suffix_lam = []
        for lamb in lamList:
            grayConfig.lam = lamb
            NPDIT_Kmax_suffix = []
            for kMax in kMaxList:
                grayConfig.kMax = kMax
                grayConfig.suffix = grayConfig.generateSuffix()
                NPDIT_Kmax_suffix.append(grayConfig.suffix)
                if Compute:
                    grayNPDIT.main()
            NPDIT_Kmax_suffix_lam.append(NPDIT_Kmax_suffix)      

        # PNPD low nu no momentum vs high nu and momentum vs high nu, momentum and high kMax
        nuListComp = [1e-1, 1e-2, 1e-2, 1e-2, 1e-2]
        momentumListComp = [True, True, True, True, False]
        kMaxListComp = [1, 1, 2, 5, 1]
        lamListComp = [lamm/nu for nu in nuListComp]
        PNPD_comp_suffix = []
        for i in range(len(nuListComp)):
            grayConfig.nu = nuListComp[i]
            grayConfig.momentum = momentumListComp[i]
            grayConfig.kMax = kMaxListComp[i]
            grayConfig.lam = lamListComp[i]
            grayConfig.suffix = grayConfig.generateSuffix()
            PNPD_comp_suffix.append(grayConfig.suffix)
            if Compute:
                grayPNPD.main()

    if Plot:
        # Plot images
        grayPlotImages.main(suffixNPD, suffixPNPD, suffixNPDIT, recIndexes)
        # Momentum vs No Momentum
        grayPlotMvsNM.main(suffixNPD, suffixNPD_NM, "NPD")
        grayPlotMvsNM.main(suffixPNPD, suffixPNPD_NM, "PNPD")

        # NPD vs PNPD vs NPDIT
        filenameList = ["NPD_"+suffixNPD, "PNPD_"+suffixPNPD, "NPDIT_"+suffixNPDIT]
        nameList = ["NPD", "PNPD", "NPDIT"]
        grayPlotRRESSIM.main(filenameList, nameList)

        if Comparison:
            # PNPD nu Comparison
            filenameList = [f"PNPD_{suffix}" for suffix in PNPD_nu_suffix]
            nameList = [f"PNPD $\\nu= {nu}$, $\lambda= {lamb}$" for lamb, nu in zip(lamListNu, nuList)]
            title = f"PNPD $\\nu= {nuList}$"
            saveStr = f"PNPD_nu"
            grayPlotRRESSIM.main(filenameList, nameList, saveStr=saveStr, title=title, showStop=False)

            # PNPD nu Comparison no momentum
            filenameList = [f"PNPD_{suffix}" for suffix in PNPD_nu_suffix_NM]
            nameList = [f"PNPD without momentum $\\nu= {nu}$, $\lambda= {lamb}$" for lamb, nu in zip(lamListNu, nuList)]
            title = f"PNPD without momentum $\\nu= {nuList}$"
            saveStr = f"PNPD_nu_NM"
            grayPlotRRESSIM.main(filenameList, nameList, saveStr=saveStr, title=title, showStop=False)

            # PNPD kMax Comparison
            for i in range(len(lamList)):
                filenameList = [f"PNPD_{suffix}" for suffix in PNPD_Kmax_suffix_lam[i]]
                nameList = [f"PNPD $k_{{max}}= {kMax}$" for kMax in kMaxList]
                title = f"PNPD $\lambda = {lamList[i]}$, $k_{{max}}= {kMaxList}$"
                saveStr = f"PNPD_K_l{lamList[i]}"
                grayPlotRRESSIM.main(filenameList, nameList, saveStr=saveStr, title=title, showStop=False)

            # NPDIT kMax Comparison
            for i in range(len(lamList)):
                filenameList = [f"NPDIT_{suffix}" for suffix in NPDIT_Kmax_suffix_lam[i]]
                nameList = [f"NPDIT $k_{{max}}= {kMax}$" for kMax in kMaxList]
                title = f"NPDIT $\lambda = {lamList[i]}$, $k_{{max}}= {kMaxList}$"
                saveStr = f"NPDIT_K_l{lamList[i]}"
                grayPlotRRESSIM.main(filenameList, nameList, saveStr=saveStr, title=title, showStop=False)

            # PNPD low nu no momentum vs high nu and momentum
            filenameList = [f"PNPD_{suffix}" for suffix in PNPD_comp_suffix]
            nameList = [f"PNPD" + (" without momentum" if not momentum else "") + f" $\\nu= {nu}$, $\lambda= {lamb}$, $k_{{max}}= {kMax}$" for lamb, nu, momentum, kMax in zip(lamListComp, nuListComp, momentumListComp, kMaxListComp)]
            title = f"PNPD momentum, $\\nu$ and $k_{{max}}$ relation"
            saveStr = f"PNPD_comp"
            grayPlotRRESSIM.main(filenameList, nameList, saveStr=saveStr, title=title, showStop=False)
