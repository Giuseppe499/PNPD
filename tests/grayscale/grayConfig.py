# Savenames Parameters
prefix = "gray"
suffix = None

# Image Parameters
noisePercent = None
image = None
psfBT = None
psf = None

# Methods Parameters
lam = None
kMax = None
nu = None
maxIt = None
momentum = None
recIndexes = []

def generateSuffix():
    suffix = f"lam{lam}_kMax{kMax}_maxIt{maxIt}"
    if nu is not None:
        suffix += f"_nu{nu}"
    if not momentum:
        suffix += f"_NM"
    
    return suffix