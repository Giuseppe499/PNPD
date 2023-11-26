import torch
from torch.fft import fft2, ifft2

def gradLeastSquares(x, bFFT, psfFFT, psfFFTC):
    xFFT = fft2(x)
    return (ifft2(psfFFTC * (psfFFT * xFFT - bFFT))).real

def mulPInLeastSquares(mu, x, psfAbsSq):
    return (ifft2(fft2(x) / (psfAbsSq + mu))).real

def grad2D(m: torch.tensor):
    dx = torch.roll(m, -1, dims=0) - m
    dy = torch.roll(m, -1, dims=1) - m
    # Comment for periodic boundary conditions
    dx[-1, :] = 0
    dy[:, -1] = 0
    return torch.stack((dx, dy))

def div2D(dxdy: torch.tensor):
    dx = dxdy[0, :, :]
    dy = dxdy[1, :, :]
    fx = torch.roll(dx, 1, dims=0) - dx
    fy = torch.roll(dy, 1, dims=1) - dy
    fx[0, 0, :] = -dx[0, 0, :]
    fx[0, -1, :] = dx[0, -2, :]
    fy[0, :, 0] = -dy[0, :, 0]
    fy[0, :, -1] = dy[0, :, -2]
    return fx + fy

def proxhsTV(lam: float, dxdy: torch.tensor):
    dx = dxdy[0, :, :]
    dy = dxdy[1, :, :]
    factor = torch.sqrt(dx*dx + dy*dy)
    factor = torch.clamp(factor/lam, min=1)
    factor = torch.stack((factor, factor))
    return dxdy / factor