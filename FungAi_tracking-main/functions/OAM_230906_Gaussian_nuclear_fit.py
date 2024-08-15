#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:43:51 2024

@author: samarth
"""

import numpy as np
from scipy.ndimage import convolve
from skimage.measure import regionprops
from scipy.stats import multivariate_normal

def OAM_230906_Gaussian_nuclear_fit(IG, peak_cutoff, x_size, y_size, ccell):
    p_nuc = np.zeros_like(IG)  # allocate

    # Gaussian fit
    h = np.ones((5, 5)) / 25  # averaging filter kernel
    Ignew = convolve(IG, h, mode='nearest')  # apply the filter using convolve

    nuc_tmp = Ignew * ccell
    Amp = np.max(nuc_tmp)
    A, B = np.where(nuc_tmp == Amp)

    if A.size != 1:  # in case we have more than 1 max intensity pixel
        A = A[1]
        B = B[1]

    if A >= (nuc_tmp.shape[0] - 5) or B >= (nuc_tmp.shape[1] - 5):  # exclusion of edge cells
        p_nuc = np.nan
    else:
        xcenter = B
        ycenter = A
        wind = 5
        xcoord = np.arange(max(0, xcenter - wind), min(y_size, xcenter + wind + 1))
        ycoord = np.arange(max(0, ycenter - wind), min(x_size, ycenter + wind + 1))
        X, Y = np.meshgrid(xcoord, ycoord)

        sX = X.shape
        sY = Y.shape
        ok_size = wind * 2 + 1
        if (sX[0] == ok_size and sX[1] == ok_size and sY[0] == ok_size and sY[1] == ok_size):
            In_part = nuc_tmp[np.round(Y).astype(int), np.round(X).astype(int)]
            edge_med = np.median(np.concatenate([In_part[0, :], In_part[-1, :], In_part[:, 0], In_part[:, -1]]))
            x0 = xcenter
            y0 = ycenter
            mu = [x0, y0]
            best_fit_score = 1e9  # first threshold
            best_fit_no = [0, 0]
            for jj in range(1, 26):
                for jj2 in range(1, 26):
                    limit_h = int(np.round(np.sqrt(jj * jj2)))
                    for ii in range(-(limit_h - 1), limit_h, 2):  # why 2
                        Sigma = np.array([[jj, -ii], [-ii, jj2]])
                        F = multivariate_normal.pdf(np.column_stack([X.ravel(), Y.ravel()]), mean=mu, cov=Sigma)
                        F = F.reshape(len(X[:, 0]), len(Y[0, :]))
                        Fmax = np.max(F)
                        Z = (((Amp - edge_med) / Fmax) * F + edge_med)
                        Z = Z * (In_part > 0)  # express the results in Z
                        tmp_score = np.sum(np.abs(Z - In_part))
                        if tmp_score < best_fit_score:
                            best_fit_no = Sigma
                            best_fit_score = tmp_score

            # Get the best solution
            F = multivariate_normal.pdf(np.column_stack([X.ravel(), Y.ravel()]), mean=mu, cov=best_fit_no)
            F = F.reshape(len(X[:, 0]), len(Y[0, :]))
            Fmax = np.max(F)
            Z = (((Amp - edge_med) / Fmax) * F + edge_med)
            Z = Z * (In_part > 0)
            p_nuc[np.round(Y).astype(int), np.round(X).astype(int)] += (Z > (Amp * peak_cutoff))

    return p_nuc.astype(float)

