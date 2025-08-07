
import numpy as np
import pandas as pd

import math
import scipy as sp
import scipy.signal as sig
from scipy.integrate import simps
from scipy.stats import entropy as scipyEntropy
from scipy.signal import butter, lfilter, welch
from scipy.spatial.distance import pdist, squareform
from scipy import fft, fftpack

from antropy import *
import pyeeg
from mne.time_frequency import psd_array_multitaper

import collections

from numba import jit, njit, prange 



##########################################
#### Fuzzy entropy calculation
##########################################
# @njit(parallel=True)
# def compute_fuzzy_entropy_jit(data, m=2, tau=1, r_factor=0.2):
#     N = len(data)
#     r = r_factor * np.std(data)
#
#     n_vectors = N - (m + 1) * tau + 1
#     if n_vectors <= 1:
#         return 0.0
#
#     ym = np.empty((n_vectors, m))
#     ya = np.empty((n_vectors, m + 1))
#
#     for i in range(n_vectors):
#         for j in range(m):
#             ym[i, j] = data[i + j * tau]
#         for j in range(m + 1):
#             ya[i, j] = data[i + j * tau]
#
#     count_m = 0.0
#     count_m1 = 0.0
#
#     for i in range(n_vectors):
#         for j in range(i + 1, n_vectors):
#             dist_m = 0.0
#             dist_m1 = 0.0
#
#             for k in range(m):
#                 diff = abs(ym[i, k] - ym[j, k])
#                 if diff > dist_m:
#                     dist_m = diff
#             count_m += np.exp(-np.log(2) * (dist_m / r) ** 2)
#
#             for k in range(m + 1):
#                 diff1 = abs(ya[i, k] - ya[j, k])
#                 if diff1 > dist_m1:
#                     dist_m1 = diff1
#             count_m1 += np.exp(-np.log(2) * (dist_m1 / r) ** 2)
#
#     cm = (2 * count_m) / (n_vectors * (n_vectors - 1))
#     ca = (2 * count_m1) / (n_vectors * (n_vectors - 1))
#
#     if cm == 0 or ca == 0:
#         return 0.0
#     return -np.log(ca / cm)


# @njit(parallel=True)
# def compute_fuzzy_entropy_jit(signal, m=2, tau=1, r_factor=0.2):
#     N = len(signal)
#     if N - (m + 1) * tau + 1 <= 0:
#         return 0.0  # Not enough data
#
#     r = r_factor * np.std(signal)
#
#     count_m = 0.0
#     count_m1 = 0.0
#     num_vectors = N - (m + 1) * tau + 1
#
#     for i in range(num_vectors - 1):
#         for j in range(i + 1, num_vectors):
#             max_diff_m = 0.0
#             max_diff_m1 = 0.0
#             for k in range(m):
#                 diff = np.abs(signal[i + k * tau] - signal[j + k * tau])
#                 if diff > max_diff_m:
#                     max_diff_m = diff
#
#             for k in range(m + 1):
#                 diff = np.abs(signal[i + k * tau] - signal[j + k * tau])
#                 if diff > max_diff_m1:
#                     max_diff_m1 = diff
#
#             count_m += np.exp(-np.log(2.0) * (max_diff_m / r) ** 2)
#             count_m1 += np.exp(-np.log(2.0) * (max_diff_m1 / r) ** 2)
#
#     cm = (2.0 * count_m) / (num_vectors * (num_vectors - 1))
#     ca = (2.0 * count_m1) / (num_vectors * (num_vectors - 1))
#
#     if ca == 0.0 or cm == 0.0:
#         return 0.0
#
#     return -np.log(ca / cm)


# @njit(parallel=True)
@njit
def fuzzy_similarity_fuzzEn_(vec1, vec2, r, log2_val):
    max_diff = 0.0
    for i in range(len(vec1)):
        diff = abs(vec1[i] - vec2[i])
        if diff > max_diff:
            max_diff = diff
    return np.exp(-log2_val * (max_diff / r) ** 2)


@njit(parallel=True)
def embed_signal_fuzzEn_(signal, m, tau):
    N = len(signal)
    num_vectors = N - (m - 1) * tau
    emb = np.empty((num_vectors, m))
    # for i in range(num_vectors):
    for i in prange(num_vectors):
        for j in range(m):
            emb[i, j] = signal[i + j * tau]
    return emb


##########################################
@njit(parallel=True)
def compute_fuzzy_entropy_jit(signal, m=2, tau=1, r_factor=0.2):
    N = len(signal)
    if N - (m + 1) * tau + 1 <= 0:
        return 0.0

    r = r_factor * np.std(signal)
    log2_val = np.log(2.0)

    emb_m = embed_signal_fuzzEn_(signal, m, tau)
    emb_m1 = embed_signal_fuzzEn_(signal, m + 1, tau)

    count_m = 0.0
    count_m1 = 0.0
    len_m = emb_m.shape[0]
    len_m1 = emb_m1.shape[0]

    # for i in range(len_m - 1):
    for i in prange(len_m - 1):
        for j in range(i + 1, len_m):
            count_m += fuzzy_similarity_fuzzEn_(emb_m[i], emb_m[j], r, log2_val)

    # for i in range(len_m1 - 1):
    for i in prange(len_m1 - 1):
        for j in range(i + 1, len_m1):
            count_m1 += fuzzy_similarity_fuzzEn_(emb_m1[i], emb_m1[j], r, log2_val)

    total_m = (2.0 * count_m) / (len_m * (len_m - 1))
    total_m1 = (2.0 * count_m1) / (len_m1 * (len_m1 - 1))

    if total_m1 == 0.0 or total_m == 0.0:
        return 0.0

    return -np.log(total_m1 / total_m)



##########################################
#### Distribution entropy calculation
##########################################
@njit(parallel=True)
def embed_signal_for_distEn_(data, m):
    N = len(data)
    emb_len = N - m + 1
    emb = np.empty((emb_len, m))
    # for i in range(emb_len):
    for i in prange(emb_len):
        for j in range(m):
            emb[i, j] = data[i + j]
    return emb

@njit(parallel=True)
def compute_chebyshev_distances_distEn_(embedded):
    N = embedded.shape[0]
    dist_list = []
    # for i in range(N):
    for i in prange(N):
        for j in range(N):
            if i != j:
                max_diff = 0.0
                for k in range(embedded.shape[1]):
                    diff = abs(embedded[i, k] - embedded[j, k])
                    if diff > max_diff:
                        max_diff = diff
                dist_list.append(max_diff)
    return np.array(dist_list)

# @njit(parallel=True)
@njit
def compute_histogram_distEn_(distances, M):
    min_d = np.min(distances)
    max_d = np.max(distances)
    if max_d == min_d:
        return np.zeros(M)
    bin_width = (max_d - min_d) / M
    hist = np.zeros(M)
    for d in distances:
        bin_idx = int((d - min_d) / bin_width)
        if bin_idx == M:  # Handle edge case when d == max_d
            bin_idx -= 1
        hist[bin_idx] += 1
    return hist / len(distances)


##########################################
# @njit(parallel=True)
@njit
def compute_distribution_entropy_jit(data, m, M):
    embedded = embed_signal_for_distEn_(data, m)
    distances = compute_chebyshev_distances_distEn_(embedded)
    prob = compute_histogram_distEn_(distances, M)

    entropy = 0.0
    for p in prob:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy / np.log2(M)



##########################################
#### Entropy profile calculation
##########################################

@njit(parallel=True)
def embed_signal_enProf_(data, m):
    N = len(data)
    L = N - m + 1
    emb = np.empty((L, m))
    # for i in range(L):
    for i in prange(L):
        for j in range(m):
            emb[i, j] = data[i + j]
    return emb

@njit(parallel=True)
def chebyshev_distances_enProf_(emb):
    N = emb.shape[0]
    dist_matrix = np.empty((N, N))
    # for i in range(N):
    for i in prange(N):
        for j in range(N):
            if i != j:
                max_diff = 0.0
                for k in range(emb.shape[1]):
                    diff = abs(emb[i, k] - emb[j, k])
                    if diff > max_diff:
                        max_diff = diff
                dist_matrix[i, j] = max_diff
            else:
                dist_matrix[i, j] = 0.0
    return dist_matrix

# @njit(parallel=True)
@njit
def cumulative_histogram_enProf_(dist_mat, bin_edges):
    N = dist_mat.shape[0]
    num_bins = len(bin_edges) - 1
    cum_hists = np.zeros((N, num_bins))
    for i in range(N):
        hist = np.zeros(num_bins)
        for j in range(N):
            if i != j:
                d = dist_mat[i, j]
                for b in range(num_bins):
                    if bin_edges[b] <= d < bin_edges[b + 1]:
                        hist[b] += 1
                        break
        cum_hists[i, :] = np.cumsum(hist) / (N - 1)
    return cum_hists



##########################################
# @njit(parallel=True)
@njit
def compute_entropy_profile_jit(data, m):
    emb_m = embed_signal_enProf_(data, m)
    emb_m1 = embed_signal_enProf_(data, m + 1)

    dist_m = chebyshev_distances_enProf_(emb_m)
    dist_m1 = chebyshev_distances_enProf_(emb_m1)

    all_d = np.concatenate((dist_m.ravel(), dist_m1.ravel()))
    range_vals = np.unique(all_d)
    if len(range_vals) < 2:
        range_vals = np.array([range_vals[0], range_vals[0] + 1e-6])
    bin_edges = np.append(range_vals, range_vals[-1] + 1e-6)

    hist_m = cumulative_histogram_enProf_(dist_m, bin_edges)
    hist_m1 = cumulative_histogram_enProf_(dist_m1, bin_edges)

    b = hist_m.sum(0) / hist_m.shape[0]
    a = hist_m1.sum(0) / hist_m1.shape[0]

    eps = 1e-12
    ratios = (b + eps) / (a + eps)
    return np.log(ratios)





