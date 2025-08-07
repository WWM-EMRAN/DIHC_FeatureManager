
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

from scipy.stats import entropy

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


@njit(parallel=True)
def embed_signal_fuzzy_(signal, m, tau):
    N = len(signal)
    num_vectors = N - (m - 1) * tau
    emb = np.empty((num_vectors, m))
    for i in prange(num_vectors):
        for j in range(m):
            emb[i, j] = signal[i + j * tau]
    return emb


@njit
def compute_fuzzy_similarity_(vec1, vec2, r, log2_val):
    max_diff = 0.0
    for i in range(len(vec1)):
        diff = abs(vec1[i] - vec2[i])
        if diff > max_diff:
            max_diff = diff
    return np.exp(-log2_val * (max_diff / r) ** 2)


@njit(parallel=True)
def compute_fuzzy_entropy_jit(signal, m=2, tau=1, r_factor=0.2):
    N = len(signal)
    if N - (m + 1) * tau + 1 <= 0:
        return 0.0

    r = r_factor * np.std(signal)
    log2_val = np.log(2.0)

    emb_m = embed_signal_fuzzy_(signal, m, tau)
    emb_m1 = embed_signal_fuzzy_(signal, m + 1, tau)

    count_m = 0.0
    count_m1 = 0.0
    len_m = emb_m.shape[0]
    len_m1 = emb_m1.shape[0]

    for i in prange(len_m - 1):
        for j in range(i + 1, len_m):
            count_m += compute_fuzzy_similarity_(emb_m[i], emb_m[j], r, log2_val)

    for i in prange(len_m1 - 1):
        for j in range(i + 1, len_m1):
            count_m1 += compute_fuzzy_similarity_(emb_m1[i], emb_m1[j], r, log2_val)

    total_m = (2.0 * count_m) / (len_m * (len_m - 1))
    total_m1 = (2.0 * count_m1) / (len_m1 * (len_m1 - 1))

    if total_m1 == 0.0 or total_m == 0.0:
        return 0.0

    return -np.log(total_m1 / total_m)




##########################################
#### Distribution entropy calculation
##########################################
# @njit(parallel=True)
# def embed_signal_dist(data, m):
#     N = len(data)
#     L = N - m + 1
#     emb = np.empty((L, m))
#     for i in prange(L):
#         for j in range(m):
#             emb[i, j] = data[i + j]
#     return emb
#
# @njit(parallel=True)
# def compute_chebyshev_distances_dist(embedded):
#     N = embedded.shape[0]
#     dist_list = np.empty(N * (N - 1) // 2)
#     idx = 0
#     for i in prange(N - 1):
#         for j in range(i + 1, N):
#             max_diff = 0.0
#             for k in range(embedded.shape[1]):
#                 diff = abs(embedded[i, k] - embedded[j, k])
#                 if diff > max_diff:
#                     max_diff = diff
#             dist_list[idx] = max_diff
#             idx += 1
#     return dist_list[:idx]
#
# @njit
# def compute_histogram_dist(distances, M):
#     min_d = np.min(distances)
#     max_d = np.max(distances)
#     if max_d == min_d:
#         return np.zeros(M)
#     bin_width = (max_d - min_d) / M
#     hist = np.zeros(M)
#     for d in distances:
#         bin_idx = int((d - min_d) / bin_width)
#         if bin_idx >= M:
#             bin_idx = M - 1
#         hist[bin_idx] += 1
#     return hist / len(distances)
#
# @njit
# def compute_distribution_entropy_jit(data, m=2, M=500):
#     embedded = embed_signal_dist(data, m)
#     distances = compute_chebyshev_distances_dist(embedded)
#     prob = compute_histogram_dist(distances, M)
#
#     entropy = 0.0
#     for p in prob:
#         if p > 0:
#             entropy -= p * np.log2(p)
#     return entropy / np.log2(M)



# @njit(parallel=True)
# def embed_signal_dist_(data, m):
#     N = len(data)
#     L = N - m + 1
#     emb = np.empty((L, m))
#     # ✅ Match MATLAB template matrix: columns as delayed versions
#     for i in prange(m):
#         for j in range(L):
#             emb[j, i] = data[i + j]
#     return emb
#
#
# @njit(parallel=True)
# def compute_chebyshev_distances_dist_(embedded):
#     N = embedded.shape[0]
#     # ✅ MATLAB loops over all pairs, excluding self
#     dist_list = np.empty(N * (N - 1), dtype=np.float64)
#     idx = 0
#     for i in prange(N):
#         for j in range(N):
#             if i != j:
#                 max_diff = 0.0
#                 for k in range(embedded.shape[1]):
#                     diff = abs(embedded[i, k] - embedded[j, k])
#                     if diff > max_diff:
#                         max_diff = diff
#                 dist_list[idx] = max_diff
#                 idx += 1
#     return dist_list[:idx]
#
#
# @njit
# def compute_histogram_dist_(distances, M):
#     min_d = np.min(distances)
#     max_d = np.max(distances)
#     if max_d == min_d:
#         return np.zeros(M)
#
#     # Generate M bin centers (MATLAB style)
#     bin_centers = np.linspace(min_d, max_d, M)
#     hist = np.zeros(M)
#
#     for d in distances:
#         # Find nearest bin center
#         closest_bin = 0
#         min_diff = abs(d - bin_centers[0])
#         for i in range(1, M):
#             diff = abs(d - bin_centers[i])
#             if diff < min_diff:
#                 min_diff = diff
#                 closest_bin = i
#         hist[closest_bin] += 1
#     return hist / len(distances)


# @njit
# def compute_distribution_entropy_jit(data, m=2, M=500):
#     data = np.asarray(data).flatten()
#     emb = embed_signal_dist_(data, m)
#     distances = compute_chebyshev_distances_dist_(emb)
#     prob = compute_histogram_dist_(distances, M)
#
#     entropy = 0.0
#     for p in prob:
#         if p > 0:
#             entropy -= p * np.log2(p)
#     return entropy / np.log2(M)




# @njit
# def compute_distribution_entropy_jit(data, m=2, M=500):
#     data = np.asarray(data, dtype=np.float64)
#     N = len(data)
#
#     # === Embedding matrix ===
#     L = N - m + 1
#     tmpltMatM = np.empty((L, m), dtype=np.float64)
#     for j in range(m):
#         for i in range(L):
#             tmpltMatM[i, j] = data[i + j]
#
#     # === Distance matrix (Chebyshev) ===
#     allDist = []
#     for i in range(L):
#         tmpl_vec = tmpltMatM[i]
#         for j in range(L):
#             if i != j:
#                 max_diff = 0.0
#                 for k in range(m):
#                     diff = abs(tmpl_vec[k] - tmpltMatM[j, k])
#                     if diff > max_diff:
#                         max_diff = diff
#                 allDist.append(max_diff)
#
#     # Convert to NumPy array
#     allDist = np.array(allDist)
#
#     # === Histogram using M fixed bins (Peng Li method) ===
#     min_d = np.min(allDist)
#     max_d = np.max(allDist)
#     if min_d == max_d:
#         return 0.0
#
#     bin_edges = np.linspace(min_d, max_d, M + 1)
#     hist = np.zeros(M, dtype=np.float64)
#
#     for d in allDist:
#         for b in range(M):
#             if bin_edges[b] <= d < bin_edges[b + 1]:
#                 hist[b] += 1
#                 break
#             elif d == bin_edges[-1]:  # Edge case: exactly max
#                 hist[-1] += 1
#                 break
#
#     prob = hist / len(allDist)
#     entropy = 0.0
#     for p in prob:
#         if p > 0:
#             entropy -= p * np.log2(p)
#     entropy /= np.log2(M)
#     return entropy




##------------------------

# @njit(parallel=True)
# def embed_signal_dist_(data, m, N, L):
#     embedded = np.empty((L, m), dtype=np.float64)
#     for i in prange(m):
#         for j in range(L):
#             embedded[j, i] = data[i + j]
#     return embedded


# @njit(parallel=True)
# def compute_chebyshev_distances_dist_(embedded, L, m):
#     distances = []
#     for i in prange(L):
#         tmpl_vec = embedded[i]
#         for j in range(L):
#             if i != j:
#                 max_diff = 0.0
#                 for k in range(m):
#                     diff = abs(tmpl_vec[k] - embedded[j, k])
#                     if diff > max_diff:
#                         max_diff = diff
#                 distances.append(max_diff)
#     # Convert to NumPy array
#     distances = np.array(distances, dtype=np.float64)
#     return distances


# @njit
# def compute_histogram_dist_(distances, M):
#     min_d = np.min(distances)
#     max_d = np.max(distances)
#     # if min_d == max_d:
#     #     return 0.0
#     if min_d == max_d:
#         return np.zeros(M, dtype=np.float64)
#
#     bin_edges = np.linspace(min_d, max_d, M + 1)
#     hist = np.zeros(M, dtype=np.float64)
#
#     for d in distances:
#         for b in range(M):
#             if bin_edges[b] <= d < bin_edges[b + 1]:
#                 hist[b] += 1
#                 break
#             elif d == bin_edges[-1]:  # Edge case: exactly max
#                 hist[-1] += 1
#                 break
#
#     prob = hist / len(distances)
#     return prob



# @njit
# def compute_distribution_entropy_jit(data, m=2, M=500):
#     data = np.asarray(data, dtype=np.float64)
#     N = len(data)
#     L = N - m + 1
#
#     # === Embedding matrix ===
#     embedded = embed_signal_dist_(data, m, N, L)
#
#     # === Distance matrix (Chebyshev) ===
#     distances = compute_chebyshev_distances_dist_(embedded, L, m)
#
#     # === Histogram using M fixed bins (Peng Li method) ===
#     prob = compute_histogram_dist_(distances, M)
#
#     dist_en = 0.0
#     for p in prob:
#         if p > 0:
#             dist_en -= p * np.log2(p)
#     dist_en /= np.log2(M)
#     return dist_en



#---------------------------------




@njit
def compute_distribution_entropy_jit(data, m=2, M=500):
    data = np.asarray(data, dtype=np.float64)
    N = len(data)
    L = N - m + 1

    # === Embedding matrix ===
    embedded = np.empty((L, m), dtype=np.float64)
    for i in prange(m):
        for j in range(L):
            embedded[j, i] = data[i + j]

    # === Distance matrix (Chebyshev) ===
    distances = []
    for i in prange(L):
        tmpl_vec = embedded[i]
        for j in range(L):
            if i != j:
                max_diff = 0.0
                for k in range(m):
                    diff = abs(tmpl_vec[k] - embedded[j, k])
                    if diff > max_diff:
                        max_diff = diff
                distances.append(max_diff)
    # Convert to NumPy array
    distances = np.array(distances, dtype=np.float64)

    # === Histogram using M fixed bins (Peng Li method) ===
    min_d = np.min(distances)
    max_d = np.max(distances)
    if min_d == max_d:
        return 0.0
    # if min_d == max_d:
    #     return np.zeros(M, dtype=np.float64)

    bin_edges = np.linspace(min_d, max_d, M + 1)
    hist = np.zeros(M, dtype=np.float64)

    for d in distances:
        for b in range(M):
            if bin_edges[b] <= d < bin_edges[b + 1]:
                hist[b] += 1
                break
            elif d == bin_edges[-1]:  # Edge case: exactly max
                hist[-1] += 1
                break

    prob = hist / len(distances)

    dist_en = 0.0
    for p in prob:
        if p > 0:
            dist_en -= p * np.log2(p)
    dist_en /= np.log2(M)
    return dist_en




##########################################
#### Entropy profile calculation
##########################################
# @njit(parallel=True)
# def embed_signal_enProf_(data, m):
#     N = len(data)
#     L = N - m + 1
#     emb = np.empty((L, m))
#     for i in prange(L):
#         for j in range(m):
#             emb[i, j] = data[i + j]
#     return emb
#
# @njit(parallel=True)
# def compute_chebyshev_matrix_enProf_(emb):
#     N = emb.shape[0]
#     mat = np.empty((N, N))
#     for i in prange(N):
#         for j in range(N):
#             if i != j:
#                 max_diff = 0.0
#                 for k in range(emb.shape[1]):
#                     diff = abs(emb[i, k] - emb[j, k])
#                     if diff > max_diff:
#                         max_diff = diff
#                 mat[i, j] = max_diff
#             else:
#                 mat[i, j] = 0.0
#     return mat
#
# @njit
# def cumulative_histogram_enProf_(dist_mat, bin_edges):
#     N = dist_mat.shape[0]
#     num_bins = len(bin_edges) - 1
#     cum_hists = np.zeros((N, num_bins))
#     for i in range(N):
#         hist = np.zeros(num_bins)
#         for j in range(N):
#             if i != j:
#                 d = dist_mat[i, j]
#                 for b in range(num_bins - 1):
#                     if bin_edges[b] <= d < bin_edges[b + 1]:
#                         hist[b] += 1
#                         break
#         cum_hists[i, :] = np.cumsum(hist) / (N - 1)
#     return cum_hists
#
# @njit
# def compute_entropy_profile_jit(data, m):
#     emb_m = embed_signal_enProf_(data, m)
#     emb_m1 = embed_signal_enProf_(data, m + 1)
#
#     dist_m = compute_chebyshev_matrix_enProf_(emb_m)
#     dist_m1 = compute_chebyshev_matrix_enProf_(emb_m1)
#
#     all_d = np.concatenate((dist_m.ravel(), dist_m1.ravel()))
#     range_vals = np.unique(all_d)
#     if len(range_vals) < 2:
#         range_vals = np.array([range_vals[0], range_vals[0] + 1e-6])
#     bin_edges = np.append(range_vals, range_vals[-1] + 1e-6)
#
#     hist_m = cumulative_histogram_enProf_(dist_m, bin_edges)
#     hist_m1 = cumulative_histogram_enProf_(dist_m1, bin_edges)
#
#     b = hist_m.sum(0) / hist_m.shape[0]
#     a = hist_m1.sum(0) / hist_m1.shape[0]
#
#     eps = 1e-12
#     ratios = (b + eps) / (a + eps)
#     return np.log(ratios)



@njit(parallel=True)
def embed_signal_enProf_(data, m):
    N = len(data)
    L = N - m + 1
    emb = np.empty((L, m))
    for i in prange(L):
        for j in range(m):
            emb[i, j] = data[i + j]
    return emb

@njit(parallel=True)
def compute_chebyshev_distances_enProf_(emb):
    N = emb.shape[0]
    dist_matrix = np.empty((N, N), dtype=np.float64)
    for i in prange(N):
        for j in range(N):
            if i != j:
                max_diff = 0.0
                for k in range(emb.shape[1]):
                    diff = abs(emb[i, k] - emb[j, k])
                    if diff > max_diff:
                        max_diff = diff
                # dist_matrix[i, j] = round(max_diff, 3)
                dist_matrix[i, j] = np.floor(max_diff * 1000 + 0.5) / 1000
            else:
                dist_matrix[i, j] = 0.0
    return dist_matrix

@njit
def cumulative_histogram_matrix_enProf_(dist_mat, range_vals):
    N = dist_mat.shape[0]
    num_bins = len(range_vals)
    cum_hist = np.zeros((N, num_bins), dtype=np.float64)
    bin_edges = np.append(range_vals, range_vals[-1] + 1e-6)  # bin edges
    for i in range(N):
        hist = np.zeros(num_bins, dtype=np.float64)
        for j in range(N):
            if i != j:
                d = dist_mat[i, j]
                for b in range(num_bins):
                    if bin_edges[b] <= d < bin_edges[b + 1]:
                        hist[b] += 1
                        break
        cum_hist[i, :] = np.cumsum(hist) / (N - 1)
    return cum_hist


@njit
def compute_entropy_profile_jit(data, m):
    data = np.asarray(data).flatten()
    emb_m = embed_signal_enProf_(data, m)
    emb_m1 = embed_signal_enProf_(data, m + 1)

    dist_m = compute_chebyshev_distances_enProf_(emb_m)
    dist_m1 = compute_chebyshev_distances_enProf_(emb_m1)

    D = np.concatenate((dist_m.ravel(), dist_m1.ravel()))
    range_vals = np.unique(D)  # unique values used for binning

    if len(range_vals) < 2:
        return np.zeros(1)

    hist_m = cumulative_histogram_matrix_enProf_(dist_m, range_vals)
    hist_m1 = cumulative_histogram_matrix_enProf_(dist_m1, range_vals)

    b = hist_m.sum(axis=0) / hist_m.shape[0]
    a = hist_m1.sum(axis=0) / hist_m1.shape[0]

    eps = 1e-12
    ratio = (b + eps) / (a + eps)
    return np.log(ratio)






