import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import matplotlib.mlab as mlab
import scipy.signal
import hashlib
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
from scipy.ndimage.filters import maximum_filter

def gaussian_kernel_1d(sigma, kernel_size):
    # your code goes here
    const = 1 / (math.sqrt(2 * math.pi) * sigma)
    kernel = np.empty((kernel_size, 1))
    for i in range(kernel_size):
        x = i - (kernel_size - 1) / 2
        kernel[i,0] = const * math.exp(-x*x/(2*sigma*sigma))
    return kernel.T

def get_specgram(sample, chunk_size=4096, samp_freq=44100, overlap_ratio=0.25):
    arr2D = mlab.specgram(sample, NFFT=chunk_size, Fs=samp_freq, window=mlab.window_hanning,
                          noverlap=chunk_size*overlap_ratio)[0]
    # apply log transform since specgram() returns linear array
    arr2D = 10 * np.log10(arr2D)
    arr2D[arr2D == -np.inf] = 0
    return arr2D

def get_2D_peaks(spectrogram, amp_min=20, max_neighbors=20):
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, max_neighbors)
    detected_peaks = (maximum_filter(spectrogram, footprint=neighborhood) == spectrogram) * (spectrogram > amp_min)
    freqs, times = np.where(detected_peaks)

    return zip(freqs, times)

def get_2D_peaks_unsharp_mask(spectrogram, k_len=3, kernel='box', amp_min=20, max_neighbors=20, sigma=1):
    double_k = np.zeros((1,k_len))
    double_k[0][k_len//2] = 2
    if kernel=='box':
        sharp_filter = double_k - 1/k_len * np.ones((1,k_len))
    elif kernel=='gauss':
        sharp_filter = double_k - gaussian_kernel_1d(sigma, k_len)
    elif kernel=='gauss_2d':
        double_k = np.zeros((k_len,k_len))
        double_k[k_len//2][k_len//2] = 2
        g = gaussian_kernel_1d(sigma, k_len)
        sharp_filter = double_k - g*g.T
    else:
        raise ValueException('Invalid kernel')
    det = scipy.signal.convolve2d(spectrogram, sharp_filter, mode='same')
    
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, max_neighbors)
    detected_peaks = (maximum_filter(det, footprint=neighborhood) == det) * (det > amp_min)
    freqs, times = np.where(detected_peaks)

    return freqs, times

def get_2D_peaks_laplacian(spectrogram, c=0.2, amp_min=20, max_neighbors=20):
    det = spectrogram - c *scipy.signal.convolve2d(spectrogram, [[-1, 2, -1]], mode='same')
    
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, max_neighbors)
    detected_peaks = (maximum_filter(det, footprint=neighborhood) == det) * (det > amp_min)
    freqs, times = np.where(detected_peaks)

    return freqs, times

def get_2D_peaks_laplacian_2d(spectrogram, c=0.2, amp_min=20, max_neighbors=20):
    det = spectrogram - c *scipy.signal.convolve2d(spectrogram, [[0,-1,0],[-1, 2, -1],[0,-1,0]], mode='same')
    
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, max_neighbors)
    detected_peaks = (maximum_filter(det, footprint=neighborhood) == det) * (det > amp_min)
    freqs, times = np.where(detected_peaks)

    return freqs, times
    
def gen_hashes(peaks, fan_factor=15, min_delta=0, max_delta=200, bits_reduction=20):
    peaks = list(sorted(peaks, key=lambda x: x[1]))
    #print(len(peaks))
    for i, peak in enumerate(peaks):
        for j in range(1, fan_factor):
            if (i + j) < len(peaks):               
                freq1 = peak[0]
                freq2 = peaks[i + j][0]
                t1 = peak[1]
                t2 = peaks[i + j][1]
                t_delta = t2 - t1

                if t_delta >= min_delta and t_delta <= max_delta:
                    h = hashlib.sha1("{}|{}|{}".format(freq1, freq2, t_delta).encode('utf-8'))
                    yield (h.hexdigest()[:bits_reduction], t1) #(h.hexdigest()[:bits_reduction], t1) 
                    
def gen_fingerprint(song, chunk_size=4096, samp_freq=44100, overlap_ratio=0.25, amp_min=10,
                    fan_factor=15, min_delta=0, max_delta=200, bits_reduction=20):
    arr2D = get_specgram(song, chunk_size=chunk_size, samp_freq=samp_freq, overlap_ratio=overlap_ratio)
    return gen_hashes(get_2D_peaks(arr2D, amp_min=amp_min),
                      fan_factor=fan_factor, min_delta=min_delta, max_delta=max_delta, bits_reduction=bits_reduction)
    
def plot_spectrograms(spectrogram, time_idx, frequency_idx):
    plt.figure(figsize=(15,10))
    # scatter of the peaks
    plt.subplot(121)
    plt.imshow(spectrogram)
    #ax.scatter(time_idx, frequency_idx)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title("Spectrogram")
    plt.gca().invert_yaxis()

    plt.subplot(122)
    plt.imshow(spectrogram)
    plt.scatter(time_idx, frequency_idx)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.xlim(0, len(spectrogram[0]))
    plt.ylim(0, len(spectrogram))
    plt.title("Spectrogram with peaks")
#     plt.gca().invert_yaxis()

    plt.show()