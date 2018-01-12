#!/usr/bin/env python3
import os, sys
import glob
from skimage import io
import numpy as np

imgdir = sys.argv[1]
target = sys.argv[2]
target = os.path.join(imgdir, target)
imgs = glob.glob(os.path.join(imgdir, '*.jpg'))
for idx, v in enumerate(imgs):
    if target == v:
        targetid = idx

imgs = [io.imread(fname).flatten() for fname in imgs]
imgs = np.array(imgs)

def diff(t1, t2, cast=True):
    img1 = np.array(list(t1))
    img2 = np.array(list(t2))
    if cast:
        img1 -= img1.min()
        img1 /= img1.max()
        img1 = (img1*255).astype(np.int16)
        img2 -= img2.min()
        img2 /= img2.max()
        img2 = (img2*255).astype(np.int16)
        diff = np.abs(img1 - img2).astype(np.uint8)
    else:
        diff = img1.astype(np.int16) - img2
    u, cnt = np.unique(diff, return_counts=True)
    res = np.concatenate((u, cnt)).reshape(2, -1)
    print(res)

mu = imgs.mean(0)
# io.imsave('a1.ave.jpg', mu.astype(np.uint8).reshape(600, 600, 3))

print('Calculating svd ...')
imgs = imgs - mu
load = False
if not load:
    U, S, V = np.linalg.svd(imgs.T, full_matrices=False)
    np.save('u.npy', U)
    np.save('s.npy', S)
else:
    U = np.load('u.npy')
    S = np.load('s.npy')

# imgs  : samples x pixels
# imgs.T: pixels  x samples
# U:      pixels  x samples
# S:      samples
# V:      samples x samples
# weight: samples x samples
e_faces = U.T

def plot_eigenface():
    print('Output eigenface ...')
    weights = imgs.dot(e_faces.T)
    e_faces_out = -e_faces
    e_faces_out -= e_faces_out.min(1).reshape(-1, 1)
    e_faces_out /= e_faces_out.max(1).reshape(-1, 1)
    e_faces_out = (e_faces_out*255).astype(np.uint8)
    e_faces_out = e_faces_out.reshape(-1, 600, 600, 3)
    for i in range(4):
        io.imsave('eigenface_%02d.jpg' % i, e_faces_out[i])
    io.imsave('eigenface_09.jpg', e_faces_out[9])

n_pcs = 4
# Reconstruct from previous weight
# recon = mu + np.dot(weights[targetid,:n_pcs], e_faces[:n_pcs,:])
# recon -= recon.min()
# recon /= recon.max()
# recon = (recon*255).astype(np.uint8)
# recon = recon.reshape(600, 600, 3)
# io.imsave('reconstruct.jpg', recon)
# for i in range(1):
#     io.imsave('recon_%02d.jpg'%i, recon_out[i])
t = io.imread(target).flatten()
t = t.astype('float64') - mu
w = np.dot(t, U)
out = mu + np.dot(w[:n_pcs], e_faces[:n_pcs,:])
out -= out.min()
out /= out.max()
out = (out*255).astype(np.uint8).reshape(600, 600, 3)
io.imsave('reconstruction.jpg', out)

ratio = S/S.sum() * 100
print('Top 4 sigular value:\n%.1f%%\t%.1f%%\t%.1f%%\t%.1f%%' % tuple(ratio[:4]))
