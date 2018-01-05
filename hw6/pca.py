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
        target = idx

imgs = [io.imread(fname).flatten() for fname in imgs]
imgs = np.array(imgs)
mu = imgs.mean(0)
io.imsave('a1.ave.jpg', mu.astype(np.uint8).reshape(600, 600, 3))

print('Calculating svd ...')
imgs = imgs - mu
load = True
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
print('Converting ...')
e_faces = U.T
weights = imgs.dot(e_faces.T)

e_faces_out = -e_faces
e_faces_out -= e_faces_out.min(1).reshape(-1, 1)
e_faces_out /= e_faces_out.max(1).reshape(-1, 1)
e_faces_out = (e_faces_out*255).astype(np.uint8)
e_faces_out = e_faces_out.reshape(-1, 600, 600, 3)
print(e_faces_out.shape)
io.imsave('eigenface_09.jpg', e_faces_out[9])

n_pcs = 4
recon = mu + np.dot(weights[target,:n_pcs], e_faces[:n_pcs,:])
recon -= recon.min()
recon /= recon.max()
recon = (recon*255).astype(np.uint8)
recon = recon.reshape(600, 600, 3)
io.imsave('reconstruct.jpg', recon)
# for i in range(1):
#     io.imsave('recon_%02d.jpg'%i, recon_out[i])

ratio = S[:4]
ratio = ratio / ratio.sum()
print('%.1f\n%.1f\n%.1f\n%.1f' % tuple(ratio[:4]))
