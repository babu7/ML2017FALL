import sys
from PIL import Image

print(sys.argv[1])
im = Image.open(sys.argv[1])
pix = im.load()
im1 = Image.open(sys.argv[2])
pix1 = im1.load()
im2 = Image.open(sys.argv[3])
pix2 = im2.load()
width, height = im1.size

for i in range(0,width):
    for j in range(0,height):
        if pix1[i, j] != pix2[i, j]:
            print("original: (%d, %d, %d)" % (pix[i, j][0],pix[i, j][1],pix[i, j][2]))


            print("[%d, %d] (%d, %d, %d) (%d, %d, %d)" % (i, j, pix1[i, j][0], pix1[i, j][1], pix1[i, j][2], pix2[i, j][0], pix2[i, j][1], pix2[i, j][2]))

