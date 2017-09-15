import sys
from PIL import Image

im = Image.open(sys.argv[1])
pix = im.load()
width, height = im.size

fout = Image.new("RGB", im.size)
pixelsNew = fout.load()
for i in range(0,width):
    for j in range(0,height):
        pixelsNew[i, j] = (pix[i, j][0] // 2,\
                pix[i, j][1] // 2,\
                pix[i, j][2] // 2)

fout.save("Q2.png")
