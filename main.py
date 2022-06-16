from ImageEncoder import ImageEncoder
from ImageDecoder import ImageDecoder
from skimage.metrics import peak_signal_noise_ratio
import os
import cv2
import csv


def PSNR(raw, noise):
    """calculate PSNR"""
    return peak_signal_noise_ratio(raw, noise)


if __name__=="__main__":
    namelist = ["image1.512", "image2.512", "image3.512", "image4.512", "image5.512"]
    for i in range(0, len(namelist)):
        name = namelist[i]  # choose image
        path = ".\\pic\\" + name  # the path of original image
        T = 5  # the times of discrete wavelet transform
        data_path = ".\\data\\csv\\" + name.split('.')[0] + ".csv"  # save path of csv file

        # csv.writer initialization
        dfile = open(data_path, "w", newline='')
        wt = csv.writer(dfile)
        wt.writerow(["q", "D", "R", "CR"])

        # the range of quantization step is between 1 and 50.
        for q in reversed(range(1, 51)):
            dlis = []
            bf_bytes = os.path.getsize(path)

            # Encode
            print(">START ENCODING...")
            encoder = ImageEncoder(path, q, T)
            encoder.run()
            print("<ENCODING OVER!")
            af_bytes = os.path.getsize(".\\out\\image.bit")

            # Decode
            print(">START DECODING...")
            bin_path = ".\\out\\image.bit"
            decoder = ImageDecoder(bin_path, q, name)
            img = decoder.run()
            print("<DECODING OVER!")

            pic_save_path = ".\\data\\pic\\" + name.split('.')[0] + "_" + str(q) + ".png"
            # print("PSNR:", PSNR(encoder.img, img))
            # print("File size(bf):", "%d bytes" % (bf_bytes))
            # print("File size(af):", "%d bytes" % (af_bytes))
            # print("Compression Rate:", '%.2f%%' % (af_bytes / bf_bytes * 100) )
            # print("Bitrate:", "%.3f bits/pixel" % (af_bytes * 8 / (decoder.H * decoder.W)))

            D = PSNR(encoder.img, img)  # PSNR
            R = af_bytes * 8 / (decoder.H * decoder.W)  # Bitrate
            CR = bf_bytes / af_bytes  # Compression Ratio

            dlis.extend([q, D, R, CR])
            wt.writerow(dlis)
            cv2.imwrite(pic_save_path, img)
            print(">q=", q, "D=", D, "R=", R, "CR=", CR)
            # cv2.imshow("final", img)
            # cv2.waitKey()
            print()

        dfile.close()