import csv

import matplotlib.pyplot as plt

if __name__ == "__main__":
    namelis = ["lena", "barb", "image2"]
    for i in range(3):
        name = namelis[i]
        path = ".\\data\\collect\\" + name + "\\" + name + ".csv"
        file = open(path, "r")
        rd = csv.reader(file)
        Q = []
        D = []
        R = []
        CR = []
        hd = next(rd)
        for [q, d, r, cr] in rd:
            # print(type(d))
            Q.append(float(q))
            D.append(float(d))
            R.append(float(r))
            CR.append(float(cr))
        file.close()
        plt.figure(name, figsize=(16,9))
        plt.subplot(2, 2, 1)
        plt.plot(D, R, c="black")
        plt.xlabel("D (dB) ", fontname="Times New Roman", fontsize="14")
        plt.ylabel("R (bpp) ", fontname="Times New Roman", fontsize="14")
        plt.title("R - D", fontname="Times New Roman", fontsize="14")
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(Q, R, c="black")
        plt.xlabel("Q (step)", fontname="Times New Roman", fontsize="14")
        plt.ylabel("R (bpp)", fontname="Times New Roman", fontsize="14")
        plt.title("R - Q", fontname="Times New Roman", fontsize="14")
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(Q, D, c="black")
        plt.xlabel("Q (step)", fontname="Times New Roman", fontsize="14")
        plt.ylabel("D (dB)", fontname="Times New Roman", fontsize="14")
        plt.title("D - Q", fontname="Times New Roman", fontsize="14")
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(Q, CR, c="black")
        plt.xlabel("Q (step)", fontname="Times New Roman", fontsize="14")
        plt.ylabel("CR", fontname="Times New Roman", fontsize="14")
        plt.title("CR - Q", fontname="Times New Roman", fontsize="14")
        plt.grid()

        plt.tight_layout()
        plt.savefig(".\\fig\\" + name + ".png")
        plt.show()

