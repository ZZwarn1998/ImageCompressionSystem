import copy
import math
import sys
import numpy as np
from WaveletTransformer import WaveletTransformer
from ZeroTreeRelativeClasses import ZeroTreeEncoder
from collections import Counter
from bitarray import bitarray
import huffman

# Start of Image
SOI = bitarray()
SOI.frombytes(b'\xff\xd8')
# Start of Image
SOS1 = bitarray()
SOS1.frombytes(b'\xff\xd9')
# Start of Image
SOS2 = bitarray()
SOS2.frombytes(b'\xff\xda')
# Start of Image
EOI = bitarray()
EOI.frombytes(b'\xff\xdb')

class ImageEncoder:
    def __init__(self, img_path, q, T=5):
        """
        A class is used to transform an image into binary file.

        :param img_path: A string represents the path of original image
        :param q: An integer represents quantization step size
        :param T: An integer represents number of DWT
        """
        self.img = self.readRawFile(img_path)  # raw image
        self.q = q  # step size of quantization
        self.T = T  # the number of down sampling

        self.dwt_img = None  # image undergoes DWT
        self.q_img = None   # image undergoes quantization

    def run(self):
        """
            Run encoder to encode image.

            The structure of encoding file image.bit:
            -----------------------------------
            [SOI] 2bytes
            -----------------------------------
              H   4bytes
              W   4bytes
              *Q   *4bytes
              T   4bytes
            -----------------------------------
            [SOS1] 2bytes
            -----------------------------------
            LENGTH OF |LL SIZE_FREQ| 4bytes
            |Huffman code 1|
            size, freq  1byte, 4bytes
            |01 stream of LL area|
            ...
            -----------------------------------
            [SOS2] 2bytes
            -----------------------------------
            LENGTH OF |NON-ZERO LIST|   4bytes
            LENGTH OF |NON-ZERO SIZE_FREQ| 4bytes
            |huffman code 2|
            size, freq 1byte, 4bytes
            |01 stream of non-zero numbers|
            ...
            LENGTH OF |HIGH SYM_FREQ|   4bytes
            |Huffman code 3|
            - Z, freq   1byte, 4bytes
            - N, freq   1byte, 4bytes
            - T, freq   1byte, 4bytes
            |01 stream of others area|
            ...
            -----------------------------------
            [EOI] 2bytes
            -----------------------------------

            PS: Base on the requirement of homework, character marked with * will be removed.

        Returns: None
        """
        # DWT
        dwt_img = self.DWT()
        self.dwt_img = dwt_img
        # Quantization
        q_img = self.quantization(dwt_img)
        self.q_img = copy.copy(q_img)

        # add prediction to lowest frequency sub-band and scan lowest frequency sub-band by raster scan
        (h, w) = (self.img.shape[0] // 2 ** (self.T), self.img.shape[1] // 2 ** (self.T))
        LL = copy.copy(q_img[0:h, 0:w])
        LL_seq = LL.reshape(-1)
        LL_seq = self.prediction(LL_seq)
        # run level representation
        LL_sym_seq, LL_size_freq = self.run_level_repre(LL_seq)
        # encode symbol sequence of lowest frequency sub-band to 01 string and get the count of each item type
        LL_bits_str = self.ll_symseq2bitsstr(LL_sym_seq, LL_size_freq)  ##

        # zero tree scan
        coeffs = self.split_quantization_area()
        zerotree_encoder = ZeroTreeEncoder(coeffs)
        H_sym_seq, H_sym_freq, nonzero_lis = zerotree_encoder.travel()
        nonzero_lis_length = len(nonzero_lis)
        nonzero_bits_str, nonzero_size_freq = self.nozerolis2bitsstr(nonzero_lis)
        H_bits_str = self.h_symseq2bitsstr(H_sym_seq, H_sym_freq)

        # output bits string into binary file
        output = ".\\out\\image.bit"
        file = open(output, "wb")

        BINFILE = bitarray()
        BINFILE.extend(SOI)

        (H, W) = self.img.shape
        bH = '{:032b}'.format(H)
        bW = '{:032b}'.format(W)
        bQ = '{:032b}'.format(self.q)
        bT = '{:032b}'.format(self.T)
        str1 = bH + bW + bQ + bT
        bstr1 = bitarray(str1)
        BINFILE.extend(bstr1)

        BINFILE.extend(SOS1)
        LL_size_freq_length_str = "{:032b}".format(len(LL_size_freq))
        LL_size_freq_str = self.sym_freq2str(LL_size_freq)
        str2 = LL_size_freq_length_str + LL_size_freq_str + LL_bits_str
        bstr2 = bitarray(str2)
        BINFILE.extend(bstr2)

        BINFILE.extend(SOS2)
        nonzero_lis_length_str = "{:032b}".format(nonzero_lis_length)
        nonzero_size_freq_length_str = "{:032b}".format(len(nonzero_size_freq))
        nonzero_size_freq_str = self.sym_freq2str(nonzero_size_freq)
        str3 = nonzero_lis_length_str + nonzero_size_freq_length_str + nonzero_size_freq_str + nonzero_bits_str
        bstr3 = bitarray(str3)
        BINFILE.extend(bstr3)

        H_sym_freq_length_str = "{:032b}".format(len(H_sym_freq))
        H_sym_freq_str = self.sym_freq2str(H_sym_freq)
        H_sym_seq_length_str = "{:032b}".format(len(H_sym_seq))
        str4 = H_sym_seq_length_str + H_sym_freq_length_str + H_sym_freq_str + H_bits_str
        bstr4 = bitarray(str4)
        BINFILE.extend(bstr4)

        BINFILE.extend(EOI)
        BINFILE.tofile(file)
        file.close()

        return

    def sym_freq2str(self, sym_freq):
        """transform symbol-frequency pairs into a string which is made up of 0 and 1"""
        bstr = ""
        for item in sym_freq:
            key = item[0] if type(item[0]) != type('a') else ord(item[0])
            val = item[1]
            bstr = bstr + "{:08b}".format(key) + "{:032b}".format(val)

        return bstr

    def split_quantization_area(self):
        """split matrix after quantization"""
        levels = self.T
        q_img = copy.copy(self.q_img)
        (H, W) = self.img.shape

        h = H // 2 ** levels
        w = W // 2 ** levels

        coeffs = []
        for i in range(levels + 1):
            if i == 0:
                coeffs.append(q_img[: h, : w])
            else:
                # print(w, 2*w)
                coeffs.append([q_img[:h, w: 2 * w],
                               q_img[h:2 * h, w: 2 * w],
                               q_img[h: 2 * h, : w]])
                h *= 2
                w *= 2

        return coeffs

    def readRawFile(self, path):
        """read original image"""
        img = None
        try:
            img = np.fromfile(path, dtype="uint8")
            img = img.reshape(512, 512)
        except Exception as e:
            print(e)
        return img

    def DWT(self):
        """execute DWT"""
        wtf = WaveletTransformer(self.img, self.T, self.img.shape, mode="a")
        img = wtf.DWT()
        return img

    def quantization(self, dwt_img):
        """quantization"""
        q_img = np.round(dwt_img / self.q)
        return q_img

    def prediction(self, seq):
        """add prediction to lowest frequency sub-band"""
        for i in reversed(range(1, len(seq))):
            seq[i] = seq[i] - seq[i - 1]
        return seq

    def val2size(self, val):
        """obtain the size of value"""
        return math.ceil(math.log2(abs(val) + 0.1))

    def amp2bitstr(self,  size, amp):
        """use size and amplitude to obtain bit string"""
        try:
            if amp == 0:
                raise Exception
        except Exception as e:
            sys.exit("ZERO doesn't have amplitude. Please check your code.")

        index = -1
        if amp > 0:
            index = amp
        elif amp < 0:
            index = amp + 2**size - 1
        fstr = '{:0' + str(size) + 'b}'
        bitstr = fstr.format(index)

        return bitstr

    def run_level_repre(self, seq):
        """change into run level representation"""
        sym_seq = []
        collect = []
        cursor = 0

        hd = int(seq[cursor])
        DC = (0, self.val2size(hd), hd)
        collect.append(self.val2size(hd))
        nozero_loc = np.nonzero(seq)[0]
        sym_seq.append(DC)

        for i, cursor in enumerate(nozero_loc[:-1]):
            next_cursor = nozero_loc[i + 1]
            zero_num = next_cursor - cursor - 1
            if zero_num < 255:
                val = np.int(seq[next_cursor])
                size = self.val2size(val)
                collect.append(size)
                mark = (zero_num, size, val)
                sym_seq.append(mark)
            else:
                k = zero_num // 255
                r = zero_num % 255
                val = np.int(seq[next_cursor])
                size = self.val2size(val)
                collect.append(size)
                mark = []
                for i in range(k):
                    mark.append((255,))
                mark.append((r, size, val))
                sym_seq.extend(mark)
        if nozero_loc[-1] < len(seq) - 1:
            sym_seq.append((0, 0))
            collect.append(0)

        counter = Counter(collect)

        sizes = sorted(counter.keys(), key= lambda x: x)
        freq = []

        for item in sizes:
            freq.append(counter.get(item))

        size_freq = list(zip(sizes, freq))
        return sym_seq, size_freq

    def ll_symseq2bitsstr(self, seq, sym_freq):
        """change symbol-frequency pairs in lowest frequency sub-band into a string which is made up of 0 and 1"""
        cdbk = huffman.codebook(sym_freq)
        bits_str = ""

        for item in seq:
            bin_str = ""
            if len(item) == 3:
                run_length = item[0]
                size = item[1]
                amplitude = item[2]

                binstr_rl = '{:08b}'.format(run_length)
                binstr_s = cdbk.get(size)
                binstr_a = self.amp2bitstr(size, amplitude)

                bin_str = binstr_rl + binstr_s + binstr_a
            elif len(item) == 2:
                run_length = item[0]
                size = item[1]

                binstr_rl = '{:08b}'.format(run_length)
                binstr_s = cdbk.get(size)

                bin_str = binstr_rl + binstr_s
            elif len(item) == 1:
                run_length = item[0]
                binstr_rl = '{:08b}'.format(run_length)

                bin_str = binstr_rl
            bits_str = bits_str + bin_str
            pass
        return bits_str

    def h_symseq2bitsstr(self, seq, sym_freq):
        """change symbol-frequency pairs in other frequency sub-band into a string which is made up of 0 and 1"""
        cdbk = huffman.codebook(sym_freq)
        ba_val = []
        for val in cdbk.values():
            ba_val.append(bitarray(val))

        dic = dict(zip(cdbk.keys(), ba_val))
        ba = bitarray()
        ba.encode(dic, seq)
        bits_str = ''.join(ba.decode({'1': bitarray('1'), '0': bitarray('0')}))

        return bits_str

    def nozerolis2bitsstr(self, lis):
        """transform non-zero list into a string which is made up of 0 and 1 and obtain symbol-frequency pairs"""
        sizes = []
        for val in lis:
            sizes.append(self.val2size(val))
        counter = Counter(sizes)

        syms = sorted(counter.keys(), key=lambda x: x)
        freq = []
        for sym in syms:
            freq.append(counter.get(sym))
        sym_freq = list(zip(syms, freq))
        cdbk = huffman.codebook(sym_freq)

        bits_str = ""
        for val in lis:
            size = self.val2size(val)
            binstr_s = cdbk.get(size)
            binstr_a = self.amp2bitstr(size, int(val))
            bits_str = bits_str + binstr_s + binstr_a

        return bits_str, sym_freq