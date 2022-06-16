import cv2
import huffman
import numpy as np
from WaveletTransformer import WaveletTransformer
from bitarray import bitarray
from ZeroTreeRelativeClasses import ZeroTreeDecoder
from tqdm import tqdm
from collections import deque

# Start of Image
SOI = bytes.fromhex("FFD8")
# Start of Scan 1
SOS1 = bytes.fromhex("FFD9")
# Start of Scan 2
SOS2 = bytes.fromhex("FFDA")
# End of File
EOI = bytes.fromhex("FFDB")


class ImageDecoder:
    def __init__(self, file_path, q, img_name):
        """
        A class is used to decode binary file and reconstruct image.

        :param file_path: A string represents the path of binary file
        :param q: quantization step
        :param img_name: the name of image
        """
        self.path = file_path
        self.q = q  # step size of quantization
        self.name = img_name

        # (H, W) image.shape
        self.H = None
        self.W = None
        # (h, w) LL.shape
        self.h = None
        self.w = None
        # the number of down sampling
        self.T = None

        self.img = None  # reconstructed img

        self.dwt_img = None  # the image undergoes wavelet transform encoding
        self.q_img = None

    def run(self):
        """
        A method is used to run decode process

        :return: reconstructed image
        """
        # read binary file
        coeffs, codes, non_zeros = self.readBinaryFile()
        # rebuild zero tree and initialize coefficients
        decoder = ZeroTreeDecoder(coeffs)
        # fill coefficients with zero tree
        decoder.revist(codes, non_zeros)
        # obtain quantization image by synthesis matrices of coefficients
        q_img = self.synthesis_split_quantization_area(decoder.coeffs)
        # inverse quantization
        self.dwt_img = self.inverseQuantization(q_img)
        # inverse discrete wavelet transformation
        self.img = self.IDWT()

        return self.img

    def synthesis_split_quantization_area(self, coeffs):
        """synthesis all sub-bands"""
        q_img = np.zeros((self.H, self.W), np.float32)
        h = self.h
        w = self.w
        levels = self.T
        for i in range(levels + 1):
            if i == 0:
                q_img[0:h, 0:w] = coeffs[i]
            else:
                (q_img[0:h, w:2*w],
                 q_img[h:2*h, w:2*w],
                 q_img[h:2*h, 0:w]) = coeffs[i]
                h *= 2
                w *= 2
        return q_img

    def readBinaryFile(self):
        """read binary file"""
        cnt = 0
        # codebook used to reconstruct LL area
        LL_cdbk = None
        # codebook used to reconstruct non-zero list
        Non_cdbk = None
        # codebook used to reconstruct zero tree
        Zt_cdbk = None

        with open(self.path, "rb") as rf:
            # Start of Image
            soi = rf.read(2)
            cnt += 2
            #
            if soi != SOI:
                raise Exception("There isn't a SOI symbol. Please check your codes.")

            # obtain height(H), width(W), quantization step(q) and times of up sampling(T)
            H = int.from_bytes(rf.read(4), 'big')
            self.H = H
            W = int.from_bytes(rf.read(4), 'big')
            self.W = W
            q = int.from_bytes(rf.read(4), 'big')
            self.q = q
            T = int.from_bytes(rf.read(4), 'big')
            self.T = T

            # (h, w) the shape of LL area
            self.h = H // 2 ** self.T
            self.w = W // 2 ** self.T

            cnt += 4 * 4
            sos1 = rf.read(2)
            cnt += 2
            # Start of Scan 1
            if sos1 != SOS1:
                raise Exception("There isn't a SOI1 symbol. Please check your codes.")

            LL_size_freq_length = int.from_bytes(rf.read(4), 'big')
            cnt += 4
            LL_size_freq = []
            for i in range(LL_size_freq_length):
                size = int.from_bytes(rf.read(1), 'big')
                freq = int.from_bytes(rf.read(4), 'big')
                LL_size_freq.append((size, freq))
                cnt += 5

            LL_cdbk = huffman.codebook(LL_size_freq)

        rf.close()

        rf = open(self.path, "rb")
        ba = bitarray()
        ba.fromfile(rf)
        del ba[:cnt * 8]

        # reverse key-value pairs of LL codebook
        rev_LL_cdbk = {value: key for (key, value) in LL_cdbk.items()}

        NEXT_SYMBOL = bitarray()
        NEXT_SYMBOL.frombytes(SOS2)

        LL_seq = []
        while (ba[:16] != NEXT_SYMBOL):
            # obtain run length
            run_length = int("".join(ba[:8].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            del ba[:8]
            if run_length == 255:
                # encounter breakpoint
                LL_seq.extend([0 for i in range(255)])
                continue

            # obtain size
            cursor = 1
            size = None
            prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            while (prefix not in rev_LL_cdbk.keys()):
                cursor += 1
                prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            size = rev_LL_cdbk.get(prefix)
            del ba[:cursor]

            if run_length == 0 and size == 0:
                zeros = [0 for i in range(self.h * self.w - len(LL_seq))]
                LL_seq.extend(zeros)
            else:
                index = int("".join(ba[: size].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
                amplitude = self.size_index2amp(size, index)
                zeros = [0 for i in range(run_length)]
                LL_seq.extend(zeros)
                LL_seq.append(amplitude)
                del ba[:size]

        del ba[:16]

        LL_seq = np.array(LL_seq, dtype=np.float32)
        LL_q_img = self.rev_prediction(LL_seq).reshape(self.h, self.w)

        coeffs = self.get_init_coeffs()
        coeffs[0] = LL_q_img

        NEXT_SYMBOL = bitarray()
        NEXT_SYMBOL.frombytes(EOI)

        non_zero_length = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
        del ba[: 32]
        non_zero_size_freq_length = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
        del ba[: 32]

        non_zero_size_freq = []
        sizes = []
        freqs =[]
        for i in range(non_zero_size_freq_length):
            size = int("".join(ba[: 8].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            del ba[:8]
            freq = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            del ba[:32]
            sizes.append(size)
            freqs.append(freq)

        non_zero_size_freq = list(zip(sizes, freqs))
        Non_cdbk = huffman.codebook(non_zero_size_freq)
        rev_Non_cdbk = {value: key for (key, value) in Non_cdbk.items()}

        # obtain non_zeros list
        non_zeros = deque()
        print("-OBTAINING NON-ZERO LIST...")
        for i in tqdm(range(non_zero_length)):
            # print(i)
            cursor = 1
            prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            while (prefix not in rev_Non_cdbk.keys()):
                cursor += 1
                prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            size = rev_Non_cdbk.get(prefix)
            del ba[:cursor]

            index = int("".join(ba[: size].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            amplitude = self.size_index2amp(size, index)
            non_zeros.append(amplitude)
            del ba[:size]
        print("-    OK!")
        H_sym_seq_length = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
        del ba[: 32]
        H_sym_freq_length = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
        del ba[: 32]

        # obtain codes list
        codes = []
        syms = []
        freqs = []

        for i in range(H_sym_freq_length):
            sym = chr(int("".join(ba[: 8].decode({'1': bitarray('1'), '0': bitarray('0')})), 2))
            del ba[:8]
            freq = int("".join(ba[: 32].decode({'1': bitarray('1'), '0': bitarray('0')})), 2)
            del ba[:32]
            syms.append(sym)
            freqs.append(freq)
            
        sym_freq = list(zip(syms, freqs))
        Zt_cdbk = huffman.codebook(sym_freq)
        # new_Zt_cdbk = {key: bitarray(value) for (key, value) in Zt_cdbk.items()}
        rev_Zt_cdbk = {value: key for (key, value) in Zt_cdbk.items()}

        print("-OBTAINING CODES LIST...")
        for i in tqdm(range(H_sym_seq_length)):
            cursor = 1
            prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            while (prefix not in rev_Zt_cdbk.keys()):
                cursor += 1
                prefix = "".join(ba[:cursor].decode({'1': bitarray('1'), '0': bitarray('0')}))
            code = rev_Zt_cdbk.get(prefix)
            del ba[:cursor]
            codes.append(code)
        print("-    OK!")

        if ba[:16] == NEXT_SYMBOL:
            print("FINISH!")
        rf.close()

        return coeffs, codes, non_zeros

    def get_init_coeffs(self):
        """get initial coefficients matrix"""
        levels = self.T
        img = np.zeros((self.H, self.W), dtype= np.float32)
        (H, W) = img.shape

        h = H // 2 ** levels
        w = W // 2 ** levels

        coeffs = []
        for i in range(levels + 1):
            if i == 0:
                coeffs.append(img[: h, : w])
            else:
                # print(w, 2*w)
                coeffs.append([img[:h, w: 2 * w],
                               img[h:2 * h, w: 2 * w],
                               img[h: 2 * h, : w]])
                h *= 2
                w *= 2

        return coeffs

    def rev_prediction(self, seq):
        for i in range(0, len(seq) - 1):
            seq[i + 1] = seq[i] + seq[i + 1]
        return seq

    def size_index2amp(self, size, index):
        if index < 2 ** (size - 1):
            return index - 2 ** size + 1
        else:
            return index

    def saveDecodedFile(self):
        """save decoded image"""
        try:
            path = ".\\out\\" + self.name + ".png"
            cv2.imwrite(path, self.img)
        except Exception as e:
            print(e)

    def inverseQuantization(self, q_img):
        """inverse quantization"""
        q_img = q_img * self.q
        return q_img

    def IDWT(self):
        """execute DWT synthesis"""
        wtf = WaveletTransformer(self.dwt_img, self.T, (self.h, self.w), "s")
        img = wtf.IDWT()
        return img

