import numpy as np
from collections import Counter, deque


class CoefficientTreeNode:
    def __init__(self, val, level, loc, quad, children=[]):
        """
        A class is used to construct a tree based on the location correspondence between
        points in lower sub-band and points in higher sub-bands in iterative process.

        :param val: An integer represents value of current element in coefficient matrix
        :param level: An integer represents sub-band level of current element in coefficient matrix
        :param loc: A two-dimensional tuple location of current element in coefficient matrix
        :param quad: An integer represents quadrant of current element in coefficient matrix
        :param children: A list contains children of current node
        """
        self.val = val
        self.lev = level
        self.loc = loc
        self.quad = quad
        self.children = children
        self.code = None

    def zero_code(self):
        """tree node marker"""
        for child in self.children:
            child.zero_code()

        if abs(self.val) > 0:
            self.code = "N"  # Non-zero
        else:
            self.code = "Z" if any([child.code != "T" for child in self.children]) else "T"

    @staticmethod
    def build_trees(coeffs):
        """construct forest"""
        def build_child_tree(lev, parent_loc, quad):
            """build children trees"""
            if lev + 1 > len(coeffs):
                return []
            (i, j) = parent_loc
            (H, W) = coeffs[lev][quad].shape

            loclis = [(2 * i, 2 * j),
                      (2 * i, 2 * j + 1),
                      (2 * i + 1, 2 * j),
                      (2 * i + 1, 2 * j + 1)]
            children = []

            for loc in loclis:
                if loc[0] >= H or loc[1] >= W:
                    continue
                node = CoefficientTreeNode(coeffs[lev][quad][loc[0]][loc[1]], lev, loc, quad)
                node.children = build_child_tree(lev + 1, loc, quad)
                children.append(node)
            return children

        LL = coeffs[0]  # coefficients of low frequency sub-band
        trees = []   # forest
        (H, W) = LL.shape  # the shape of LL

        # travel around all points in LL
        for i in range(H):
            for j in range(W):
                children = [CoefficientTreeNode(subband[i][j], 1, (i, j), quad, children=build_child_tree(2, (i, j), quad)) for quad, subband in enumerate(coeffs[1])]
                trees.append(CoefficientTreeNode(LL[i, j], 0, (i, j), None, children=children))
        return trees


class ZeroTreeEncoder:
    def __init__(self, coeffs):
        """
        A class is used to travel all nodes in Coefficient Tree Nodes in a certain way.

        :param coeffs: detached coefficient matrix
        """
        self.trees = CoefficientTreeNode.build_trees(coeffs)

    def travel(self):
        """travel all nodes"""
        nonzero_lis = []    # non-zero list
        sym_seq = []    # symbol sequence
        q = deque()  # queue

        for parent in self.trees:
            parent.zero_code()
            q.append(parent)

        # BFS travel
        while len(q) != 0:
            node = q.popleft()

            if node.code != "T":
                for child in node.children:
                    q.append(child)

            if node.code == "N":
                if node.quad != None:
                    # add value of node whose symbol is 'N'
                    nonzero_lis.append(node.val)
                node.val = 0

            # add node.code
            sym_seq.append(node.code)

        counter = Counter(sym_seq)
        syms = sorted(counter.keys(), key=lambda x: x)
        freq = []
        for sym in syms:
            freq.append(counter.get(sym))
        sym_freq = list(zip(syms, freq))
        nonzero_lis = np.array(nonzero_lis)

        return sym_seq, sym_freq, nonzero_lis


class ZeroTreeDecoder:
    def __init__(self, coeffs):
        """
        A class is used to refill values of all nodes by using non-zero list in a certain way.

        :param coeffs: initial detached coefficient matrix
        """
        self.coeffs = coeffs
        self.trees = CoefficientTreeNode.build_trees(self.coeffs)

    def revist(self, code_list, non_zeros_ls):
        """refill values of all nodes"""
        q = deque()
        for parent in self.trees:
            q.append(parent)

        for code in code_list:
            if len(q) == 0:
                break
            node = q.popleft()
            if code != "T":
                for child in node.children:
                    q.append(child)

            if code == "N" and node.quad != None:
                node.val = non_zeros_ls.popleft()
                self.coeffs[node.lev][node.quad][node.loc] = node.val
