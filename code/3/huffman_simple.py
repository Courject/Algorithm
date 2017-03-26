#!/usr/bin/python
# Huffman-code Encoder
# Author: HongXin
# 2016.10.28

import json
import sys


class Encoder():
    def __init__(self):
        self.huff = Huffman()

    def encode(self, source, target):
        print('Encoding ...')
        fs = open(source, 'rb')
        ft = open(target, 'wb')
        plain = fs.read()
        (bits_len, fq, cypher) = self.huff.encode(plain)
        ft.write(self.link(bits_len, json.dumps(fq), cypher))
        print('Rate: {0:.2f}%'.format(float(len(cypher)) / len(plain) * 100))
        fs.close()
        ft.close()

    def decode(self, source, target):
        print('Decoding ...')
        fs = open(source, 'rb')
        ft = open(target, 'wb')
        (bits_len_str, fq_str, cypher) = self.split(fs.read())
        plain = self.huff.decode(int(bits_len_str), json.loads(fq_str), cypher)
        ft.write(plain)
        print('Done')
        fs.close()
        ft.close()

    @staticmethod
    def link(bits_len, fq, cypher):
        return str(bits_len) + ' ' + fq + '\n' + cypher

    @staticmethod
    def split(mix):  # split encoded message into three part
        x = mix.find(' ')
        y = mix.find('\n')
        return mix[0:x], mix[x + 1:y], mix[y + 1:]


class Node:
    def __init__(self, l, fq=0, left=None, right=None):
        self.l = l
        self.fq = fq
        self.right = right
        self.left = left


class Huffman:
    tree = None
    __nodes = {}
    code = {}

    def __init__(self):
        pass

    # Encode Part
    def encode(self, plain):
        fq = self.__count(plain)
        self.__fq2nodes(fq)
        self.__gen_tree()
        self.__gen_code(self.tree, '')
        print(json.dumps(self.code, indent=4, sort_keys=True))
        bits_len, cypher = self.__compress(plain)
        return bits_len, fq, cypher

    @staticmethod
    def __count(plain):
        fq = {}
        for l in plain:
            fq[l] = fq.get(l, 0) + 1
        return fq

    def __fq2nodes(self, fq):
        for key, value in fq.iteritems():
            self.__nodes[key] = Node(key, value)

    def __pop_min(self):
        return self.__nodes.pop(min(self.__nodes.values(),
                                    key=lambda x: x.fq).l)

    def __push_link(self, left, right):
        root = Node(left.l + right.l, left.fq + right.fq, left, right)
        self.__nodes[root.l] = root

    def __gen_tree(self):
        if len(self.__nodes) == 1:  # for one node tree
            self.tree = Node('', 0, self.__nodes.values()[0])
        else:
            while len(self.__nodes) > 1:
                self.__push_link(self.__pop_min(), self.__pop_min())
            self.tree = self.__nodes.values()[0]

    def __gen_code(self, root, c):
        if root.left is None:
            self.code[root.l] = c
        else:
            self.__gen_code(root.left, c + '0')
            self.__gen_code(root.right, c + '1')

    def __compress(self, plain):
        bits_tmp, cypher_tmp = [], []
        # compress according to huffman-code
        for byte in plain:
            bits_tmp.append(self.code[byte])
        bits = ''.join(bits_tmp)
        bits_len = len(bits)
        bits += (8 - bits_len % 8) * '0'    # fill '0' to the end
        # bits to bytes
        for x in range(len(bits))[0::8]:
            cypher_tmp.append(chr(int(bits[x:x + 8], 2)))
        cypher = ''.join(cypher_tmp)
        return bits_len, cypher

    # Decode part
    def decode(self, bits_len, fq, cypher):
        bits = self.__get_bits(cypher, bits_len)
        self.__fq2nodes(fq)
        self.__gen_tree()
        plain = self.__decompress(bits)
        return plain

    @staticmethod
    def __get_bits(cypher, bits_len):
        bits = []
        for l in cypher:
            bits.append(('0' * 8 + bin(ord(l))[2:])[-8:])
        return ''.join(bits)[0:bits_len]

    def __decompress(self, bits):
        plain = []
        current = self.tree
        for bit in bits:
            if bit == '0':
                current = current.left
            elif bit == '1':
                current = current.right
            if current.left is None:
                plain.append(current.l)
                current = self.tree
        return ''.join(plain)

if __name__ == '__main__':
    args = sys.argv
    n = len(args)
    if n != 4:
        print('Argument Error!')
    elif args[1] == 'encode':
        Encoder().encode(args[2], args[3])
    elif args[1] == 'decode':
        Encoder().decode(args[2], args[3])
