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
        print('\nRate: {0:.2f}%'.format(float(len(cypher)) / len(plain) * 100))
        fs.close()
        ft.close()

    def decode(self, source, target):
        print('Decoding ...')
        fs = open(source, 'rb')
        ft = open(target, 'wb')
        (bits_len_str, fq_str, cypher) = self.split(fs.read())
        plain = self.huff.decode(int(bits_len_str), json.loads(fq_str), cypher)
        ft.write(plain)
        print('\nDone')
        fs.close()
        ft.close()

    def link(self, bits_len, fq, cypher):
        return str(bits_len) + ' ' + fq + '\n' + cypher

    def split(self, mix):  # split encoded message into three part
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

    def encode(self, plain):
        fq = self.__count(plain)
        self.__fq2nodes(fq)
        self.__gen_tree()
        self.__gen_code(self.tree, '')
        bits_len, cypher = self.__compress(plain)
        return bits_len, fq, cypher

    def __count(self, plain):
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
        plain_len = len(plain)
        for i, byte in enumerate(plain):
            bits_tmp.append(self.code[byte])
            sys.stdout.write(
                '\r' + str(round(float(i) / plain_len * 50, 2)) + '%')
            sys.stdout.flush()
        bits = ''.join(bits_tmp)
        bits_len = len(bits)
        bits += (8 - bits_len % 8) * '0'
        for i, x in enumerate(range(len(bits))[0::8]):
            cypher_tmp.append(chr(int(bits[x:x + 8], 2)))
            sys.stdout.write(
                '\r' + "{0:.2f}%".format(float(i) / bits_len * 400 + 50))
            sys.stdout.flush()
        cypher = ''.join(cypher_tmp)
        return bits_len, cypher

    def decode(self, bits_len, fq, cypher):
        bits = self.__get_bits(cypher, bits_len)
        self.__fq2nodes(fq)
        self.__gen_tree()
        plain = self.__decompress(bits)
        return plain

    def __get_bits(self, cypher, bits_len):
        bits_tmp = []
        for l in cypher:
            bits_tmp.append(('0' * 8 + bin(ord(l))[2:])[-8:])
        bits = ''.join(bits_tmp)[0:bits_len]
        return bits

    def __decompress(self, bits):
        tmp, bits_len = [], len(bits)
        current = self.tree
        for i, bit in enumerate(bits):
            if bit == '0':
                current = current.left
            elif bit == '1':
                current = current.right
            if current.left is None:
                tmp.append(current.l)
                current = self.tree
            sys.stdout.write(
                '\r' + "{0:.2f}%".format(float(i) / bits_len * 100))
            sys.stdout.flush()

        plain = ''.join(tmp)
        return plain


if __name__ == '__main__':
    args = sys.argv
    n = len(args)
    if n != 4:
        print('Argument Error!')
    elif args[1] == 'encode':
        Encoder().encode(args[2], args[3])
    elif args[1] == 'decode':
        Encoder().decode(args[2], args[3])
    else:
        print('Error!')
