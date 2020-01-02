import itertools

import sympy as sm
import numpy as np


class GeometricAlgebra(object):

    def __init__(self, bases, cayley_table):
        # Multiplication table (used for automatic simpliciation)
        self.cayley_table = cayley_table

        # Generate blades within this dimension
        self.blades = self.create_blades(bases)

        self.blades_list = sum(self.blades, [])
        self.coeff_grades = [len(a.free_symbols) for a in self.blades_list]

    @classmethod
    def generate(cls, p, m, z, start_inx=0):
        """
        See section (5.4) Sylvester signature theorem
            https://bivector.net/PROJECTIVE_GEOMETRIC_ALGEBRA.pdf

        Generate a symmetric bilinear form of dimension n = p + m + z

        e_i.dot(e_j) = 0

        Args:
            p (int): e_i.dot(e_i) = +1
            m (int): e_i.dot(e_i) = -1
            z (int): e_i.dot(e_i) =  0
        """
        assert (p >= 0) and (m >= 0) and (z >= 0)
        n = p + m + z
        e = sm.symbols('e{}:{}'.format(start_inx, start_inx + n),
                       commutative=False)

        subs = {}
        squares = [0] * z + [1] * p + [-1] * m
        for i in range(n):
            for j in range(n):
                if i == j:
                    subs[e[i] * e[i]] = squares[i]
                elif i > j:
                    subs[e[i] * e[j]] = -e[j] * e[i]

        return cls(e, subs)

    @staticmethod
    def create_blades(bases):
        blades = []
        for dim in range(len(bases) + 1):
            # Blades are products of all unique combinations of dim # of bases
            inds_list = itertools.combinations(range(len(bases)), dim)
            blades.append(
                [sm.S(sm.prod([bases[i] for i in inds])) for inds in inds_list])
        return blades

    def simp(self, a, collect=False):
        while True:
            b = sm.expand(a).subs(self.cayley_table, simultaneous=True)
            if b == a:
                break
            a = b
        
        if collect:
            b = sm.collect(b, sum(reversed(self.blades), []))
    
        return b

    def blade(self, dim, coeffs):
        assert (dim > 0) and (dim < len(self.blades))
        assert len(coeffs) == len(self.blades[dim])
        return sum(self.blades[dim][i] * coeff for i, coeff in enumerate(coeffs))
    
    @property
    def pseudoscalar(self):
        assert len(self.blades[-1]) == 1
        return self.blades[-1][0]
    
    def coeffs(self, a):
        blade_bases = [{e for e in blade.free_symbols}
                       for blade in self.blades_list]

        coeffs = [sm.S(0)] * len(blade_bases)
        for k, v in sm.S(a).as_coefficients_dict().items():
            for blade_inx in reversed(range(len(blade_bases))):
                if blade_bases[blade_inx].issubset(k.free_symbols):
                    coeffs[blade_inx] += k.subs(self.blades_list[blade_inx], 1) * v
                    break
    
        return coeffs
    
    def from_coeffs(self, coeffs):
        assert len(coeffs) == len(self.blades_list)
        return sum(k * v for k, v in zip(self.blades_list, coeffs))

    def to_blades(self, a):
        return [self.grade(a, i) for i in self.grades(a)]

    def is_blade(self, a):
        return len(self.grades(a)) == 1

    def grade(self, a, grade):
        return self.from_coeffs([c if self.coeff_grades[i] == grade else 0
                                for i, c in enumerate(self.coeffs(a))])

    def grades(self, a):
        """
        Return all grades contained by a.
        """
        if a == 0:
            return set([0])
        coeffs = self.coeffs(a)
        return set(g for c, g in zip(coeffs, self.coeff_grades) if c != 0)

    # -------------------------------------------------------------------------
    # Implementation of key operations
    # -------------------------------------------------------------------------

    def sum(self, a, b):
        return self.simp(a + b)
    
    def product(self, a, b):
        return self.simp(a * b)

    def inner_blade(self, a, b):
        assert self.is_blade(a)
        assert self.is_blade(b)
        output_grade = abs(self.grades(a).pop() - self.grades(b).pop())
        return self.grade(self.simp(a * b), output_grade)

    def outer_blade(self, a, b):
        assert self.is_blade(a)
        assert self.is_blade(b)
        output_grade = self.grades(a).pop() + self.grades(b).pop()
        return self.grade(self.simp(a * b), output_grade)

    def inner(self, a, b):
        return sum(self.inner_blade(a_blade, b_blade)
                for a_blade in self.to_blades(a)
                for b_blade in self.to_blades(b))

    def outer(self, a, b):
        return sum(self.outer_blade(a_blade, b_blade)
                for a_blade in self.to_blades(a)
                for b_blade in self.to_blades(b))

    def dual(self, a):
        return sum(k * v for k, v in zip(reversed(self.blades_list), self.coeffs(a)))

    def polarity(self, a):
        return self.simp(a * self.pseudoscalar)

    def reverse(self, a):
        result = 0
        for k, v in sm.S(a).as_coefficients_dict().items():
            grade = len(k.free_symbols.intersection(self.blades[1]))
            num_flips = len(list(itertools.combinations(range(grade), 2)))
            if num_flips % 2 == 1:
                result += -k * v
            else:
                result += k * v

        return self.simp(result)

    def regressive(self, a, b):
        return self.dual(self.outer(self.dual(a), self.dual(b)))

    def commutator(self, a, b):
        return self.simp(((a * b) - (b * a)) / sm.S(2))

    def sandwich(self, a, b):
        return self.product(self.product(a, b), self.reverse(a))

    def normalized(self, a, epsilon=0):
        return self.simp(a / (sm.sqrt(sm.Abs(self.product(a, a))) + epsilon))

    # -------------------------------------------------------------------------
    # Flattened codegen version of key operations
    # -------------------------------------------------------------------------
    def codegen_op(self, func, num_args, name):
        """
        Print flattened code for an operation that takes in any number of
        multivectors and returns one multivector.
        """
    
        # Create symbolic dense multivectors
        dim = len(self.blades_list)
        chars = [chr(ord('a') + i) for i in range(num_args)]
        args = [self.from_coeffs(sm.symbols('{}[0:{}]'.format(char, dim))) for char in chars]
    
        # Apply the function and simplify
        res = func(*args)
        res_coeffs = self.coeffs(self.simp(res))

        s = 'def {}({}):\n'.format(name, ', '.join(['self'] + chars))
        for char in chars:
            s += '    {} = self.coeffs({})\n'.format(char, char)

        # Print result
        s += '    res = [0] * {}\n'.format(dim)
        for i, expr in enumerate(res_coeffs):
            s += '    res[{}] = {}\n'.format(i, expr)
        s += '    return self.from_coeffs(res)'

        return s

    def codegen_unary_op(self, func, name):
        return self.codegen_op(func, 1, name)

    def codegen_binary_op(self, func, name):
        return self.codegen_op(func, 2, name)

    # -------------------------------------------------------------------------
    # Flattened codegen version of key operations
    # -------------------------------------------------------------------------
    def sum(self, a, b):
        a = self.coeffs(a)
        b = self.coeffs(b)
        res = [0] * 16
        res[0] = a[0] + b[0]
        res[1] = a[1] + b[1]
        res[2] = a[2] + b[2]
        res[3] = a[3] + b[3]
        res[4] = a[4] + b[4]
        res[5] = a[5] + b[5]
        res[6] = a[6] + b[6]
        res[7] = a[7] + b[7]
        res[8] = a[8] + b[8]
        res[9] = a[9] + b[9]
        res[10] = a[10] + b[10]
        res[11] = a[11] + b[11]
        res[12] = a[12] + b[12]
        res[13] = a[13] + b[13]
        res[14] = a[14] + b[14]
        res[15] = a[15] + b[15]
        return self.from_coeffs(res)

    def product(self, a, b):
        a = self.coeffs(a)
        b = self.coeffs(b)
        res = [0] * 16
        res[0] = a[0]*b[0] - a[10]*b[10] - a[14]*b[14] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] - a[8]*b[8] - a[9]*b[9]
        res[1] = a[0]*b[1] - a[10]*b[13] - a[11]*b[8] - a[12]*b[9] - a[13]*b[10] + a[14]*b[15] - a[15]*b[14] + a[1]*b[0] - a[2]*b[5] - a[3]*b[6] - a[4]*b[7] + a[5]*b[2] + a[6]*b[3] + a[7]*b[4] - a[8]*b[11] - a[9]*b[12]
        res[2] = a[0]*b[2] - a[10]*b[14] - a[14]*b[10] + a[2]*b[0] - a[3]*b[8] - a[4]*b[9] + a[8]*b[3] + a[9]*b[4]
        res[3] = a[0]*b[3] + a[10]*b[4] + a[14]*b[9] + a[2]*b[8] + a[3]*b[0] - a[4]*b[10] - a[8]*b[2] + a[9]*b[14]
        res[4] = a[0]*b[4] - a[10]*b[3] - a[14]*b[8] + a[2]*b[9] + a[3]*b[10] + a[4]*b[0] - a[8]*b[14] - a[9]*b[2]
        res[5] = a[0]*b[5] - a[10]*b[15] + a[11]*b[3] + a[12]*b[4] - a[13]*b[14] + a[14]*b[13] - a[15]*b[10] + a[1]*b[2] - a[2]*b[1] + a[3]*b[11] + a[4]*b[12] + a[5]*b[0] - a[6]*b[8] - a[7]*b[9] + a[8]*b[6] + a[9]*b[7]
        res[6] = a[0]*b[6] + a[10]*b[7] - a[11]*b[2] + a[12]*b[14] + a[13]*b[4] - a[14]*b[12] + a[15]*b[9] + a[1]*b[3] - a[2]*b[11] - a[3]*b[1] + a[4]*b[13] + a[5]*b[8] + a[6]*b[0] - a[7]*b[10] - a[8]*b[5] + a[9]*b[15]
        res[7] = a[0]*b[7] - a[10]*b[6] - a[11]*b[14] - a[12]*b[2] - a[13]*b[3] + a[14]*b[11] - a[15]*b[8] + a[1]*b[4] - a[2]*b[12] - a[3]*b[13] - a[4]*b[1] + a[5]*b[9] + a[6]*b[10] + a[7]*b[0] - a[8]*b[15] - a[9]*b[5]
        res[8] = a[0]*b[8] + a[10]*b[9] + a[14]*b[4] + a[2]*b[3] - a[3]*b[2] + a[4]*b[14] + a[8]*b[0] - a[9]*b[10]
        res[9] = a[0]*b[9] - a[10]*b[8] - a[14]*b[3] + a[2]*b[4] - a[3]*b[14] - a[4]*b[2] + a[8]*b[10] + a[9]*b[0]
        res[10] = a[0]*b[10] + a[10]*b[0] + a[14]*b[2] + a[2]*b[14] + a[3]*b[4] - a[4]*b[3] - a[8]*b[9] + a[9]*b[8]
        res[11] = a[0]*b[11] + a[10]*b[12] + a[11]*b[0] - a[12]*b[10] + a[13]*b[9] - a[14]*b[7] + a[15]*b[4] + a[1]*b[8] - a[2]*b[6] + a[3]*b[5] - a[4]*b[15] + a[5]*b[3] - a[6]*b[2] + a[7]*b[14] + a[8]*b[1] - a[9]*b[13]
        res[12] = a[0]*b[12] - a[10]*b[11] + a[11]*b[10] + a[12]*b[0] - a[13]*b[8] + a[14]*b[6] - a[15]*b[3] + a[1]*b[9] - a[2]*b[7] + a[3]*b[15] + a[4]*b[5] + a[5]*b[4] - a[6]*b[14] - a[7]*b[2] + a[8]*b[13] + a[9]*b[1]
        res[13] = a[0]*b[13] + a[10]*b[1] - a[11]*b[9] + a[12]*b[8] + a[13]*b[0] - a[14]*b[5] + a[15]*b[2] + a[1]*b[10] - a[2]*b[15] - a[3]*b[7] + a[4]*b[6] + a[5]*b[14] + a[6]*b[4] - a[7]*b[3] - a[8]*b[12] + a[9]*b[11]
        res[14] = a[0]*b[14] + a[10]*b[2] + a[14]*b[0] + a[2]*b[10] - a[3]*b[9] + a[4]*b[8] + a[8]*b[4] - a[9]*b[3]
        res[15] = a[0]*b[15] + a[10]*b[5] + a[11]*b[4] - a[12]*b[3] + a[13]*b[2] - a[14]*b[1] + a[15]*b[0] + a[1]*b[14] - a[2]*b[13] + a[3]*b[12] - a[4]*b[11] + a[5]*b[10] - a[6]*b[9] + a[7]*b[8] + a[8]*b[7] - a[9]*b[6]
        return self.from_coeffs(res)

    def inner(self, a, b):
        a = self.coeffs(a)
        b = self.coeffs(b)
        res = [0] * 16
        res[0] = a[0]*b[0] - a[10]*b[10] - a[14]*b[14] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] - a[8]*b[8] - a[9]*b[9]
        res[1] = a[0]*b[1] - a[10]*b[13] - a[11]*b[8] - a[12]*b[9] - a[13]*b[10] + a[14]*b[15] - a[15]*b[14] + a[1]*b[0] - a[2]*b[5] - a[3]*b[6] - a[4]*b[7] + a[5]*b[2] + a[6]*b[3] + a[7]*b[4] - a[8]*b[11] - a[9]*b[12]
        res[2] = a[0]*b[2] - a[10]*b[14] - a[14]*b[10] + a[2]*b[0] - a[3]*b[8] - a[4]*b[9] + a[8]*b[3] + a[9]*b[4]
        res[3] = a[0]*b[3] + a[10]*b[4] + a[14]*b[9] + a[2]*b[8] + a[3]*b[0] - a[4]*b[10] - a[8]*b[2] + a[9]*b[14]
        res[4] = a[0]*b[4] - a[10]*b[3] - a[14]*b[8] + a[2]*b[9] + a[3]*b[10] + a[4]*b[0] - a[8]*b[14] - a[9]*b[2]
        res[5] = a[0]*b[5] - a[10]*b[15] + a[11]*b[3] + a[12]*b[4] - a[15]*b[10] + a[3]*b[11] + a[4]*b[12] + a[5]*b[0]
        res[6] = a[0]*b[6] - a[11]*b[2] + a[13]*b[4] + a[15]*b[9] - a[2]*b[11] + a[4]*b[13] + a[6]*b[0] + a[9]*b[15]
        res[7] = a[0]*b[7] - a[12]*b[2] - a[13]*b[3] - a[15]*b[8] - a[2]*b[12] - a[3]*b[13] + a[7]*b[0] - a[8]*b[15]
        res[8] = a[0]*b[8] + a[14]*b[4] + a[4]*b[14] + a[8]*b[0]
        res[9] = a[0]*b[9] - a[14]*b[3] - a[3]*b[14] + a[9]*b[0]
        res[10] = a[0]*b[10] + a[10]*b[0] + a[14]*b[2] + a[2]*b[14]
        res[11] = a[0]*b[11] + a[11]*b[0] + a[15]*b[4] - a[4]*b[15]
        res[12] = a[0]*b[12] + a[12]*b[0] - a[15]*b[3] + a[3]*b[15]
        res[13] = a[0]*b[13] + a[13]*b[0] + a[15]*b[2] - a[2]*b[15]
        res[14] = a[0]*b[14] + a[14]*b[0]
        res[15] = a[0]*b[15] + a[15]*b[0]
        return self.from_coeffs(res)

    def outer(self, a, b):
        a = self.coeffs(a)
        b = self.coeffs(b)
        res = [0] * 16
        res[0] = a[0]*b[0]
        res[1] = a[0]*b[1] + a[1]*b[0]
        res[2] = a[0]*b[2] + a[2]*b[0]
        res[3] = a[0]*b[3] + a[3]*b[0]
        res[4] = a[0]*b[4] + a[4]*b[0]
        res[5] = a[0]*b[5] + a[1]*b[2] - a[2]*b[1] + a[5]*b[0]
        res[6] = a[0]*b[6] + a[1]*b[3] - a[3]*b[1] + a[6]*b[0]
        res[7] = a[0]*b[7] + a[1]*b[4] - a[4]*b[1] + a[7]*b[0]
        res[8] = a[0]*b[8] + a[2]*b[3] - a[3]*b[2] + a[8]*b[0]
        res[9] = a[0]*b[9] + a[2]*b[4] - a[4]*b[2] + a[9]*b[0]
        res[10] = a[0]*b[10] + a[10]*b[0] + a[3]*b[4] - a[4]*b[3]
        res[11] = a[0]*b[11] + a[11]*b[0] + a[1]*b[8] - a[2]*b[6] + a[3]*b[5] + a[5]*b[3] - a[6]*b[2] + a[8]*b[1]
        res[12] = a[0]*b[12] + a[12]*b[0] + a[1]*b[9] - a[2]*b[7] + a[4]*b[5] + a[5]*b[4] - a[7]*b[2] + a[9]*b[1]
        res[13] = a[0]*b[13] + a[10]*b[1] + a[13]*b[0] + a[1]*b[10] - a[3]*b[7] + a[4]*b[6] + a[6]*b[4] - a[7]*b[3]
        res[14] = a[0]*b[14] + a[10]*b[2] + a[14]*b[0] + a[2]*b[10] - a[3]*b[9] + a[4]*b[8] + a[8]*b[4] - a[9]*b[3]
        res[15] = a[0]*b[15] + a[10]*b[5] + a[11]*b[4] - a[12]*b[3] + a[13]*b[2] - a[14]*b[1] + a[15]*b[0] + a[1]*b[14] - a[2]*b[13] + a[3]*b[12] - a[4]*b[11] + a[5]*b[10] - a[6]*b[9] + a[7]*b[8] + a[8]*b[7] - a[9]*b[6]
        return self.from_coeffs(res)

    def dual(self, a):
        a = self.coeffs(a)
        res = [0] * 16
        res[0] = a[15]
        res[1] = a[14]
        res[2] = a[13]
        res[3] = a[12]
        res[4] = a[11]
        res[5] = a[10]
        res[6] = a[9]
        res[7] = a[8]
        res[8] = a[7]
        res[9] = a[6]
        res[10] = a[5]
        res[11] = a[4]
        res[12] = a[3]
        res[13] = a[2]
        res[14] = a[1]
        res[15] = a[0]
        return self.from_coeffs(res)

    def reverse(self, a):
        a = self.coeffs(a)
        res = [0] * 16
        res[0] = a[0]
        res[1] = a[1]
        res[2] = a[2]
        res[3] = a[3]
        res[4] = a[4]
        res[5] = -a[5]
        res[6] = -a[6]
        res[7] = -a[7]
        res[8] = -a[8]
        res[9] = -a[9]
        res[10] = -a[10]
        res[11] = -a[11]
        res[12] = -a[12]
        res[13] = -a[13]
        res[14] = -a[14]
        res[15] = a[15]
        return self.from_coeffs(res)

    def polarity(self, a):
        a = self.coeffs(a)
        res = [0] * 16
        res[0] = 0
        res[1] = a[14]
        res[2] = 0
        res[3] = 0
        res[4] = 0
        res[5] = -a[10]
        res[6] = a[9]
        res[7] = -a[8]
        res[8] = 0
        res[9] = 0
        res[10] = 0
        res[11] = -a[4]
        res[12] = a[3]
        res[13] = -a[2]
        res[14] = 0
        res[15] = a[0]
        return self.from_coeffs(res)

    def regressive(self, a, b):
        a = self.coeffs(a)
        b = self.coeffs(b)
        res = [0] * 16
        res[0] = a[0]*b[15] + a[10]*b[5] - a[11]*b[4] + a[12]*b[3] - a[13]*b[2] + a[14]*b[1] + a[15]*b[0] - a[1]*b[14] + a[2]*b[13] - a[3]*b[12] + a[4]*b[11] + a[5]*b[10] - a[6]*b[9] + a[7]*b[8] + a[8]*b[7] - a[9]*b[6]
        res[1] = a[11]*b[7] - a[12]*b[6] + a[13]*b[5] + a[15]*b[1] + a[1]*b[15] + a[5]*b[13] - a[6]*b[12] + a[7]*b[11]
        res[2] = a[11]*b[9] - a[12]*b[8] + a[14]*b[5] + a[15]*b[2] + a[2]*b[15] + a[5]*b[14] - a[8]*b[12] + a[9]*b[11]
        res[3] = a[10]*b[11] + a[11]*b[10] - a[13]*b[8] + a[14]*b[6] + a[15]*b[3] + a[3]*b[15] + a[6]*b[14] - a[8]*b[13]
        res[4] = a[10]*b[12] + a[12]*b[10] - a[13]*b[9] + a[14]*b[7] + a[15]*b[4] + a[4]*b[15] + a[7]*b[14] - a[9]*b[13]
        res[5] = -a[11]*b[12] + a[12]*b[11] + a[15]*b[5] + a[5]*b[15]
        res[6] = -a[11]*b[13] + a[13]*b[11] + a[15]*b[6] + a[6]*b[15]
        res[7] = -a[12]*b[13] + a[13]*b[12] + a[15]*b[7] + a[7]*b[15]
        res[8] = -a[11]*b[14] + a[14]*b[11] + a[15]*b[8] + a[8]*b[15]
        res[9] = -a[12]*b[14] + a[14]*b[12] + a[15]*b[9] + a[9]*b[15]
        res[10] = a[10]*b[15] - a[13]*b[14] + a[14]*b[13] + a[15]*b[10]
        res[11] = a[11]*b[15] + a[15]*b[11]
        res[12] = a[12]*b[15] + a[15]*b[12]
        res[13] = a[13]*b[15] + a[15]*b[13]
        res[14] = a[14]*b[15] + a[15]*b[14]
        res[15] = a[15]*b[15]
        return self.from_coeffs(res)

    def commutator(self, a, b):
        a = self.coeffs(a)
        b = self.coeffs(b)
        res = [0] * 16
        res[0] = 0
        res[1] = a[14]*b[15] - a[15]*b[14] - a[2]*b[5] - a[3]*b[6] - a[4]*b[7] + a[5]*b[2] + a[6]*b[3] + a[7]*b[4]
        res[2] = -a[3]*b[8] - a[4]*b[9] + a[8]*b[3] + a[9]*b[4]
        res[3] = a[10]*b[4] + a[2]*b[8] - a[4]*b[10] - a[8]*b[2]
        res[4] = -a[10]*b[3] + a[2]*b[9] + a[3]*b[10] - a[9]*b[2]
        res[5] = -a[13]*b[14] + a[14]*b[13] + a[1]*b[2] - a[2]*b[1] - a[6]*b[8] - a[7]*b[9] + a[8]*b[6] + a[9]*b[7]
        res[6] = a[10]*b[7] + a[12]*b[14] - a[14]*b[12] + a[1]*b[3] - a[3]*b[1] + a[5]*b[8] - a[7]*b[10] - a[8]*b[5]
        res[7] = -a[10]*b[6] - a[11]*b[14] + a[14]*b[11] + a[1]*b[4] - a[4]*b[1] + a[5]*b[9] + a[6]*b[10] - a[9]*b[5]
        res[8] = a[10]*b[9] + a[2]*b[3] - a[3]*b[2] - a[9]*b[10]
        res[9] = -a[10]*b[8] + a[2]*b[4] - a[4]*b[2] + a[8]*b[10]
        res[10] = a[3]*b[4] - a[4]*b[3] - a[8]*b[9] + a[9]*b[8]
        res[11] = a[10]*b[12] - a[12]*b[10] + a[13]*b[9] - a[14]*b[7] + a[15]*b[4] - a[4]*b[15] + a[7]*b[14] - a[9]*b[13]
        res[12] = -a[10]*b[11] + a[11]*b[10] - a[13]*b[8] + a[14]*b[6] - a[15]*b[3] + a[3]*b[15] - a[6]*b[14] + a[8]*b[13]
        res[13] = -a[11]*b[9] + a[12]*b[8] - a[14]*b[5] + a[15]*b[2] - a[2]*b[15] + a[5]*b[14] - a[8]*b[12] + a[9]*b[11]
        res[14] = 0
        res[15] = a[11]*b[4] - a[12]*b[3] + a[13]*b[2] - a[14]*b[1] + a[1]*b[14] - a[2]*b[13] + a[3]*b[12] - a[4]*b[11]
        return self.from_coeffs(res)

    def sandwich(self, a, b):
        a = self.coeffs(a)
        b = self.coeffs(b)
        res = [0] * 16
        res[0] = a[0]**2*b[0] + 2*a[0]*a[2]*b[2] + 2*a[0]*a[3]*b[3] + 2*a[0]*a[4]*b[4] + a[10]**2*b[0] + 2*a[10]*a[14]*b[2] + 2*a[10]*a[3]*b[4] - 2*a[10]*a[4]*b[3] + a[14]**2*b[0] + 2*a[14]*a[8]*b[4] - 2*a[14]*a[9]*b[3] + a[2]**2*b[0] + 2*a[2]*a[8]*b[3] + 2*a[2]*a[9]*b[4] + a[3]**2*b[0] - 2*a[3]*a[8]*b[2] + a[4]**2*b[0] - 2*a[4]*a[9]*b[2] + a[8]**2*b[0] + a[9]**2*b[0]
        res[1] = a[0]**2*b[1] + 2*a[0]*a[14]*b[15] + 2*a[0]*a[1]*b[0] + 2*a[0]*a[5]*b[2] + 2*a[0]*a[6]*b[3] + 2*a[0]*a[7]*b[4] + a[10]**2*b[1] + 2*a[10]*a[13]*b[0] + 2*a[10]*a[15]*b[2] - 2*a[10]*a[2]*b[15] + 2*a[10]*a[6]*b[4] - 2*a[10]*a[7]*b[3] + 2*a[11]*a[14]*b[4] + 2*a[11]*a[2]*b[3] - 2*a[11]*a[3]*b[2] + 2*a[11]*a[8]*b[0] - 2*a[12]*a[14]*b[3] + 2*a[12]*a[2]*b[4] - 2*a[12]*a[4]*b[2] + 2*a[12]*a[9]*b[0] + 2*a[13]*a[14]*b[2] + 2*a[13]*a[3]*b[4] - 2*a[13]*a[4]*b[3] - a[14]**2*b[1] + 2*a[14]*a[15]*b[0] + 2*a[15]*a[8]*b[4] - 2*a[15]*a[9]*b[3] + 2*a[1]*a[2]*b[2] + 2*a[1]*a[3]*b[3] + 2*a[1]*a[4]*b[4] - a[2]**2*b[1] + 2*a[2]*a[5]*b[0] - a[3]**2*b[1] + 2*a[3]*a[6]*b[0] + 2*a[3]*a[9]*b[15] - a[4]**2*b[1] + 2*a[4]*a[7]*b[0] - 2*a[4]*a[8]*b[15] + 2*a[5]*a[8]*b[3] + 2*a[5]*a[9]*b[4] - 2*a[6]*a[8]*b[2] - 2*a[7]*a[9]*b[2] + a[8]**2*b[1] + a[9]**2*b[1]
        res[2] = a[0]**2*b[2] + 2*a[0]*a[2]*b[0] + 2*a[0]*a[8]*b[3] + 2*a[0]*a[9]*b[4] + a[10]**2*b[2] + 2*a[10]*a[14]*b[0] + 2*a[10]*a[8]*b[4] - 2*a[10]*a[9]*b[3] + a[14]**2*b[2] + 2*a[14]*a[3]*b[4] - 2*a[14]*a[4]*b[3] + a[2]**2*b[2] + 2*a[2]*a[3]*b[3] + 2*a[2]*a[4]*b[4] - a[3]**2*b[2] + 2*a[3]*a[8]*b[0] - a[4]**2*b[2] + 2*a[4]*a[9]*b[0] - a[8]**2*b[2] - a[9]**2*b[2]
        res[3] = a[0]**2*b[3] + 2*a[0]*a[10]*b[4] + 2*a[0]*a[3]*b[0] - 2*a[0]*a[8]*b[2] - a[10]**2*b[3] + 2*a[10]*a[4]*b[0] - 2*a[10]*a[9]*b[2] + a[14]**2*b[3] - 2*a[14]*a[2]*b[4] + 2*a[14]*a[4]*b[2] - 2*a[14]*a[9]*b[0] - a[2]**2*b[3] + 2*a[2]*a[3]*b[2] - 2*a[2]*a[8]*b[0] + a[3]**2*b[3] + 2*a[3]*a[4]*b[4] - a[4]**2*b[3] - a[8]**2*b[3] - 2*a[8]*a[9]*b[4] + a[9]**2*b[3]
        res[4] = a[0]**2*b[4] - 2*a[0]*a[10]*b[3] + 2*a[0]*a[4]*b[0] - 2*a[0]*a[9]*b[2] - a[10]**2*b[4] - 2*a[10]*a[3]*b[0] + 2*a[10]*a[8]*b[2] + a[14]**2*b[4] + 2*a[14]*a[2]*b[3] - 2*a[14]*a[3]*b[2] + 2*a[14]*a[8]*b[0] - a[2]**2*b[4] + 2*a[2]*a[4]*b[2] - 2*a[2]*a[9]*b[0] - a[3]**2*b[4] + 2*a[3]*a[4]*b[3] + a[4]**2*b[4] + a[8]**2*b[4] - 2*a[8]*a[9]*b[3] - a[9]**2*b[4]
        res[5] = a[0]**2*b[5] - 2*a[0]*a[13]*b[14] + 2*a[0]*a[14]*b[13] - 2*a[0]*a[15]*b[10] + 2*a[0]*a[3]*b[11] + 2*a[0]*a[4]*b[12] - 2*a[0]*a[6]*b[8] - 2*a[0]*a[7]*b[9] + 2*a[0]*a[8]*b[6] + 2*a[0]*a[9]*b[7] + a[10]**2*b[5] + 2*a[10]*a[1]*b[14] - 2*a[10]*a[2]*b[13] + 2*a[10]*a[3]*b[12] - 2*a[10]*a[4]*b[11] + 2*a[10]*a[5]*b[10] - 2*a[10]*a[6]*b[9] + 2*a[10]*a[7]*b[8] + 2*a[10]*a[8]*b[7] - 2*a[10]*a[9]*b[6] - 2*a[11]*a[14]*b[9] - 2*a[11]*a[2]*b[8] + 2*a[11]*a[4]*b[10] - 2*a[11]*a[9]*b[14] + 2*a[12]*a[14]*b[8] - 2*a[12]*a[2]*b[9] - 2*a[12]*a[3]*b[10] + 2*a[12]*a[8]*b[14] - 2*a[13]*a[2]*b[10] + 2*a[13]*a[3]*b[9] - 2*a[13]*a[4]*b[8] - a[14]**2*b[5] + 2*a[14]*a[1]*b[10] - 2*a[14]*a[3]*b[7] + 2*a[14]*a[4]*b[6] + 2*a[14]*a[5]*b[14] - 2*a[14]*a[8]*b[12] + 2*a[14]*a[9]*b[11] - 2*a[15]*a[2]*b[14] + 2*a[15]*a[8]*b[9] - 2*a[15]*a[9]*b[8] + 2*a[1]*a[3]*b[8] + 2*a[1]*a[4]*b[9] - a[2]**2*b[5] - 2*a[2]*a[3]*b[6] - 2*a[2]*a[4]*b[7] - 2*a[2]*a[8]*b[11] - 2*a[2]*a[9]*b[12] + a[3]**2*b[5] + 2*a[3]*a[7]*b[14] - 2*a[3]*a[9]*b[13] + a[4]**2*b[5] - 2*a[4]*a[6]*b[14] + 2*a[4]*a[8]*b[13] + 2*a[5]*a[8]*b[8] + 2*a[5]*a[9]*b[9] + 2*a[6]*a[9]*b[10] - 2*a[7]*a[8]*b[10] - a[8]**2*b[5] - a[9]**2*b[5]
        res[6] = a[0]**2*b[6] + 2*a[0]*a[10]*b[7] + 2*a[0]*a[12]*b[14] - 2*a[0]*a[14]*b[12] + 2*a[0]*a[15]*b[9] - 2*a[0]*a[2]*b[11] + 2*a[0]*a[4]*b[13] + 2*a[0]*a[5]*b[8] - 2*a[0]*a[7]*b[10] - 2*a[0]*a[8]*b[5] - a[10]**2*b[6] - 2*a[10]*a[11]*b[14] + 2*a[10]*a[14]*b[11] - 2*a[10]*a[15]*b[8] - 2*a[10]*a[2]*b[12] - 2*a[10]*a[3]*b[13] + 2*a[10]*a[5]*b[9] + 2*a[10]*a[6]*b[10] - 2*a[10]*a[9]*b[5] - 2*a[11]*a[14]*b[10] - 2*a[11]*a[3]*b[8] - 2*a[11]*a[4]*b[9] + 2*a[12]*a[2]*b[10] - 2*a[12]*a[3]*b[9] + 2*a[12]*a[4]*b[8] + 2*a[13]*a[14]*b[8] - 2*a[13]*a[2]*b[9] - 2*a[13]*a[3]*b[10] + 2*a[13]*a[8]*b[14] - a[14]**2*b[6] - 2*a[14]*a[1]*b[9] + 2*a[14]*a[2]*b[7] - 2*a[14]*a[4]*b[5] + 2*a[14]*a[6]*b[14] - 2*a[14]*a[8]*b[13] - 2*a[15]*a[3]*b[14] + 2*a[15]*a[8]*b[10] - 2*a[1]*a[2]*b[8] + 2*a[1]*a[4]*b[10] - 2*a[1]*a[9]*b[14] + a[2]**2*b[6] - 2*a[2]*a[3]*b[5] - 2*a[2]*a[7]*b[14] + 2*a[2]*a[9]*b[13] - a[3]**2*b[6] - 2*a[3]*a[4]*b[7] - 2*a[3]*a[8]*b[11] - 2*a[3]*a[9]*b[12] + a[4]**2*b[6] + 2*a[4]*a[5]*b[14] - 2*a[4]*a[8]*b[12] + 2*a[4]*a[9]*b[11] - 2*a[5]*a[9]*b[10] + 2*a[6]*a[8]*b[8] + 2*a[6]*a[9]*b[9] + 2*a[7]*a[8]*b[9] - 2*a[7]*a[9]*b[8] - a[8]**2*b[6] - 2*a[8]*a[9]*b[7] + a[9]**2*b[6]
        res[7] = a[0]**2*b[7] - 2*a[0]*a[10]*b[6] - 2*a[0]*a[11]*b[14] + 2*a[0]*a[14]*b[11] - 2*a[0]*a[15]*b[8] - 2*a[0]*a[2]*b[12] - 2*a[0]*a[3]*b[13] + 2*a[0]*a[5]*b[9] + 2*a[0]*a[6]*b[10] - 2*a[0]*a[9]*b[5] - a[10]**2*b[7] - 2*a[10]*a[12]*b[14] + 2*a[10]*a[14]*b[12] - 2*a[10]*a[15]*b[9] + 2*a[10]*a[2]*b[11] - 2*a[10]*a[4]*b[13] - 2*a[10]*a[5]*b[8] + 2*a[10]*a[7]*b[10] + 2*a[10]*a[8]*b[5] - 2*a[11]*a[2]*b[10] + 2*a[11]*a[3]*b[9] - 2*a[11]*a[4]*b[8] - 2*a[12]*a[14]*b[10] - 2*a[12]*a[3]*b[8] - 2*a[12]*a[4]*b[9] + 2*a[13]*a[14]*b[9] + 2*a[13]*a[2]*b[8] - 2*a[13]*a[4]*b[10] + 2*a[13]*a[9]*b[14] - a[14]**2*b[7] + 2*a[14]*a[1]*b[8] - 2*a[14]*a[2]*b[6] + 2*a[14]*a[3]*b[5] + 2*a[14]*a[7]*b[14] - 2*a[14]*a[9]*b[13] - 2*a[15]*a[4]*b[14] + 2*a[15]*a[9]*b[10] - 2*a[1]*a[2]*b[9] - 2*a[1]*a[3]*b[10] + 2*a[1]*a[8]*b[14] + a[2]**2*b[7] - 2*a[2]*a[4]*b[5] + 2*a[2]*a[6]*b[14] - 2*a[2]*a[8]*b[13] + a[3]**2*b[7] - 2*a[3]*a[4]*b[6] - 2*a[3]*a[5]*b[14] + 2*a[3]*a[8]*b[12] - 2*a[3]*a[9]*b[11] - a[4]**2*b[7] - 2*a[4]*a[8]*b[11] - 2*a[4]*a[9]*b[12] + 2*a[5]*a[8]*b[10] - 2*a[6]*a[8]*b[9] + 2*a[6]*a[9]*b[8] + 2*a[7]*a[8]*b[8] + 2*a[7]*a[9]*b[9] + a[8]**2*b[7] - 2*a[8]*a[9]*b[6] - a[9]**2*b[7]
        res[8] = a[0]**2*b[8] + 2*a[0]*a[10]*b[9] + 2*a[0]*a[4]*b[14] - 2*a[0]*a[9]*b[10] - a[10]**2*b[8] - 2*a[10]*a[3]*b[14] + 2*a[10]*a[8]*b[10] + a[14]**2*b[8] - 2*a[14]*a[2]*b[9] - 2*a[14]*a[3]*b[10] + 2*a[14]*a[8]*b[14] - a[2]**2*b[8] + 2*a[2]*a[4]*b[10] - 2*a[2]*a[9]*b[14] - a[3]**2*b[8] - 2*a[3]*a[4]*b[9] + a[4]**2*b[8] + a[8]**2*b[8] + 2*a[8]*a[9]*b[9] - a[9]**2*b[8]
        res[9] = a[0]**2*b[9] - 2*a[0]*a[10]*b[8] - 2*a[0]*a[3]*b[14] + 2*a[0]*a[8]*b[10] - a[10]**2*b[9] - 2*a[10]*a[4]*b[14] + 2*a[10]*a[9]*b[10] + a[14]**2*b[9] + 2*a[14]*a[2]*b[8] - 2*a[14]*a[4]*b[10] + 2*a[14]*a[9]*b[14] - a[2]**2*b[9] - 2*a[2]*a[3]*b[10] + 2*a[2]*a[8]*b[14] + a[3]**2*b[9] - 2*a[3]*a[4]*b[8] - a[4]**2*b[9] - a[8]**2*b[9] + 2*a[8]*a[9]*b[8] + a[9]**2*b[9]
        res[10] = a[0]**2*b[10] + 2*a[0]*a[2]*b[14] - 2*a[0]*a[8]*b[9] + 2*a[0]*a[9]*b[8] + a[10]**2*b[10] + 2*a[10]*a[14]*b[14] + 2*a[10]*a[8]*b[8] + 2*a[10]*a[9]*b[9] + a[14]**2*b[10] + 2*a[14]*a[3]*b[8] + 2*a[14]*a[4]*b[9] + a[2]**2*b[10] - 2*a[2]*a[3]*b[9] + 2*a[2]*a[4]*b[8] - a[3]**2*b[10] + 2*a[3]*a[8]*b[14] - a[4]**2*b[10] + 2*a[4]*a[9]*b[14] - a[8]**2*b[10] - a[9]**2*b[10]
        res[11] = a[0]**2*b[11] + 2*a[0]*a[10]*b[12] - 2*a[0]*a[12]*b[10] + 2*a[0]*a[13]*b[9] - 2*a[0]*a[14]*b[7] + 2*a[0]*a[1]*b[8] - 2*a[0]*a[2]*b[6] + 2*a[0]*a[3]*b[5] + 2*a[0]*a[7]*b[14] - 2*a[0]*a[9]*b[13] - a[10]**2*b[11] + 2*a[10]*a[11]*b[10] - 2*a[10]*a[13]*b[8] + 2*a[10]*a[14]*b[6] + 2*a[10]*a[1]*b[9] - 2*a[10]*a[2]*b[7] + 2*a[10]*a[4]*b[5] - 2*a[10]*a[6]*b[14] + 2*a[10]*a[8]*b[13] + 2*a[11]*a[14]*b[14] + 2*a[11]*a[8]*b[8] + 2*a[11]*a[9]*b[9] - 2*a[12]*a[2]*b[14] + 2*a[12]*a[8]*b[9] - 2*a[12]*a[9]*b[8] - 2*a[13]*a[3]*b[14] + 2*a[13]*a[8]*b[10] - a[14]**2*b[11] + 2*a[14]*a[15]*b[8] + 2*a[14]*a[2]*b[12] + 2*a[14]*a[3]*b[13] - 2*a[14]*a[5]*b[9] - 2*a[14]*a[6]*b[10] + 2*a[14]*a[9]*b[5] - 2*a[15]*a[2]*b[9] - 2*a[15]*a[3]*b[10] + 2*a[15]*a[8]*b[14] + 2*a[1]*a[4]*b[14] - 2*a[1]*a[9]*b[10] + a[2]**2*b[11] - 2*a[2]*a[4]*b[13] - 2*a[2]*a[5]*b[8] + 2*a[2]*a[7]*b[10] + 2*a[2]*a[8]*b[5] + a[3]**2*b[11] + 2*a[3]*a[4]*b[12] - 2*a[3]*a[6]*b[8] - 2*a[3]*a[7]*b[9] + 2*a[3]*a[8]*b[6] + 2*a[3]*a[9]*b[7] - a[4]**2*b[11] + 2*a[4]*a[5]*b[10] - 2*a[4]*a[6]*b[9] + 2*a[4]*a[7]*b[8] + 2*a[4]*a[8]*b[7] - 2*a[4]*a[9]*b[6] - 2*a[5]*a[9]*b[14] + a[8]**2*b[11] + 2*a[8]*a[9]*b[12] - a[9]**2*b[11]
        res[12] = a[0]**2*b[12] - 2*a[0]*a[10]*b[11] + 2*a[0]*a[11]*b[10] - 2*a[0]*a[13]*b[8] + 2*a[0]*a[14]*b[6] + 2*a[0]*a[1]*b[9] - 2*a[0]*a[2]*b[7] + 2*a[0]*a[4]*b[5] - 2*a[0]*a[6]*b[14] + 2*a[0]*a[8]*b[13] - a[10]**2*b[12] + 2*a[10]*a[12]*b[10] - 2*a[10]*a[13]*b[9] + 2*a[10]*a[14]*b[7] - 2*a[10]*a[1]*b[8] + 2*a[10]*a[2]*b[6] - 2*a[10]*a[3]*b[5] - 2*a[10]*a[7]*b[14] + 2*a[10]*a[9]*b[13] + 2*a[11]*a[2]*b[14] - 2*a[11]*a[8]*b[9] + 2*a[11]*a[9]*b[8] + 2*a[12]*a[14]*b[14] + 2*a[12]*a[8]*b[8] + 2*a[12]*a[9]*b[9] - 2*a[13]*a[4]*b[14] + 2*a[13]*a[9]*b[10] - a[14]**2*b[12] + 2*a[14]*a[15]*b[9] - 2*a[14]*a[2]*b[11] + 2*a[14]*a[4]*b[13] + 2*a[14]*a[5]*b[8] - 2*a[14]*a[7]*b[10] - 2*a[14]*a[8]*b[5] + 2*a[15]*a[2]*b[8] - 2*a[15]*a[4]*b[10] + 2*a[15]*a[9]*b[14] - 2*a[1]*a[3]*b[14] + 2*a[1]*a[8]*b[10] + a[2]**2*b[12] + 2*a[2]*a[3]*b[13] - 2*a[2]*a[5]*b[9] - 2*a[2]*a[6]*b[10] + 2*a[2]*a[9]*b[5] - a[3]**2*b[12] + 2*a[3]*a[4]*b[11] - 2*a[3]*a[5]*b[10] + 2*a[3]*a[6]*b[9] - 2*a[3]*a[7]*b[8] - 2*a[3]*a[8]*b[7] + 2*a[3]*a[9]*b[6] + a[4]**2*b[12] - 2*a[4]*a[6]*b[8] - 2*a[4]*a[7]*b[9] + 2*a[4]*a[8]*b[6] + 2*a[4]*a[9]*b[7] + 2*a[5]*a[8]*b[14] - a[8]**2*b[12] + 2*a[8]*a[9]*b[11] + a[9]**2*b[12]
        res[13] = a[0]**2*b[13] - 2*a[0]*a[11]*b[9] + 2*a[0]*a[12]*b[8] - 2*a[0]*a[14]*b[5] + 2*a[0]*a[1]*b[10] - 2*a[0]*a[3]*b[7] + 2*a[0]*a[4]*b[6] + 2*a[0]*a[5]*b[14] - 2*a[0]*a[8]*b[12] + 2*a[0]*a[9]*b[11] + a[10]**2*b[13] + 2*a[10]*a[11]*b[8] + 2*a[10]*a[12]*b[9] + 2*a[10]*a[13]*b[10] + 2*a[10]*a[15]*b[14] + 2*a[10]*a[2]*b[5] + 2*a[10]*a[3]*b[6] + 2*a[10]*a[4]*b[7] + 2*a[10]*a[8]*b[11] + 2*a[10]*a[9]*b[12] + 2*a[11]*a[3]*b[14] - 2*a[11]*a[8]*b[10] + 2*a[12]*a[4]*b[14] - 2*a[12]*a[9]*b[10] + 2*a[13]*a[14]*b[14] + 2*a[13]*a[8]*b[8] + 2*a[13]*a[9]*b[9] - a[14]**2*b[13] + 2*a[14]*a[15]*b[10] - 2*a[14]*a[3]*b[11] - 2*a[14]*a[4]*b[12] + 2*a[14]*a[6]*b[8] + 2*a[14]*a[7]*b[9] - 2*a[14]*a[8]*b[6] - 2*a[14]*a[9]*b[7] + 2*a[15]*a[3]*b[8] + 2*a[15]*a[4]*b[9] + 2*a[1]*a[2]*b[14] - 2*a[1]*a[8]*b[9] + 2*a[1]*a[9]*b[8] - a[2]**2*b[13] + 2*a[2]*a[3]*b[12] - 2*a[2]*a[4]*b[11] + 2*a[2]*a[5]*b[10] - 2*a[2]*a[6]*b[9] + 2*a[2]*a[7]*b[8] + 2*a[2]*a[8]*b[7] - 2*a[2]*a[9]*b[6] + a[3]**2*b[13] - 2*a[3]*a[5]*b[9] - 2*a[3]*a[6]*b[10] + 2*a[3]*a[9]*b[5] + a[4]**2*b[13] + 2*a[4]*a[5]*b[8] - 2*a[4]*a[7]*b[10] - 2*a[4]*a[8]*b[5] + 2*a[6]*a[8]*b[14] + 2*a[7]*a[9]*b[14] - a[8]**2*b[13] - a[9]**2*b[13]
        res[14] = a[0]**2*b[14] + 2*a[0]*a[2]*b[10] - 2*a[0]*a[3]*b[9] + 2*a[0]*a[4]*b[8] + a[10]**2*b[14] + 2*a[10]*a[14]*b[10] + 2*a[10]*a[3]*b[8] + 2*a[10]*a[4]*b[9] + a[14]**2*b[14] + 2*a[14]*a[8]*b[8] + 2*a[14]*a[9]*b[9] + a[2]**2*b[14] - 2*a[2]*a[8]*b[9] + 2*a[2]*a[9]*b[8] + a[3]**2*b[14] - 2*a[3]*a[8]*b[10] + a[4]**2*b[14] - 2*a[4]*a[9]*b[10] + a[8]**2*b[14] + a[9]**2*b[14]
        res[15] = a[0]**2*b[15] + 2*a[0]*a[11]*b[4] - 2*a[0]*a[12]*b[3] + 2*a[0]*a[13]*b[2] - 2*a[0]*a[14]*b[1] + 2*a[0]*a[15]*b[0] + a[10]**2*b[15] - 2*a[10]*a[11]*b[3] - 2*a[10]*a[12]*b[4] - 2*a[10]*a[1]*b[2] + 2*a[10]*a[2]*b[1] - 2*a[10]*a[5]*b[0] + 2*a[11]*a[4]*b[0] - 2*a[11]*a[9]*b[2] - 2*a[12]*a[3]*b[0] + 2*a[12]*a[8]*b[2] + 2*a[13]*a[2]*b[0] + 2*a[13]*a[8]*b[3] + 2*a[13]*a[9]*b[4] - a[14]**2*b[15] - 2*a[14]*a[1]*b[0] - 2*a[14]*a[5]*b[2] - 2*a[14]*a[6]*b[3] - 2*a[14]*a[7]*b[4] + 2*a[15]*a[2]*b[2] + 2*a[15]*a[3]*b[3] + 2*a[15]*a[4]*b[4] - 2*a[1]*a[8]*b[4] + 2*a[1]*a[9]*b[3] - a[2]**2*b[15] + 2*a[2]*a[6]*b[4] - 2*a[2]*a[7]*b[3] - a[3]**2*b[15] - 2*a[3]*a[5]*b[4] + 2*a[3]*a[7]*b[2] - 2*a[3]*a[9]*b[1] - a[4]**2*b[15] + 2*a[4]*a[5]*b[3] - 2*a[4]*a[6]*b[2] + 2*a[4]*a[8]*b[1] + 2*a[6]*a[9]*b[0] - 2*a[7]*a[8]*b[0] + a[8]**2*b[15] + a[9]**2*b[15]
        return self.from_coeffs(res)

    # aliases
    join = regressive
    meet = outer


class ProjectiveGeometry3D(GeometricAlgebra):
    """
    Projective geometric algebra in 3D space - R(3, 0, 1)

    Scalars are dual to pseudoscalar
    Planes are the same as vectors (vector is 0 fourth component, plane through origin)
    Points are dual to planes
    Lines are intermediate, dual to themselves

    https://bivector.net/3DPGA.pdf
    """
    @classmethod
    def create(cls):
        return ProjectiveGeometry3D.generate(3, 0, 1)

    # -----------------------------------------------------------------------
    # Pretty sure are correct
    # -----------------------------------------------------------------------

    # One-vector is plane
    # A plane is defined using its homogenous equation ax + by + cz + d = 0 
    def plane(self, a, b, c, d):
        return self.blade(1, (d, a, b, c))

    def line(self, e01, e02, e03, e12, e13, e23):
        return self.blade(2, (e01, e02, e03, e12, e13, e23))

    # Three-vector is a point
    # A point is just a homogeneous point, euclidean coordinates plus the origin
    def point(self, x, y, z):
        return self.blade(3, (x, y, z, 1))

    # An ideal point is a vector (direction)
    # def direction(self, x, y, z):
    #     return self.blade(3, (x, y, z, 0))

    def format(self, a):
        grades = self.grades(a)
        if grades == {0}:
            s = '<Scalar {}>'.format(a)
        elif grades == {1}:
            s = '<Plane {}>'.format(a)
        elif grades == {2}:
            s = '<Line {}>'.format(a)
        elif grades == {3}:
            s = '<Point {}>'.format(a)
        elif grades == {4}:
            s = '<Pseudoscalar {}>'
        else:
            s = '<Multivector {}>'.format(a)
    
        return s

    # -----------------------------------------------------------------------
    # WIP
    # -----------------------------------------------------------------------

    # ehhhhhhh
    def inverse(self, a):
        return self.simp(self.reverse(a) / self.simp(self.reverse(a) * a))

    def plane_normal(self, plane):
        return self.inner(plane, self.point(0, 0, 0))

    def line_through_origin(self, x, y, z):
        return self.line(0, 0, 0, x, y, z)

    def _rotator(self, line, angle):
        assert self.grades(line) == {2}
        assert self.grades(angle) == {0}
        half_angle = sm.S(angle) / 2
        return sm.cos(half_angle) + sm.sin(half_angle) * line

    def _translator(self, line, distance):
        assert self.grades(line) == {2}
        assert self.grades(distance) == {0}
        assert all(c == 0 for c in self.coeffs(line)[8:11])
        return 1 + sm.S(distance) / 2 * line

    def translator(self, x, y, z):
        return self.simp(self._translator(self.line(x, -y, z, 0, 0, 0), 1))

    def axis_angle(self, axis, angle, epsilon=0, normalize=False):
        line = self.line(0, 0, 0, axis[0], axis[1], axis[2])
        if normalize:
            line = self.normalized(line, epsilon=epsilon)
        return self.simp(self._rotator(line, angle))
