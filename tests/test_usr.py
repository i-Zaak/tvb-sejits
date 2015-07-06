import unittest

from nes.usr import *
from sympy import ProductSet, Range, EmptySet


class USRTest(unittest.TestCase):
    def setUp(self):

        self.empty_usr = USR()
        self.fusr = USR(subscripts = [3,(5,9), (3,10,2)]) # [0:3, 5:9, 3:10:2]
        self.usr1 = USR(subscripts=[2,5])  # [0:2, 0:5]
        self.usr2 = USR(subscripts=[(2,5),5])  # [2:5, 0:5]
        self.usr12 = USR(subscripts=[5,5])  # [0:5, 0:5] 


    # constructor tests are ugly (?) and dependent on the underlying implementation
    def empty_constructor_test(self):
        self.assertTrue(self.empty_usr._symbolic_set is EmptySet()) 
        self.assertTrue(self.empty_usr.is_empty())

    def single_dimension_lmad_test(self):
        single_dim = USR(subscripts = [5])
        self.assertTrue(single_dim._symbolic_set == Range(5))

    def multidimensional_lmad_test(self):
        multi_dim = USR(subscripts = [5,8])
        self.assertTrue(multi_dim._symbolic_set == ProductSet(Range(5),Range(8)))

    def fancy_lmad_test(self):
        self.assertTrue(self.fusr._symbolic_set == ProductSet(Range(3), Range(5,9), Range(3,10,2)  ))

    def equalities_test(self):
        fusr = USR(subscripts = [3,(5,9), (3,10,2)]) # [0:3, 5:9, 3:10:2]
        usr1 = USR(subscripts=[2,5])  # [0:2, 0:5]
        self.assertTrue(self.fusr == fusr) 
        self.assertTrue(self.usr1 == usr1)
        self.assertFalse(self.usr1 == self.usr2)
        self.assertFalse(self.usr1 == self.usr12)

    def union_test(self):
        self.assertTrue( (USR().union(USR())).is_empty() )
        self.assertFalse(self.fusr.union(USR()).is_empty())
        self.assertTrue(self.fusr.union(USR()) == self.fusr)
        self.assertTrue(self.usr1.union(self.usr2) == self.usr12)

    def intersection_test(self):
        self.assertTrue((USR().intersect(self.fusr) ).is_empty())
        self.assertTrue((self.fusr.intersect(USR() ) ).is_empty())
        self.assertTrue( self.usr1.intersect(self.usr12) == self.usr1) 
        
    def complement_test(self):
        self.assertTrue((USR().complement(self.fusr) ).is_empty())
        self.assertTrue((self.fusr.complement( USR()) ) == self.fusr)
        self.assertTrue( self.usr12.complement(self.usr1) == self.usr2)
        self.assertTrue( self.usr12.complement(self.usr2) == self.usr1)
        self.assertFalse( self.usr12.complement(self.usr2) == self.usr2)
        self.assertFalse( self.usr12.complement(self.usr1) == self.usr1)

