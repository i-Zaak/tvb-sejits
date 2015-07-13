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
    
    def pseudo_dfun_test(self):
        derivative = USR([3,4]) # initialization: 3 variables, 4 modes
        
        x1 = USR([(0,1), 4 ]) # partial kill
        c1 = derivative.complement(x1) # reaching def not killed yet
        
        x2 = USR([(1,2), 4 ]) # partial kill
        c2 = c1.complement(x2) # reaching def not killed yet
        c3 = x1.complement(x2) # reaching def not killed yet

        x3 = USR([(2,3), 4 ]) # partial kill
        c4 = c2.complement(x3) # reaching def for derivative
        c5 = c3.complement(x3) # reaching def for x1
        c6 = x2.complement(x3) # reaching def for x2

        use = USR([3,4]) # now we need the whole array again
        self.assertTrue(c4.is_empty()) # initial value completely overwritten 
        self.assertTrue( c5 == x1 ) # x1 still valid reaching def
        self.assertTrue( c6 == x2 ) # x2 still valid reaching def
        self.assertTrue( x1.union(x2).union(x3) == use) # just a sanity check: we cover the whole array.
