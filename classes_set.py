# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 04:38:33 2019

@author: Neeraj
Description: This file contains an example of how to create classes in Python. I've create a "Set" class to which values 
can be added or removed along with a function to check the existence of a value.

Reference: Chapter 2 A crash course in Python
"""

class Set:
    # Create a Set class with operations similar to a Python set
    def __init__(self, values = None):
        self.dict ={} # a set contains a dictionary
        """ if elements are passed as argument then add them to set otherwise
        create an empty set"""
        if values is not None:
            for value in values:
                self.add(value)
      # Representation of this set          
    def __repr__(self):
        return "Set:" + str(self.dict.keys())
      # add a value to the set  
    def add(self, value):
        self.dict[value]  = True
       # delete a value from the set 
    def remove(self,value):
        del self.dict[value]
        # check if something exists in the set
    def contain(self, value):
        return value in self.dict

# Example usage
s = Set([1,2,5])

s.add(3)

s.contain(5)

s.remove(1)
 
print s 
