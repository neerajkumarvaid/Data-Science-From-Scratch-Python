#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:08:39 2020

@author: Neeraj

Description: Demonstrates how to do simple network analysis in Python. 
Reference: Chapter 22 : Network Analysis
"""
from typing import NamedTuple

class User(NamedTuple):
    id: int
    name: str
        
        
users = [User(0, "Hero"), User(1,"Dunn"), User(2, "Sue"), User(3,"Chi"), User(4, "Thor"), User(5, "Clive"),
        User(6, "Hicks"), User(7,"Devin"), User(8,"Kate"), User(9,"Klein")]

friend_pairs = [(0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)]
