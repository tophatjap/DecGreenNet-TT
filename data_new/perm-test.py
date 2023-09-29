#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:11:16 2023

@author: kishan
"""

import itertools
a =  ["".join(seq) for seq in itertools.product("01", repeat=2)]

b=  ["".join(seq) for seq in itertools.product("01", repeat=3)]

print(int(a[3][1])*5)
