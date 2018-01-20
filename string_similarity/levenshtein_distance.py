# -*- coding: utf-8 -*-
"""
Levenshtein distance for measuring string difference
Created on Apr. 6th, 2017 
@author: Stephen
"""
import numpy as np

class levenshtein_distance:
    def le_dis(self, input_x, input_y):
        xlen = len(input_x) + 1  
        ylen = len(input_y) + 1

        dp = np.zeros(shape=(xlen, ylen), dtype=int)
        for i in range(0, xlen):
            dp[i][0] = i
        for j in range(0, ylen):
            dp[0][j] = j

        for i in range(1, xlen):
            for j in range(1, ylen):
                if input_x[i - 1] == input_y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[xlen - 1][ylen - 1]


if __name__ == '__main__':
    ld = levenshtein_distance()
    print(ld.le_dis('abcd', 'abd'))  # print out 1
    print(ld.le_dis('ace', 'abcd'))   # print out 2
    print(ld.le_dis('hello world', 'hey word'))   # print out 4
    