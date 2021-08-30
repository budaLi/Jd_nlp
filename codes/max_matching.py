# coding=utf-8
# Copyright (c) 2020 ichinae.com, Inc. All Rights Reserved
"""
Module Summary Here.
Authors: lijinjun1351@ichinae.com
"""

# 前向最大匹配
def forward_max_mathcing(mathing_str,dic,max_len):
    cur_start= 0
    cur_end = max_len
    res = []
    while cur_end<=len(mathing_str) and cur_start<=cur_end:
        cur_str = mathing_str[cur_start:cur_end]

        if cur_str not in dic:
            cur_end -=1
        else:
            res.append(cur_str)
            cur_start = cur_end
            cur_end = min(len(mathing_str),cur_end+max_len)
        print(cur_start,cur_end,cur_str,res)
    if cur_end!=len(mathing_str)-1:
        print("no matching ")
    else:
        print(res)


dic = ["李","不搭","李不搭","武功","武功盖世","天下","第一","一"]
strs = "李不搭武功盖世天下第一"
max_len = 4
forward_max_mathcing(strs,dic,max_len)