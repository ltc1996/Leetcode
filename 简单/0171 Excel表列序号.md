171. Excel表列序号

- 简单

## 题目描述：
给定一个Excel表格中的列名称，返回其相应的列序号。

例如，
```
   A -> 1
   B -> 2
   C -> 3
   ...
   Z -> 26
   AA -> 27
   AB -> 28
   ...
```

- 示例1：
```
输入: "A"
输出: 1
```

- 示例2：
```
输入: "AB"
输出: 28
```

- 示例3：
```
输入: "AB"
输出: 28
```

## 我的解答：
``` python
class Solution(object):
    # 用时40ms
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        num = [ord(x)-64 for x in s]
        res = 0
        for i in range(len(num)):
            res += num[i]*pow(26, len(num)-i-1)
        return res
```

## 最快解答：
``` python
class Solution(object):
    # 用时24ms
    def titleToNumber(self, s):
        res = 0
        list_str = list(s)
        for i in range(len(list_str)):
            res += 26**(len(list_str)-1-i)*(ord(list_str[i])-64)
        return res
```
