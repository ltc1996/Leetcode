1078. Bigram分词

- 简单

给出第一个词 `first` 和第二个词 `second`，考虑在某些文本 `text` 中可能以 "`first second third`" 形式出现的情况，其中 `second` 紧随 `first` 出现，`third` 紧随 `second` 出现。

对于每种这样的情况，将第三个词 "`third`" 添加到答案中，并返回答案。


- 示例1：
```
输入：text = "alice is a good girl she is a good student", first = "a", second = "good"
输出：["girl","student"]
```

- 示例2：
```
输入：text = "we will we will rock you", first = "we", second = "will"
输出：["we","rock"]
```

**提示**:
1. `1 <= text.length <= 1000`
2. `text` 由一些用空格分隔的单词组成，每个单词都由小写英文字母组成
3. `1 <= first.length, second.length <= 10`
4. `first` 和 `second` 由小写英文字母组成


## 我的解答：
```python
class Solution(object):
    # 用时20ms
    def findOcurrences(self, text, first, second):
        """
        :type text: str
        :type first: str
        :type second: str
        :rtype: List[str]
        """
        t = text.split()
        res = []
        i = 0
        while(i < len(t) - 2):
            if t[i] == first and t[i + 1] == second:
                # 满足则右移两位
                res.append(t[i+2])
                i += 1
            i += 1
        return res
```

## 最快解答：
```python
class Solution(object):
    # 用时12ms
    def findOcurrences(self, text, first, second):
        list1=[]
        word=text.split(' ')
        for i in range(0,len(word)-2):
        # 用for loop
            if word[i]==first and word[i+1]==second:
                list1.append(word[i+2])
        return list1
```
