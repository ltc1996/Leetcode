0712. 两个字符串的最小ASCII删除和

- 中等

给定两个字符串`s1`, `s2`，找到使两个字符串相等所需删除字符的ASCII值的最小和。

- 示例1：
```
输入: s1 = "sea", s2 = "eat"
输出: 231
解释: 在 "sea" 中删除 "s" 并将 "s" 的值(115)加入总和。
在 "eat" 中删除 "t" 并将 116 加入总和。
结束时，两个字符串相等，115 + 116 = 231 就是符合条件的最小和。
```

- 示例2：
```
输入: s1 = "delete", s2 = "leet"
输出: 403
解释: 在 "delete" 中删除 "dee" 字符串变成 "let"，
将 100[d]+101[e]+101[e] 加入总和。在 "leet" 中删除 "e" 将 101[e] 加入总和。
结束时，两个字符串都等于 "let"，结果即为 100+101+101+101 = 403 。
如果改为将两个字符串转换为 "lee" 或 "eet"，我们会得到 433 或 417 的结果，比答案更大。
```

## 注意：
- `0 < s1.length, s2.length <= 1000`
- 所有字符串中的字符ASCII值在`[97, 122]`之间

## 我的解答：
```cpp
class Solution {
public:
    // 用时36ms
    int minimumDeleteSum(string s1, string s2) {
        // 初始化为无穷大，224即可
        vector<vector<int> > dp(1 + s1.size(), vector<int>(1 + s2.size(), 224));
        dp[0][0] = 0;
        // 第一行 / 列为累计删除到该位置的删除和
        for(int i = 0; i < s1.size(); i++)
            dp[i + 1][0] = dp[i][0] + s1[i];
        for(int j = 0; j < s2.size(); j++)
            dp[0][j + 1] = dp[0][j] + s2[j];
        for(int i = 1; i <= s1.size(); i++){
            for(int j = 1; j <= s2.size(); j++){
                // 如果字符相等则各自退一位
                if(s1[i - 1] == s2[j - 1])
                    dp[i][j] = dp[i - 1][j - 1];
                // 不相等则比较删除哪个字符代价更小：当前 + 之前累计
                else
                    dp[i][j] = min(s1[i - 1] + dp[i - 1][j], s2[j - 1] + dp[i][j - 1]);
            }
        }
        return dp[s1.size()][s2.size()];
    }
};
```

## 最快解答：
```cpp
class Solution {
public:
    // 用时12ms
    int minimumDeleteSum(string s1, string s2) {
        int sum = 0;
        for (const auto c : s1)
            sum += c;
        for (const auto c : s2)
            sum += c;
        //cout << "sum is " << sum << endl;
        const int size = s2.size();
        vector<int> dp(size, 0);

        for (const auto c : s1) {
            int last = dp[0];
            if (c == s2[0]) {
                dp[0] = c;
            }
            for (int i = 1; i < size; i++) {
                int backup = last;
                last = dp[i];
                if (c == s2[i]) {
                    dp[i] = backup + c;
                } else if (dp[i - 1] > dp[i]) {                    
                    dp[i] = dp[i - 1];
                }
            }
        }
        return sum - 2 * dp[size - 1];
    }
};
```
