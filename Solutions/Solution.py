from enum import Flag
from functools import reduce
from itertools import count
from operator import imod, le, lshift, setitem
import re
from re import T, match, search
from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def treeNodeToString(root):
    if not root:
        return "[]"
    output = ""
    queue = [root]
    current = 0
    while current != len(queue):
        node = queue[current]
        current = current + 1

        if not node:
            output += "null, "
            continue

        output += str(node.val) + ", "
        queue.append(node.left)
        queue.append(node.right)
    return "[" + output[:-2] + "]"


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def listNodeToString(node):
    if not node:
        return "[]"

    result = ""
    while node:
        result += str(node.val) + ", "
        node = node.next
    return "[" + result[:-2] + "]"


def p(obj):
    if isinstance(obj, TreeNode):
        print(treeNodeToString(obj))
    elif isinstance(obj, ListNode):
        print(listNodeToString(obj))
    elif isinstance(obj, list):
        for line in obj:
            print(line)
    else:
        print(obj)


class Solution(object):
    def minJumps(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        from collections import defaultdict

        n = len(arr)
        if n < 3:
            return n - 1

        d = defaultdict(list)

        for i in range(n):
            d[arr[i]].append(i)     # 统计高度为arr[i]的所有下标
        vis = [0] * n
        vis[-1] = 1         # 是否搜索过
        queue = [(n-1, 0)]  # bfs, 存储[位置下标, 次数]
        print(arr)

        while queue:
            next_quene = []
            for i in range(len(queue)):
                poi, t = queue[i]
                next_t = t + 1
                print('now:', poi, 'is:', arr[poi])
                for j in d[arr[poi]]:
                    # if j == 0:
                    #     return next_t
                    if j != poi and not vis[j]:
                        print('j =', j)
                        if j == 0:
                            return next_t
                        vis[j] = 1
                        # dp[i] = next_t
                        next_quene.append((j, next_t))

                left, right = poi - 1, poi + 1
                if left >= 0 and not vis[left]:
                    print('left', left)
                    if left == 0:
                        return next_t
                    next_quene.append((left, next_t))
                    # dp[left] = next_t
                    vis[left] = 1
                if right < n and not vis[right]:
                    print('right', right)
                    next_quene.append((right, next_t))
                    vis[right] = 1
                    # dp[right] = next_t

            queue = next_quene
            print(queue, vis)
            print()

        # while queue:
        #     cur, temp = queue.pop(0)
        #     for i in d[arr[cur]]:
        #         if i != cur and not vis[i]:
        #             dp[i] = temp + 1
        #             vis[i] = 1
        #             queue.append((i, temp+1))
        #             if not i:
        #                 # i == 0
        #                 return dp[0]

        #     i = cur
        #     # 右边界
        #     if i+1 < n and not vis[i+1]:
        #         dp[i+1] = temp+1
        #         vis[i+1] = 1
        #         queue.append((i+1, temp+1))
        #     # 左边界
        #     if i-1 >= 0 and not vis[i-1]:
        #         dp[i-1] = temp+1
        #         vis[i-1] = 1
        #         queue.append((i-1, temp+1))
        #         if i-1 == 0:
        #             return dp[0]

    def checkIfExist(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        from collections import defaultdict
        d = defaultdict(int)
        # arr = set(arr)
        for i in arr:
            d[i] += 1
        for k in sorted(d.keys()):
            if 2 * k in d:
                if k == 0:
                    if d[0] >= 2:
                        return True
                    continue
                return True
        return False

    def minSteps(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        from collections import Counter
        cs, ct = map(Counter, [s, t])
        # print(cs, ct)
        print(cs - ct, ct - cs)
        return sum((cs - ct).values())

    def maxStudents(self, seats):
        """
        :type seats: List[List[str]]
        :rtype: int
        """
        row = len(seats)
        if not row:
            return 0
        col = len(seats[0])

        res = 0
        dp = [[0] * (1 << col) for _ in range(row)]

        def ok(num):
            # 返回二进制num中是否出现相邻的1, 非法
            last = 0
            while num:
                curr = num & 1
                if curr and last:
                    return False
                last = curr
                num = num >> 1
            return True

        def get_one(num):
            count = 0
            while num:
                count += num & 1
                num = num >> 1
            return count

        def valid(i, pos):
            # 第pos行中的二进制1的对应位置是否为凳子
            for y in range(col):
                if seats[i][y] == '#' and pos & 1:
                    return False
                pos = pos >> 1
            # return True
            # t = 0
            # while pos:
            #     if pos % 2 and seats[i][t] == '#':
            #         return False
            #     pos //= 2
            #     t += 1
            return True

        for i in range(row):
            for j in range(1 << col):
                if not valid(i, j) or not ok(j):
                    continue
                curr_num = get_one(j)
                if i == 0:
                    dp[i][j] = curr_num
                else:
                    for k in range(1 << col):
                        if ok(j | k):
                            dp[i][j] = max(dp[i][j], curr_num + dp[i - 1][k])
                res = max(res, dp[i][j])

        return res

    # def calculate(self, s):
    #     """
    #     :type s: str
    #     :rtype: int
    #     """
    #     from collections import deque
    #     import re
    #     res = 0
    #     stack = []
    #     sign = 1
    #     n = len(s)
    #     index = 0
    #     while index < n:
    #         if s[index] == ' ':
    #             index += 1
    #         elif s[index] == '-':
    #             sign = -1
    #             index += 1
    #         elif s[index] == '+':
    #             sign = 1
    #             index += 1
    #         elif s[index] == '(':
    #             stack += [res, sign]
    #             res = 0
    #             sign = 1
    #             index += 1
    #         elif s[index] == ')':
    #             res = res * stack.pop() + stack.pop()
    #             index += 1
    #         elif s[index].isdigit():
    #             tmp = int(s[index])
    #             while index < n and s[index].isdigit():
    #                 tmp = 10 * tmp + int(s[index])
    #                 index += 1
    #             res += tmp * sign
    #     return res

    def countNegatives(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        count = 0
        col = len(grid[0])
        for i in range(len(grid)):
            for j in range(col):
                if grid[i][j] < 0:
                    count += col - j
                    break
        return count

    def maxEvents(self, events):
        """
        :type events: List[List[int]]
        :rtype: int
        """
        events = sorted(events, key=lambda x: x[0])
        return events

    def isPossible(self, target):
        flag = True
        def change(target):
            last = min(target)
            if last < 1:
                flag = False

            s = sum(target)
            m = max(target)
            if 2 * m < s:
                flag = False
            mi = target.index(m)
            tmp = 2 * m - s
            if tmp < 1:
                flag = False
            target[mi] = tmp

        while target != [1] * len(target) :
            change(target)
            if not flag:
                return False

        return True

    def minDifficulty(self, jobDifficulty, d):
        """
        :type jobDifficulty: List[int]
        :type d: int
        :rtype: int
        """
        # from collections import defaultdict
        n = len(jobDifficulty)          # 任务数
        if n < d:
            return -1
        # # dp[i][j] 在第j天还差i个任务
        # dp = [[float('inf')] * (d + 1) for _ in range(n)]
        # tmp = jobDifficulty[-1]
        # for i in range(n - 1, -1, -1):
        #     tmp = max(tmp, jobDifficulty[i])
        #     dp[i][1] = tmp
        # print(dp)

        # for i in range(2, d + 1):
        #     for j in range(n - i + 1):
        #         tmp = 0
        #         for k in range(j, n - i + 1):
        #             tmp = max(tmp, jobDifficulty[k])
        #             dp[j][i] = min(dp[j][i], tmp + dp[k + 1][i - 1])
        # print(dp)

        # return dp[0][d]
        # dp[i][j] 代表前i天完成j项任务
        dp = [[float('inf')] * n for _ in range(d)]
        pre = 0
        # 第0天完成前j项任务(共n = len(jobDifficulty))
        for j in range(n):
            pre = max(pre, jobDifficulty[j])
            dp[0][j] = pre
        print(dp)
        for i in range(1, d):       # 第i天
            for j in range(i, n):       # 一天至少一个任务, j不小于i
                pre = jobDifficulty[j]  # 第j个任务
                for k in range(j, i - 1, -1):   # 任务i->k分配给第k - 1天
                # for k in range(i, j + 1):     # 任务k->j分配给第k天
                    pre = max(pre, jobDifficulty[k])    # pre刷新k->j区间的最大值
                    # 反向, 因为第j个任务一定在这天完成
                    # pre为k -> j, dp[i-1][j-1]为前一天的i -> k-1
                    dp[i][j] = min(dp[i][j], pre + dp[i - 1][k - 1])
        print(dp)

        return dp[-1][-1]

    def sortByBits(self, arr):
        """
        :type arr: List[int]
        :rtype: List[int]
        """
        arr.sort()
        def number(n):
            count = 0
            while n:
                count += n & 1
                n = n >> 1
            return count
        return sorted(arr, key=lambda x: number(x))

    def numberOfSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        from collections import Counter
        n = len(s)
        dp = [[False] * (1 + n) for _ in range(1 + n)]
        print(dp)
        count = 0
        for i in range(n - 2):
            for j in range(i + 3, 1 + n):
                tmp = s[i: j]
                # print(tmp)
                if dp[i][j - 1]:
                    dp[i][j] = True
                    count += 1
                    # print(tmp)

                elif len(Counter(tmp)) == 3:
                    dp[i][j] = True
                    count += 1
                    # print(tmp)

        return count

    def countOrders(self, n):
        """
        :type n: int
        :rtype: int
        """
        from itertools import combinations
        from functools import reduce
        def fact(n):
            from functools import reduce
            return reduce(lambda x, y: x * y, range(1, 1 + n))
        print(fact(3))
        if n == 1:
            return 1
        else:
            return 2

    def daysBetweenDates(self, date1, date2):
        """
        :type date1: str
        :type date2: str
        :rtype: int
        """
        from datetime import datetime, timedelta
        f = lambda x: [int(i) for i in x.split('-')]
        if date2 > date1:
            date1, date2 = date2, date1
        y1, m1, d1 = f(date1)
        y2, m2, d2 = f(date2)
        # date1, date2 = map(datetime, *map(f, (date1, date2)))
        delta = datetime(y1, m1, d1) - datetime(y2, m2, d2)
        return delta.days

    def largestMultipleOfThree(self, digits):
        """
        :type digits: List[int]
        :rtype: str
        """
        from collections import defaultdict
        d = defaultdict(list)
        digits = sorted(digits, reverse=True)
        for num in digits:
            d[num % 3] += [num]
        zero, one, two = map(lambda x: d.get(x, []), range(3))
        zero_l, one_l, two_l = map(len, [zero, one, two])
        print(zero, one, two)
        print()
        print(zero_l, one_l, two_l)
        left = (1 * one_l + 2 * two_l) % 3
        print('left, ', left)
        f = False
        if left == 0:       # 整除
            if sum(digits):
                f = True
        if left == 1:       # 1
            print(1)
            if one_l:
                one.pop()
                f = True
            elif zero_l >= 3:
                two = []
                f = True
        if left == 2:       # 2
            print(2)
            if two_l:
                two.pop()
                f = True
            elif one_l >= 2:
                one.pop()
                one.pop()
                print(2)
                f = True

        print(d)

        res = zero + one + two
        res = sorted(res, reverse=True)
        print(res)
        if f and res:
            return ''.join(map(str, res))
        else:
            return '0'
        # if zero_l == 0:
        #     if one_l == 0:
        #         if two_l < 3:
        #             return '0'
        #     if two_l == 0:
        #         for i in (0, 1):
        #             if one_l < 3 - i:
        #                 return '0'

    def closestDivisors(self, num):
        res = float('inf')
        ret = (1, num)
        for i in range(1, 2 + int((2 + num)**0.5)):
            left1, left2, right = num // i, 1 + num // i, i
            if left1 * right in (num + 1, num + 2):
                if abs(left1 - right) < res:
                    ret = left1, right
                    res = abs(left1 - right)
            if left2 * right in (num + 1, num + 2):
                if abs(left2 - right) < res:
                    ret = left2, right
                    res = abs(left2 - right)
        return ret

    def orangesRotting(self, grid):
        from collections import deque
        row, col = len(grid), len(grid[0])
        count = 0
        dirs = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        )
        rots = deque()
        for i in range(row):
            for j in range(col):
                if grid[i][j] == 2:
                    rots.append((i, j, 0))
        print(rots)
        while rots:
            # rots_n = []
            # count += 1
            print(count, 'start')
            x, y, time = rots.popleft()
            for i, j in dirs:
                dx = i + x
                dy = j + y
                if 0 <= dx < row and 0 <= dy < col:
                    if grid[dx][dy] == 1:
                        print(dx, dy)
                        grid[dx][dy] = 2
                        rots.append((dx, dy, 1 + time))
            print(count, 'over')
            print(rots)

        for i in range(row):
            for j in range(col):
                if grid[i][j] == 1:
                    return -1
        else:
            return time

    def matrixScore(self, A):
        """
        :type A: List[List[int]]
        :rtype: int
        """
        row, col = map(len, (A, A[0]))
        mark = [False] * col
        print(row, col)

        for i in range(row):
            mark[i] = A[i][0] == 1
        print(mark)

        res = (1 << (col - 1)) * row
        for j in range(1, col):
            tmp = 0
            for i in range(row):
                if A[i][j] and mark[i]:
                    tmp += 1
            tmp = max(row - tmp, tmp)
            res += tmp * (1 << (col - 1 - j))

        return res

    def sortString(self, s):
        """
        :type s: str
        :rtype: str
        """
        from collections import Counter
        c = Counter(s)

        def update(d):
            c = dict()
            for k in d:
                if d[k]:
                    c[k] = d[k]
            return c
        print(c)
        res = ''
        time = 1
        while c:
            k = sorted(c.keys())
            tmp = ''
            for i in k:
                tmp += i
                c[i] -= 1
            if time % 2:
                res += tmp
            else:
                res += tmp[::-1]
            time = 1 - time
            c = update(c)

        return res

    def numOfMinutes(self, n, headID, manager, informTime):
        """
        :type n: int
        :type headID: int
        :type manager: List[int]
        :type informTime: List[int]
        :rtype: int
        """
        from collections import defaultdict
        d = defaultdict(list)
        for i in range(len(manager)):
            d[manager[i]].append(i)

        print(d)
        res = []

        def dfs(i, s):
            print('now', i)
            if informTime[i] == 0:
                res.append(s)
                print('over')
                return
            s += informTime[i]
            for j in d[i]:
                print(j)
                dfs(j, s)

        dfs(headID, 0)
        m = max(res)
        return m

    def frogPosition(self, n, edges, t, target):
        """
        :type n: int
        :type edges: List[List[int]]
        :type t: int
        :type target: int
        :rtype: float
        """
        from collections import defaultdict
        d = defaultdict(list)
        edges = [[x, y] if x < y else [y, x] for x, y in edges]
        print(edges)
        for x, y in edges:
            d[x].append(y)

        print(d)
        self.line = []
        def dfs(i, tmp, time):
            tt = tmp + [i]
            if i not in d or time >= t:
                if i == target:
                    self.line = tt
                # res.append(tt)
                return
            for j in d[i]:
                dfs(j, tt, time + 1)

        dfs(1, [], 0)
        # print(self.line)
        if not self.line:
            return 0

        ret = 1
        for i in self.line:
            ret *= 1 / (len(d[i]) or 1)

        return ret

    def sortedArrayToBST(self, nums) -> TreeNode:
        def dfs(left, right):
            if left > right:
                return
            mid = (left + right) >> 1
            root = TreeNode(nums[mid])
            root.left = dfs(left, mid - 1)
            root.right = dfs(mid + 1, right)
            return root
        r = dfs(0, len(nums) - 1)
        return r

    def luckyNumbers (self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        row, col = len(matrix), len(matrix[0])
        tar = []
        res = []
        for i in range(row):
            m = float('inf')
            t = (i, 0)
            for j in range(col):
                print(m)
                print(matrix[i][j])
                if matrix[i][j] < m:
                    m = matrix[i][j]
                    t = (i, j)
            tar.append(t)
        print(tar)
        for x, y in tar:
            m = matrix[x][y]
            flag = True
            for i in range(row):
                if matrix[i][y] > m:
                    flag = False
                    break
            if flag:
                res.append(m)
        return res

    def maxPerformance(self, n, speed, efficiency, k):
        """
        :type n: int
        :type speed: List[int]
        :type efficiency: List[int]
        :type k: int
        :rtype: int
        """
        from heapq import nlargest
        tar = [[speed[i], efficiency[i]] for i in range(n)]
        tar.sort(key=lambda d: d[1])
        # print(tar)

        res = 0
        poi = 0
        for i in range(n):
            if tar[i][0] * tar[i][1] > res:
                res = tar[i][0] * tar[i][1]
                poi = i
        print(res)
        print(poi)
        eff = tar[poi][1]
        print(eff)
        sp = [tar[i][0] for i in range(poi, n)]
        nl = nlargest(min(k, len(sp)), sp)
        print(nl)
        res = sum(nl) * eff
        return res

    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        res = 0
        row, col = map(len, (grid, grid[0]))
        dirs = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        )
        def dfs(x, y):
            area = 1
            grid[x][y] = 0
            for dir in dirs:
                dx = x + dir[0]
                dy = y + dir[1]
                if 0 <= dx < row and 0 <= dy < col:
                    if grid[dx][dy]:
                        area += dfs(dx, dy)
            return area

        for i in range(row):
            for j in range(col):
                if grid[i][j]:
                    t = dfs(i, j)
                    res = max(res, t)

        return res

    def compressString(self, S):
        """
        :type S: str
        :rtype: str
        """
        import re
        res = ''
        l = 0
        p = re.finditer(r'([a-zA-Z])\1*', S)
        for char in map(lambda x: x.group(), p):
            t = str(len(char))
            l += 1 + len(t)
            res += char[0] + t

        return res * (l < len(S)) or S

        last = S[0]
        res = ''
        count = 0
        tmp = 1
        for i in range(1, len(S)):
            if S[i] == last:
                tmp += 1
            else:
                t = str(tmp)
                res += last + t
                count += 1 + len(t)
                last = S[i]
                tmp = 1
        t = str(tmp)
        res += last + t
        count += 1 + len(t)
        print(count, len(S))
        if count < len(S):
            return res
        else:
            return S

    def longestPalindrome(self, s):
        from collections import Counter
        c = Counter(s)
        flag = False
        for num in c.values():
            if num % 2:
                flag = True
                break
        return sum([num & 0xfffffffe for num in c.values()]) + flag

    def partitionLabels(self, S):
        # from collections import defaultdict
        d = {}     # defaultdict(list)
        for i in range(len(S)):
            if S[i] in d:
                d[S[i]][-1] = i
            else:
                d[S[i]] = [i, i]
        v = sorted(d.values(), key=lambda x: x[0])
        print(v)
        res = []
        s, e = v[0]
        for i in range(1, len(v)):
            if v[i][0] > e:
                # 开始 > 结尾, 说明不再重叠
                res.append(e - s + 1)
                s = v[i][0]
            if v[i][1] > e:
                # 延长结尾位置
                e = v[i][1]
        res.append(e - s + 1)

        return res

    def minDeletionSize(self, A):
        """
        :type A: List[str]
        :rtype: int
        """
        A = [x for x in zip(*A)]
        B = [list(y) != sorted(y) for y in A]
        return sum(B)

    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        p = sorted(people, key=lambda x: (-x[0], x[1]))
        res = []
        for item in p:
            res.insert(item[1], item)
        return res

    def minSwapsCouples(self, row):
        count = 0
        for i in range(0, len(row), 2):
            r = row[i] ^ 1
            t = row.index(r)
            if t - i > 1:
                count += 1
                row[t], row[i + 1] = row[i + 1], row[t]
        return count

    def findTheDistanceValue(self, arr1, arr2, d):
        """
        :type arr1: List[int]
        :type arr2: List[int]
        :type d: int
        :rtype: int
        """
        count = 0
        for i in arr1:
            for j in arr2:
                if abs(i - j) <= d:
                    break
            else:
                count += 1
        return count

    def maxNumberOfFamilies(self, n, reservedSeats):
        """
        :type n: int
        :type reservedSeats: List[List[int]]
        :rtype: int
        """
        from collections import defaultdict
        d = defaultdict(set)
        for x, y in reservedSeats:
            d[x].add(y)
        l, m, r = set((2, 3)), set((4, 5, 6, 7)), set((8, 9))
        print(d)
        count = 0
        for i in range(1, 1 + n):
            print('now', i)
            if i not in d or not (l | m | r) & d[i]:
                count += 2
                print(1, count)
                continue
            t = d[i]
            if l & t or r & t:
                # print(1)
                # print(l, r, m, t)
                if not set((6, 7, 8, 9)) & t:
                    count += 1
                    continue
                if not set((2, 3, 4, 5)) & t:
                    count += 1
                    continue
                if not m & t:
                    count += 1
                    continue
            else:
                if not set((4, 5)) & t:
                    count += 1
                if not set((6, 7)) & t:
                    count += 1
            print(count)
        return count

    def getKth(self, lo, hi, k):
        """
        :type lo: int
        :type hi: int
        :type k: int
        :rtype: int
        """
        def change(n):
            if n == 1:
                return 0
            if n % 2:
                return 1 + change(3 * n + 1)
            else:
                return 1 + change(n // 2)

        tar = [i for i in range(lo, 1 + hi)]
        tar.insert
        tar.sort(key=lambda x: change(x))
        return tar[k - 1]

    def sumFourDivisors(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def helper(n):
            count = 0
            s = 0
            # print('now', n)
            if n == 1:
                return 0
            for i in range(1, 1 + int(n**0.5)):
                if n % i == 0:
                    # print('has', i)
                    if i**2 == n:
                        return 0
                    count += 1
                    if count > 2:
                        return 0
                    s += i + n // i
            # print(n, count)
            if count == 2:
                return s
            else:
                return 0
        print([helper(i) for i in nums])
        return sum([helper(i) for i in nums])

    def longestPrefix(self, s):
        n = len(s)
        if n == 1:
            return ''
        for i in range(n - 1, 0, -1):
            if s[:i] == s[n - i:]:
                return s[:i]
        else:
            return ''

    def hasValidPath(self, grid):
        pipes = {       # 管道流出的方向
            1: ('左', '右'),
            2: ('上', '下'),
            3: ('左', '下'),
            4: ('右', '下'),
            5: ('左', '上'),
            6: ('右', '上'),
        }
        rev = {         # 反方向
            '左': '右',
            '右': '左',
            '下': '上',
            '上': '下',
        }
        dirs = {        # 方向与坐标的转换
            '左': (0, -1),
            '右': (0, 1),
            '下': (1, 0),
            '上': (-1, 0),
        }
        row, col = map(len, (grid, grid[0]))
        if row == col == 1:
            return True
        self.res = False

        def dfs(x, y, d):
            print('now', x, y)
            if x == row - 1 and y == col - 1:
                self.res = True
                print('end')
                return
            if x < 0 or x >= row or y < 0 or y >= col:
                self.res = False
                print('overflow')
                return
            curr = grid[x][y]       # 当前管道型号
            curr_dir = pipes[curr]  # 当前管道两个流出方向
            print('@ pipe', curr)
            _from = rev[d]      # 离开向下, 迎接向上
            if _from not in curr_dir:   # 与来向不匹配
                print('wrong pipe')
                return False
            for d in curr_dir:
                if d != _from:  # 向管道另外一头流出
                    _to = d
            dx = dirs[_to][0] + x
            dy = dirs[_to][1] + y
            print('from', _from, x, y, 'to', _to, dx, dy)
            dfs(dx, dy, _to)

        first = grid[0][0]
        if first in (4, 5):
            return False
        elif first in (1, 6):
            dfs(0, 0, '右')
        else:
            dfs(0, 0, '下')
        return self.res

    def hasValidPath1(self, grid):
        row, col = len(grid), len(grid[0])
        if row == col == 1:
            return True
        U = -1 + 0j
        D = 1 + 0j
        L = 0 - 1j
        R = 0 + 1j
        pipes = {
            1: (L, R),
            2: (U, D),
            3: (L, D),
            4: (R, D),
            5: (L, U),
            6: (R, U),
        }

        def dfs(cmpx, d):
            print('in')
            x = int(cmpx.real)
            y = int(cmpx.imag)
            if x < 0 or x >= row or y < 0 or y >= col:
                return cmpx
            curr = grid[x][y]
            print(x, y, pipes[curr])
            if -d not in pipes[curr]:
                print(2)
                return False
            if x == row - 1 and y == col - 1:
                return cmpx
            for dir in pipes[curr]:
                if dir != -d:
                    print('out as', cmpx + dir, dir)
                    return dfs(cmpx + dir, dir)
            # print('this', cmpx)
            # return cmpx

        first = grid[0][0]
        if first == 5:
            return False
        elif first == 4:
            print(dfs(0 + 0j, D))
            res = []
            # res = (dfs(0 + 0j, R), dfs(0 + 0j, D))
        elif first in (1, 6):
            res = dfs(0 + 0j, R),
        else:
            res = dfs(0 + 0j, D),
        print('end @', res)
        return (row - 1) + (col - 1) * 1j in res

    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        intervals.sort(key=lambda x: x[1])
        # print(intervals)
        e, count = float('-inf'), 0
        for x, y in intervals:
            if x >= e:
                count += 1
                e = y
            # count += 1
            # print(e, count)

        return len(intervals) - count

    def canJump(self, nums):
        n = len(nums)
        next_jump = 0
        for i in range(n):
            print(i)
            if i > next_jump:
                return False
            next_jump = max(next_jump, i + nums[i])
            print(i, next_jump)
            if next_jump >= n:
                return True
        return True

    def main(self, ran):
        # from collections import Counter
        # cs, ct = map(Counter, (s, t))
        # if cs != ct:
        #     return -1
        # i = j = count = 0
        # n = len(s)

        # while i < n and j < n:
        #     if s[i] != t[j]:
        #         count += 1
        #         j += 1
        #     i += 1

        # return count


        from collections import defaultdict
        d = defaultdict(int)
        n = len(ran)
        tar = list(zip(*ran))
        tar = list([j for j in range(i[0], 1 + i[1])] for i in tar)
        print('范围:', tar)

        res = []
        def helper(i, lst):
            if i == n:
                res.append(lst)
                return
            for x in tar[i]:
                helper(i + 1, lst + [x])
            
        helper(0, [])
        print('所有可能取值:', res)
        for lst in res:
            d[min(lst)] += 1
        # print(d)

        a, b = 0, 0
        for k, v in d.items():
            a += k * v
            b += v
            print('最小为{}有{}种情况'.format(k, v))

        return round(a / b, 6)

    def minimumLengthEncoding(self, words):
        from collections import defaultdict, deque
        from functools import reduce
        words = list(set(words))

        Trie = lambda: defaultdict(Trie)
        trie = Trie()

        nodes = [reduce(dict.__getitem__, word[::-1], trie)  \
                 for word in words]

        return sum(len(word) + 1                             \
                   for i, word in enumerate(words)           \
                   if len(nodes[i]) == 0)
        
    def findLucky(self, arr):
        from collections import defaultdict
        d = defaultdict(int)
        res = -1
        for n in arr:
            d[n] += 1
        for k, v in d.items():
            if v == k:
                res = max(res, k)
        return res

    def numTeams(self, rating):
        from itertools import combinations
        n = len(rating)
        if n <= 2:
            return 0
        count = 0
        c = combinations(range(n), 3)
        for i, j, k in c:
            if rating[i] < rating[j] < rating[k] or rating[i] > rating[j] > rating[k]:
                count += 1
        return count

    def maxDistance(self, grid):
        row, col = len(grid), len(grid[0])
        count = 0
        dirs = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1)
        )

        islands = []#deque([])
        for i in range(row):
            for j in range(col):
                if grid[i][j] == 1:
                    islands.append((i, j))

        if not islands or len(islands) == row * col:
            return -1

        while islands:
            # print(count, islands)
            count += 1
            next = set()
            for x, y in islands:
                grid[x][y] = 1
                for dir in dirs:
                    dx = x + dir[0]
                    dy = y + dir[1]
                    if 0 <= dx < row and 0 <= dy < col:
                        if grid[dx][dy] == 0 and (dx, dy) not in islands:
                            next.add((dx, dy))
            islands = list(next)
        return count - 1

    def findMinArrowShots(self, points):
        points.sort(key=lambda x: x[1])
        start, end = points[0]
        count = 1
        for left, right in points[1:]:
            if left > end: 
                count += 1
                end = right
            # else:
            #     end = min(end, right)

        return count

    def findDiagonalOrder(self, matrix):
        if not matrix:
            return []
        row, col = len(matrix), len(matrix[0])
        res = []
        diag = row + col - 1
        # print(diag)
        i = j = 0
        index = 0
        for i in range(row):
            for j in range(1 + index):
                print(j, index - j)
            index += 1
            # print(index)
        print()
        for j in range(1, col):
            for i in range(row - 1, 1 + index):
                print(i, index - i)
            index += 1  

        return 
    # TODU
    def leastInterval(self, tasks, n):
        from collections import Counter
        c = Counter(tasks)
        print(c)

        return tasks

    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """
        people.sort()
        count = 0
        i, j = 0, len(people) - 1
        while i <= j:
            if people[i] + people[i] <= limit:
                i += 1
            count += 1
            j -= 1

        return count

    def countLargestGroup(self, n):
        """
        :type n: int
        :rtype: int
        """
        from collections import defaultdict
        def cal(n):
            count = 0
            for i in str(n):
                count += int(i)
            return count
        d = defaultdict(int)
        for i in range(1, n + 1):
            d[cal(i)] += 1
        res = 0
        max_d = max(d.values())
        for k, v in d.items():
            if v == max_d:
                res += 1

        return res

    def canConstruct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: bool
        """
        from collections import Counter
        c = Counter(s)
        odd = 0
        for v in c.values():
            odd += v & 1
        return odd <= k and len(s) >= k

    def checkOverlap(self, radius, x_center, y_center, x1, y1, x2, y2):
        """
        :type radius: int
        :type x_center: int
        :type y_center: int
        :type x1: int
        :type y1: int
        :type x2: int
        :type y2: int
        :rtype: bool
        """
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        x1 -= mid_x
        x2 -= mid_x
        x_center -= mid_x
        y1 -= mid_y
        y2 -= mid_y
        y_center -= mid_y
        if x_center < 0:
            x_center *= -1
        if y_center < 0:
            y_center *= -1
        print(x_center, y_center, x1, y1, x2, y2)

        # m1 = max(x_center - x1, 0) ** 2 + max(0, y_center - y1) ** 2
        # m2 = max(x_center - x2, 0) ** 2 + max(0, y_center - y1) ** 2
        # m3 = max(x_center - x1, 0) ** 2 + max(0, y_center - y2) ** 2
        m4 = max(x_center - x2, 0) ** 2 + max(0, y_center - y2) ** 2
        m = (x_center - x2) ** 2 + (y_center - y2) ** 2
        # m = min(m1, m2, m3, m4)
        if m4 <= radius ** 2:
            return True
        else:
            return False
    def maxSatisfaction(self, satisfaction):
        """
        :type satisfaction: List[int]
        :rtype: int
        """
        R = 0
        t = 0
        satisfaction.sort(reverse=True)
        for s in satisfaction:
            t += s
            N = R + t
            if N >= R:
                R = N
        return R

    def minSubsequence(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        s = 0
        for num in nums:
            s += num
        res = []
        tmp_s = 0
        for num in sorted(nums, reverse=True):
            if 2 * tmp_s <= s:
                tmp_s += num
                res.append(num)
        return res

    def numSteps(self, s):
        count = 0
        s = int(s, 2)
        while s > 1:
            if s & 1:
                s += 1
            else:
                s = s >> 1
            count += 1
            # print(count, s)
        return count

    def longestDiverseString(self, a, b, c):
        """
        :type a: int
        :type b: int
        :type c: int
        :rtype: str
        """
        s = ''
        t = [a, b, c]
        tt = ('a', 'b', 'c')
        while any(t):
            vis = set()
            m = t.index(max(t[0], t[1], t[2]))
            if t[0]:
                if not s or s[0] != 'a':
                    s = 'a' * min(t[0], 2) + s
                    t[0] -= min(t[0], 2)
            if t[0]:
                if not s or s[-1] != 'a':
                    s += 'a' * min(t[0], 2)
                    t[0] -= min(t[0], 2)
            if t[m]:
                if not s or s[0] != tt[m]:
                    s = tt[m] * min(t[m], 2) + s
                    t[m] -= min(t[m], 2)
            if t[m]:
                if not s or s[-1] != tt[m]:
                    s += tt[m] * min(t[m], 2)
                    t[m] -= min(t[m], 2)
            if t[1]:
                if not s or s[0] != 'b':
                    s = 'b' * min(t[1], 2) + s
                    t[1] -= min(t[1], 2)
            if t[1]:
                if not s or s[-1] != 'b':
                    s += 'b' * min(t[1], 2)
                    t[1] -= min(t[1], 2)
            if t[2]:
                if not s or s[0] != 'c':
                    s = 'c' * min(t[2], 2) + s
                    t[2] -= min(t[2], 2)
            if t[2]:
                if not s or s[-1] != 'c':
                    s += 'c' * min(t[2], 2)
                    t[2] -= min(t[2], 2)
        return s

    def strWithout3a3b(self, A, B):
        s = ''
        while A or B:
            if not s or s[0] != 'a':
                t = min(A, 2)
                s = 'a' * t + s
                A -= t
            if not s or s[-1] != 'a':
                t = min(A, 2)
                s = s + 'a' * t
                A -= t
            if not s or s[0] != 'b':
                t = min(B, 2)
                s = 'b' * t + s
                B -= t
            if not s or s[-1] != 'b':
                t = min(B, 2)
                s = s + 'b' * t
                B -= t
        return s

    def numberOfArithmeticSlices(self, A):
        from operator import __add__
        n = len(A)
        count = 2
        ret = 0
        for i in range(2, n):
            if A[i] - A[i - 1] == A[i - 1] - A[i - 2]:
                count += 1
            else:
                ret += reduce(__add__, range(1, count - 1))
                count = 2
        
        ret += reduce(__add__, range(1, count - 1))
        
        # res.append(count)
        # print(res)
        # ret = reduce(__add__, [reduce(__add__, range(1, n - 1)) for n in res])
        # for n in res:
            # for i in range(2, n):
            #     ret += n - i
            # ret += reduce(lambda x, y: x + y, range(1, n - 1))

        return ret

    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n1, n2 = map(len, (word1, word2))
        if not n1 or not n2:
            return n1 or n2
        dp = [[0] * n1 for _ in range(n2)]

        dp[0][0] = int(word1 != word2)
        for i in range(1, n1):
            dp[0][i] = 1 + dp[0][i - 1]
        for j in range(1, n2):
            dp[j][0] = 1 + dp[j - 1][0]

        for i in range(1, n2):
            tar = word2[i]
            for j in range(1, n1):
                if tar == word1[j]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
        for i in dp:
            print(i)
        return dp[-1][-1]

    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [1] * n
        dp[0] = dp[1] = 0
        for i in range(2, int(n ** 0.5) + 1):
            if not dp[i]:
                continue
            dp[i * i: n: i] = [0] * len(dp[i * i: n: i])
        print(dp)
        return sum(dp)

    def isValidSudoku(self, board):
        nine = range(9)
        three = range(3)
        def judge(line):
            alt = set()
            for n in line:
                if n != '.':
                    if n not in alt:
                        alt.add(n)
                    else:
                        return False
            else:
                return True
        for i in nine:
            row = board[i]
            col = [board[j][i] for j in nine]
            box = [board[i // 3 * 3 + x][i % 3 * 3 + y] for x in three for y in three]
            # print(col, row, box)
            if not all(map(judge, (row, col, box))):
                return False
        else:
            return True

    def generateParenthesis(self, n):
        res = []
        def helper(i, j, s):
            if i > n or j > n or j > i:
                return
            if i == j == n:
                res.append(s)
                return
            helper(i + 1, j, s + '(')
            helper(i, j + 1, s + ')')
        helper(0, 0, '')
        return res

    def movingCount(self, m, n, k):
        """
        :type m: int
        :type n: int
        :type k: int0
        :rtype: int
        """
        count = 0
        f = lambda x: x // 10 + x % 10
        dirs = (
            (0, 1),
            (1, 0),
        )
        dp = [[0] * n for _ in range(m)]
        def dfs(i, j):
            if f(i) + f(j) > k:
                return
            dp[i][j] = 1
            for dir in dirs:
                x = i + dir[0]
                y = j + dir[1]
                if 0 <= x < m and 0 <= y < n and not dp[x][y]:
                    # dp[x][y] = 1
                    dfs(x, y)
    
        dfs(0, 0)

        for i in dp:
            print(i)
            count += sum(i)

        return count

    def spiralOrder(self, matrix):
        res = []
        up, down =  0, len(matrix)
        if down <= 1:
            return matrix or matrix[0]
        left, right = 0, len(matrix[0])
        while left <= right and up <= down:
            tmp = []
            for i in range(left, right):
                tmp.append(matrix[up][i])
            up += 1
            # print(1, tmp)
            if tmp:
                res += tmp
            else:
                break

            tmp = []
            for j in range(up, down):
                tmp.append(matrix[j][right - 1])
            right -= 1
            # print(2, tmp)
            if tmp:
                res += tmp
            else:
                break            

            tmp = []
            for i in range(left, right):
                tmp.append(matrix[down - 1][i])
            down -= 1
            # print(3, tmp)
            if tmp:
                res += tmp[::-1]
            else:
                break

            tmp = []
            for j in range(up, down):
                tmp.append(matrix[j][left])
            left += 1
            # print(4, tmp)
            if tmp:
                res += tmp[::-1]
            else:
                break
        print(res[0])
        return 

    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        import heapq
        heapq.heapify(nums)
        while len(nums) > k:
            heapq.heappop(nums)
        return nums[0]

    def removeDuplicates(self, nums):
        index = 0
        for i in range(len(nums)):
            if index < 2 or nums[i] != nums[index - 2]:
                nums[index] = nums[i]
                index += 1
        return index

    def checkPossibility(self, nums):
        flag = True
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                if flag:
                    nums[i] = nums[i + 1]
                    flag = False
                else:
                    return False
        return True

    def groupAnagrams(self, strs):
        from collections import Counter, defaultdict
        d = defaultdict(list)
        def helper(string):
            c = Counter(string)
            res = ''
            for char in sorted(c):
                res += char + str(c[char])
            return res

        for s in strs:
            d[helper(s)].append(s)

        return d.values()

    def increasingTriplet(self, nums):
        n = len(nums)
        if n < 3:
            return False
        dp = [1] * n
        for i in range(n - 1):
            for j in range(i, n):
                if nums[j] > nums[i]:
                    if dp[i] >= 2:
                        return True
                    dp[j] = max(dp[j], 1 + dp[i + 1])
        return False

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        row, col = map(len, (board, board[0]))
        self.flag = False
        dirs = (
            (0, 1),
            (0, -1),
            (-1, 0),
            (1, 0),
        )
        def dfs(x, y, n, vis):
            if n > len(word) - 1:
                return 
            if word[n] != board[x][y]:
                return
            if (x, y) in vis:
                return
            if vis[x][y]:
                return
            vis[x][y] = True
            if n == len(word) - 1:
                self.flag = True
                return
            # print('from', x, y, board[x][y], word[n], vis)
            for dir in dirs:
                dx = x + dir[0]
                dy = y + dir[1]
                if 0 <= dx < row and 0 <= dy < col:
                    # print('to', dx, dy)
                    dfs(dx, dy, n + 1, vis)
            vis[x][y] = False

        vis = [[False] * col for _ in range(row)]
        for i in range(row):
            for j in range(col):
                if board[i][j] == word[0]:
                    dfs(i, j, 0, vis)
                    if self.flag:
                        return True
        return False

    def maxProfit(self, prices):
        res = 0
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            res += max(diff, 0)
        return res

    def evalRPN(self, tokens):
        from operator import __add__, __sub__, __mul__, __truediv__ 
        res = 0
        operators = {
            '+': __add__,
            '-': __sub__,
            '*': __mul__,
            '/': __truediv__,
        }
        nums = []
        for char in tokens:
            if char in operators:
                last = (int(nums.pop()), int(nums.pop()))[::-1]
                res = int(operators[char](*last))
                nums.append(str(res))
            else:
                nums.append(char)
            # print(nums)
        return res

    def stringMatching(self, words):
        res = set()
        n = len(words)
        if n <= 1:
            return []
        words.sort(key=lambda x: len(x))
        # print(words)
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                # print(words[i], words[j])
                if words[i] in words[j]:
                    res.add(words[i])
        return res

    def processQueries(self, queries, m):
        """
        :type queries: List[int]
        :type m: int
        :rtype: List[int]
        """
        nums = [i + 1 for i in range(m)]
        res = []
        for i in queries:
            index = nums.index(i)
            nums = [i] + nums[:index] + nums[index + 1:]
            res.append(index)
            print(nums)
        return res
    
    def entityParser(self, text):
        import re
        d = {
            '&quot;': '"',
            '&apos;': '\'',
            '&amp;': '&',
            '&gt;': '>',
            '&lt;': '<',
            '&frasl;': '/',
        }
        for k in d:
            text = re.sub(k, d[k], text)
        return text

    def numOfWays(self, n):
        from itertools import combinations, permutations
        x, y = 6, 6
        for _ in range(n - 1):
            X, Y = x, y
            x = X * 3 + Y * 2
            y = X * 2 + y * 2
        return (x + y) % 1000000007
    
    def calculate(self, s):
        res = 0
        stack = []
        sign = 1
        num = 0
        for i in s:
            if i.isdigit():
                num = 10 * num + int(i)
            elif i == '+':
                res += num * sign
                print(res)
                num = 0
                sign = 1
            elif i == '-':
                print(res)
                res += num * sign
                num = 0
                sign = -1
            elif i == '(':
                stack.append(res)
                stack.append(sign)
                sign = 1
                res = 0
                print(stack)
            elif i == ')':
                res += num * sign
                num = 0
                res = stack.pop() * res + stack.pop()
                print(stack)
        res += num * sign
        return res

    def lengthOfLongestSubstring(self, s):
        d = {}
        n = len(s)
        i, res = 0, 1
        for j in range(n):
            if s[j] in d:
                i = max(i, d[s[j]])
            res = max(res, j - i + 1)
            d[s[j]] = j + 1
        return res

    def maximumSwap(self, num):
        num_s = str(num)
        nums = [i for i in num_s]
        s = sorted(nums, reverse=True)
        n = len(nums)
        if nums == s:
            return num
        # print(nums, s)
        for i in range(n):
            if nums[i] < s[i]:
                tmp = num_s.rindex(s[i])
                nums[i], nums[tmp] = nums[tmp], nums[i]
                break
        return int(''.join(nums))

    def minCount(self, coins):
        res = 0

    def numWays(self, n, relation, k):
        """
        :type n: int
        :type relation: List[List[int]]
        :type k: int
        :rtype: int
        """
        from collections import defaultdict
        d = defaultdict(list)
        for i, j in relation:
            d[i].append(j)
        # print(d)
        self.count = 0
        # relation_n = len(relation)
        def dfs(start, lst, l):
            # print(lst)
            if l >= k:
                if lst[-1] == n - 1:
                    self.count += 1
                return
            for i in d[start]:
                dfs(i, lst + [i], l + 1)

        dfs(0, [0], 0)

        return self.count

    def getTriggerTime(self, increase, requirements):
        """
        :type increase: List[List[int]]
        :type requirements: List[List[int]]
        :rtype: List[int]
        """
        res = []
        n = len(increase) + 1
        increase = [[0, 0, 0]] + increase
        for i in range(1, n):
            for j in range(3):
                increase[i][j] += increase[i - 1][j]
        # print(increase)
        def helper(l1, l2):
            for i in range(3):
                if l1[i] < l2[i]:
                    return False
            return True

        for curr in requirements:
            print(curr)
            for i in range(n):
                if helper(increase[i], curr):
                    print('y', increase[i], curr)
                    res.append(i)
                    break
            else:
                print('n')
                res.append(-1)
        print(res)
        return res
    

    def minJump(self, jump):
        n = len(jump)
        dp = [float('inf')] * n
        dp[0] = 0
        # print(dp)
        mark = []
        for i in range(n):
            max_jump = i + jump[i]
            if max_jump >= n:
                mark.append(i)
                return 1 + dp[i]
            dp[max_jump] = min(dp[max_jump], 1 + dp[i])
            for j in mark:
                dp[j] = min(dp[j], 1 + dp[i])
        print(dp)
        # return 1 + min(dp[i] for i in mark)

    def minStartValue(self, nums):
        n = len(nums)
        m = 0
        s = 0
        for i in range(n):
            s += nums[i]
            print(s, m)
            if s < 1:
                print(1 - s)
                m += 1 - s
                s = 1
            print('now', m, s)
        return m or m + 1

    def getHappyString(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        limit = 3 * 2 ** (n - 1)
        print(limit)
        if k > limit:
            return ''
        res = []
        def helper(lst, l):
            if l == n:
                res.append(lst)
                return
            for i in ('a', 'b', 'c'):
                if not lst or lst[-1] != i:
                    helper(lst + i, l + 1)
        helper('', 0)   
        print(res)
        return sorted(res)[k - 1]

    def findMinFibonacciNumbers(self, k):
        data = [1,1,2,3,5,8,13,21,34,55 , 89 ,144,233,377, 610, 987, 1597, 2584 , 4181, 6765, 10946, 17711,28657, 46368 ,75025,121393,
196418, 317811, 514229, 832040, 1346269 , 2178309, 3524578, 5702887 , 9227465, 14930352, 24157817, 39088169,
63245986 , 102334155 , 165580141, 267914296, 433494437 , 701408733, 1134903170, 1836311903][::-1]
        n = len(data)
        r = []
        def helper(start, s):
            if s == 0:
                return 0
            # print('sssssss', start)
            for i in range(start, n):
                # print(i, data[i])
                if data[i] <= s:
                    r.append(data[i])
                    print(data[i])
                    return 1 + helper(i, s - data[i])
        res = helper(0, k)
        print(r, sum(r))
        return res

    def numberOfArrays(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        res = 0
        n = len(s)
        t = len(str(k))
        # print(n, t)
        # if t >= n:
        #     return int(k >= int(s))
        for i in range(n):
            # print(i)
            try:
                left = int(s[:i])
                right = int(s[i:])
                print(left, right)
                if left <= k and right <= k:
                    res += 1
            except Exception as e:
                print(e)
        print(res)
        return res % (10**9 + 7)

    def reformat(self, s):
        nums = []
        chars = []
        for char in s:
            if char.isdigit():
                nums.append(char)
            else:
                chars.append(char)
        print(nums, chars)
        res = ''
        n1, n2 = map(len, (nums, chars))
        if abs(n1 - n2) > 1:
            return res
        else:
            if n1 > n2:
                res += nums.pop()
            while nums or chars:
                if chars:
                    res += chars.pop()
                if nums:
                    res += nums.pop()
        return res

    def displayTable(self, orders):
        """
        :type orders: List[List[str]]
        :rtype: List[List[str]]
        """
        from collections import defaultdict
        data = {}
        a = set()
        for _, num, cuisine in orders:
            # print(num, cuisine)
            a.add(cuisine)
            n = int(num)
            if n not in data:
                data[n] = defaultdict(int)
            data[n][cuisine] += 1
        # print(data)
        a = sorted(list(a))
        res = []
        res.append(['Table'] + a)
        for table in sorted(data.keys()):
            # print(table)
            num = str(table)
            tmp = [num]
            for s in a:
                tmp.append(str(data[table][s]))
            # print(tmp)
            res.append(tmp)
        # print(res)
        return res

    def minNumberOfFrogs(self, croakOfFrogs):
        """
        :type croakOfFrogs: str
        :rtype: int
        """
        frog = {
            'c': 0,
            'r': 0,
            'o': 0,
            'a': 0,
            'k': 0,
        }
        res = 0        
        for croakOfFrog in croakOfFrogs:
            frog[croakOfFrog] += 1
            # print(frog)
            if frog['c'] >=  frog['r'] >= frog['o'] >= frog['a'] >= frog['k']:
                if frog['c'] >=  frog['r'] >= frog['o'] >= frog['a'] >= frog['k'] >= 1:
                    res = max(res, frog['c'],  frog['r'], frog['o'], frog['a'], frog['k'])
                    for k in frog:
                        frog[k] -= 1
            else:
                return -1
        if len(set(frog.values())) != 1:
            return -1
        return res or -1

    def numOfArrays(self, n, m, k):
        """
        :type n: int
        :type m: int
        :type k: int
        :rtype: int
        """
        res = 1

    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        n = len(num)
        if n <= k:
            return '0'
        res = ''

    def reorganizeString(self, S):
        if not S:
            return ''
        d = {}
        for s in S:
            if s in d:
                d[s] += 1
            else:
                d[s] = 1
        # print(d)
        v = d.values()
        m = max(v)
        l = len(S)
        if l < 2 * m:
            return ''
        res = ''
        c = sorted(d.items(), key=lambda x: x[1], reverse=True)
        c = list(map(list, c))
        while l:
            for i in c:
                if i[1]:
                    res += i[0]
                    i[1] -= 1
                    l -= 1
        return res

    def translateNum(self, num):
        num = str(num)
        n = len(num)
        if n <= 1:
            return 1
        dp = [1] * n
        # print(int(num[:1]))
        print('[' + ', '.join(num) +']')
        for i in range(1, n):
            dp[i] = dp[i - 1]
            # print(num[i - 1], num[i])
            if int(num[i - 1]) <= 2 and int(num[i]) <= 5:
                dp[i] += dp[i - 2]
                print(dp)
        print('\n', dp)
        return dp[-1]

    def numberOfSubarrays(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        from collections import defaultdict
        n = len(nums)
        for i in range(n):
            nums[i] &= 1
        print(nums)
        d = defaultdict(int)
        s = res = 0
        d[0] = 1
        for i in range(1, 1 + n):
            s += nums[i - 1]
            print(s)
            res += d[s - k]
            d[s] += 1
        return res

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        res = []
        left, right = 0, len(nums) - 1
        mid = 0
        while left < right:
            mid = (left + right) >> 1
            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1
        if not nums or nums[left] != target:
            return [-1, -1]
        right = l = r = left
        while left >= 0 and nums[left] == target:
            l = left
            left -= 1
        res.append(l)
        while right < len(nums) and nums[right] == target:
            r = right
            right += 1
        res.append(r)

        return res

    def waysToChange(self, n):
        dp = [1] * (1 + n)
        for i in (5, 10, 25):
            for j in range(i, 1 + n):
                dp[j] += dp[j - i]
                dp[j] %= 1000000007
                print(dp)
            print()
        # print(dp)
        return dp[-1]

    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int        0
        :type n: int        1
        :rtype: int
        """
        res = 0
        if m >= n:
            f = lambda x: (1 + x.count('1')) / (1 + x.count('0'))#, x.count('0'))
        else:
            f = lambda x: (1 + x.count('0')) / (1 + x.count('1'))#, x.count('1'))
        strs.sort(key=lambda x: (f(x), len(x)))
        print(strs)
        for string in strs:
            # print(f(string))
            zero = string.count('0')
            one = string.count('1')
            # print(string, zero, one)
            if zero <= m and one <= n:
                res += 1
                m -= zero
                n -= one
        return res

    def maxNumberOfFamilies(self, n, reservedSeats):
        """
        :type n: int
        :type reservedSeats: List[List[int]]
        :rtype: int
        """
        reservedSeats.sort(key=lambda x: x[0])
        print(reservedSeats)
        res = 0
        row_last = -1
        count = 0
        row = reservedSeats[0][0]
        while row <= n:
            while row == row_last:
                pass
            else:
                row_last = row
                count += 1
                dp = [0] * 10

        for row, col in reservedSeats:
            if row_last != row:
                row_last = row
                count += 1
                dp = [0] * 10
            dp[col - 1] = 1
            s = []
            for i in (1, 3, 5):
                s.append(sum(dp[i: i + 4]) == 0)
            print(s)
            # if s[1]:
            res += max(s[1], s[0] + s[2])
            # else:
            #     res += s[0] + s[2]
        return res + 2 * (n - count)

    def longestValidParentheses(self, s):
        def helper(s, par):
            res = left = right = 0
            for i in s:
                if i == par:
                    left += 1
                else:
                    right += 1
                    if left == right:
                        res = max(res, 2 * min(left, right))
                    if left < right:
                        left = right = 0
            return res
        res = min(helper(s, '('), helper(s[::-1], ')'))
        return res
    
    def splitArray(self, s):
        from math import gcd
        min_s = list(range(len(s)))
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                if s[i] == s[j] and (i - j < 2 or dp[j + 1][i - 1]):
                    dp[j][i] = True
                    # 说明不用分割
                    if j == 0:
                        min_s[i] = 0
                    else:
                        min_s[i] = min(min_s[i], min_s[j - 1] + 1)
        return min_s[-1] + 1

    
    def minTime(self, time, m):
        """
        :type time: List[int]
        :type m: int
        :rtype: int
        """
        n = len(time)
        if n <= m:
            return 0
        dp = [[float('inf') * (1 + n) for _ in range(m + 1)]]
        for i in range(1, m + 1):
            for j in range(i, n + 1):
                for k in range(i - 1, j):
                    s = sum(time[k:j]) - max(time[k:j])
                    # print(s)
                    dp[i][j] = min(dp[i][j], dp[i-1][k] + s)
        print(dp)
        

    def maxScore(self, s):
        res = 0
        for i in range(1, len(s)):
            # print(s[i:], s[:i])
            num = s[:i].count('0') + s[i:].count('1')
            # print(num)
            res = max(res, num)
        return res

    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        n = len(cardPoints)
        if n <= k:
            return sum(cardPoints)
        s = 0
        for i in range(0, n - k):
            s += cardPoints[i]
        print(s)
        m = s
        for i in range(n - k, n):
            print(i, i - (n - k), cardPoints[i], cardPoints[i - (n - k)])
            t = s + cardPoints[i] - cardPoints[i - (n - k)]
            print(t)
            m = min(m, t)
            s = t
        return sum(cardPoints) - m

    def findDiagonalOrder(self, matrix):
        from collections import deque
        m, n, r = len(matrix), len(matrix) and len(matrix[0]), []
        max_row = max(len(i) for i in matrix)
        # print(max_row)
        for i in matrix:
            l = len(i)
            if l < max_row:
                i += [0] * (max_row - l)
        # print(matrix)
        n = max_row
        for l in range(m + n - 1):
            temp = []
            for i in range(max(0, l + 1 - n), min(l + 1, m)):
                t = matrix[i][l - i]
                if t:
                    temp.append(t)
            # [matrix[i][l - i] for i in range(max(0, l + 1 - n), min(l + 1, m))]
            r += temp[::-1]
            # print(temp[::-1])
        return r

    def constrainedSubsetSum(self, nums, k) -> int:
        from heapq import heappop, heappush
        ans = float('-inf')
        cur = 0
        h = []
        for i, n in enumerate(nums):
            pres = 0
            print('第', i, '个是', n)
            while h:
                pres, prei = heappop(h)
                print('pres', 'prei', pres, prei)
                if prei >= i - k:
                    print('in')
                    heappush(h, (pres, prei))
                    break
            cur = max(n - pres, n)
            print('cur = ', cur)
            heappush(h, (-cur, i))
            print(h, '\n')
            ans = max(cur, ans)
        return ans

    def allPathsSourceTarget(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: List[List[int]]
        """
        from collections import defaultdict
        d = defaultdict(list)
        for i, n in enumerate(graph):
            for j in n:
                d[i].append(j)
        n = len(graph) - 1
        # print(d)
        res = []
        def helper(start, lst):
            print(start, n , lst)
            if start == n:
                res.append(lst)
                return
            if start not in d:
                return
            for next in d[start]:
                helper(next, lst + [next])
        helper(0, [0])
        # print(res)
        return res

    def numTilePossibilities(self, tiles):
        from collections import Counter
        res = []
        c = Counter(tiles)
        alpha = set(tiles)
        def helper(string, char, a_dit):
            if not any(a_dit.values()):
                return
            string += char
            a_dit[char] -= 1
            res.append(string)
            for char in alpha:
                if a_dit[char]:
                    helper(string, char, dict(a_dit))
        for char in alpha:
            helper('', char, dict(c))
        print(res)
        return len(res)

    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        res = []
        flag = [True] * 9
        def helper(num, s, l, lst, f):
            if l >= k:
                if s == n:
                    res.append(lst)
                return
            for i in range(num, 10):
                p = f[:]
                if p[i - 1]:
                    p[i - 1] = False
                    helper(i, s + i, l + 1, lst + [i], p)

        for i in range(1, 10):
            f = flag[:]
            f[i - 1] = False
            helper(i, i, 1, [i], f)
        return res

    def findDiagonalOrder(self, nums):
        from collections import defaultdict
        d = defaultdict(list)
        res = []
        for i in range(len(nums)):
            for j in range(len(nums[i])):
                d[i + j].append(nums[i][j])
        print(d)
        for i in sorted(list(d.keys())):
            res.append(d[i][::-1])
        print(res)

    def findNumberIn2DArray(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        row, col = len(matrix), len(matrix) and len(matrix[0])
        r = 0
        c = col - 1
        while r < row and c >= 0:
            curr = matrix[r][c]
            if curr == target:
                return True
            if curr > target:
                c -= 1
            else:
                r += 1
            print(r, c)
        return False

    def letterCasePermutation(self, S):
        res = []
        n = len(S)
        def helper(i, string, S, n):
            if i >= n:
                res.append(string)
                return
            curr = S[i]
            if curr.isalpha():
                s1 = string + curr.lower()
                s2 = string + curr.upper()
                helper(i + 1, s1, S, n)
                helper(i + 1, s2, S, n)
            else:
                string += curr
                helper(i + 1, string, S, n)
            
        helper(0, '', S, n)
        return res

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        if n == 0:
            return []
        def helper(nums, n, curr, lst, marked):
            # print(lst)
            if curr == n:
                res.append(lst.copy())
                return
            for i in range(n):
                if not marked[i]:
                    if i and nums[i] == nums[i - 1] and not marked[i - 1]:
                        continue
                
                    marked[i] = True
                    lst.append(nums[i])
                    helper(nums, n, curr + 1, lst, marked)
                    marked[i] = False
                    lst.pop()

        nums.sort()
        marked = [False] * n
        helper(nums, n, 0, [], marked)
        return res

    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        return [i + extraCandies >= max(candies) for i in candies]

    def maxDiff(self, num: int) -> int:
        s = str(num)
        n = len(s)
        a = b = s
        for i in range(n):
            if s[i] != '9':
                a = s.replace(s[i], '9')
                break
        if n == 1:
            b = '1'
        else:
            if s[0] == '1':
                i = 1
                while i < n:
                    if s[i] != '0' and s[i] != '1':
                        b = s.replace(s[i], '0')
                        # print(b)
                        break
                    i += 1
            else:
                b = s.replace(s[0], '1')
        print(a, b)
        return int(a) - int(b)

    def checkIfCanBreak(self, s1: str, s2: str) -> bool:
        n1, n2 = map(len, (s1, s2))
        if n1 - n2:
            return False
        str1 = sorted([i for i in s1])
        str2 = sorted([j for j in s2])
        print(str1, str2)
        def judge(s1, s2, n):
            for i in range(n):
                print(s1[i], s2[i])
                if s1[i] < s2[i]:
                    print(i)
                    return False
            return True
        return judge(str1, str2, n1) or judge(str2, str1, n1)

    def numberWays(self, hats: List[List[int]]) -> int:
        n = len(hats)
        dp = [0] * (1 << n)
        connected = [[0] * 41 for _ in range(n + 1)]
        for i in range(n):
            for j in hats[i]:
                connected[i + 1][1 + j] = 1
        dp[0] = 1
        for mask in range(1, n):
            # print(mask)
            i = bin(mask).count('1') - 1
            # print(i)
            for j in range(40):
                if mask & (1 << j) and connected[i][j]:
                    print(j)
                    # print(1111111)
                    dp[mask] += dp[mask - (1 << j)]
                    # dp[mask] %= 10**9 + 7
        print(dp)
        return dp[-1]

    def destCity(self, paths: List[List[str]]) -> str:
        from collections import defaultdict
        d = {}#defaultdict(list)
        for _from, _to in paths:
            if _from not in d:
                d[_from] = _to
        # print(d)
        for _from, _to in paths:
            if _to not in d:
                # print(_from, _to)
                return _to
        
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        res = []
        if not nums:
            return True
        n = len(nums)
        for i in range(n):
            if nums[i] & 1:
                res.append(i)
        # print(res)
        n = len(res)
        if n <= 1:
            return True
        last = res[0]
        for i in range(1, n):
            if res[i] - last >= k + 1:
                last = res[i]
            else:
                return False
        else:
            return True

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        from itertools import combinations, permutations
        c = permutations(range(3), 2)
        print(list(c))

    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        n = len(mat)
        k = min(k, n ** 2)
        import heapq
        mark = [0] * n
        def helper(mat, lst):
            res = 0
            n = len(mat)
            for i in range(n):
                res += mat[i][lst[i]]
            return res
        head = []
        res = 0
        for i in range(n):
            heapq.heappush(head, (-mat[i][0], 0, 0))
            res += mat[i][0]
        print(head)

        # res = helper(mat, mark)
        while k:
            cur, x, y = heapq.heappop(head)
            print(-cur, x, y)
            res -= cur
            res += mat[x][y + 1]
            heapq.heappush(head, (-mat[x][y + 1], x, y + 1))
            k -= 1

        return 1

    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        if n <= 3:
            return []
        lines = [1 << i for i in range(n)]
        def make_row(n, row):
            line = ['.'] * row
            for i in range(row):
                if n & 1:
                    line[i] = 'Q'
                n = n >> 1
            return ''.join(line)

        def dfs(row, occupy, lst, n, lines, vis_l, vis_r):
            if row == n:
                res.append(lst.copy())
                return
            for i in range(n):
                if lines[i] & occupy == 0:
                    tl, tr = map(set, (vis_l, vis_r))
                    # print("第{}行第{}个".format(row, i))
                    # print(bin(lines[i]), bin(occupy))
                    left = row + i
                    right = row - i
                    if left in tl:
                        continue
                    tl.add(left)
                    if right in tr:
                        continue
                    tr.add(right)
                    # print(t)
                    # print(i)
                    occupy_n = lines[i] | occupy
                    cur_line = make_row(lines[i], n)
                    lst.append(cur_line)
                    dfs(row + 1, occupy_n, lst, n, lines, tl, tr)
                    lst.pop()
        # for i in lines:
        vis_l = vis_r = set()
        dfs(0, 0, [], n, lines, vis_l, vis_r)

        return res

    def buildArray(self, target: List[int], n: int) -> List[str]:
        res = []
        i = 1
        j = 0
        l = len(target)
        while i <= n and j < l:
            res.append('Push')
            if i != target[j]:
                res.append('Pop')
            if i == target[j]:
                j += 1
            i += 1
            #     if j >= l:
            #         break
        print(res)

    def countTriplets(self, arr: List[int]) -> int:
        n = len(arr)
        arr2 = arr[:]
        for i in range(1, n):
            arr2[i] ^= arr2[i - 1]
        print(arr2)
        res = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j, n):
                    if arr2[j - 1] ^ arr2[i] ^ arr[i] == arr2[k] ^ arr2[j - 1]:
                        res += 1
        # print(res)
        return res

    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        from collections import defaultdict
        d = defaultdict(list)
        for _from, _to in edges:
            d[_from].append(_to)
        res = []
        print(d)
        def helper(i, lst):
            if i not in d:
                res.append(lst)
                return
            for next in d[i]:
                helper(next, lst + [next])
        helper(0, [0])
        res.sort(key=lambda x: len(x))
        print(res)
        d = defaultdict(int)
        d[0] = 0
        flag = [False] * n
        for i in res:
            f = False
            for j in range(len(i) - 1, 0, -1):
                if hasApple[i[j]]:
                    f = True
                if f:
                    flag[i[j]] = flag[i[j - 1]] = True
                # d[i[j]] = 1 + d[i[j - 1]]
        # print(d, flag)
        return max(0, 2 * (sum(flag) - 1))

    def maxPower(self, s: str) -> int:
        if not s:
            return 0
        res = 1
        tmp = 0
        last = ''
        for i in s:
            if i == last:
                tmp += 1
                res = max(res, tmp)
            else:
                tmp = 1
                last = i
        return res

    def simplifiedFractions(self, n: int) -> List[str]:
        from math import gcd
        res = []
        for i in range(2, 1 + n):
            for j in range(1, n):
                # print(i, j)
                if j == 1 or gcd(i, j) == 1 and j < i:
                    t = str(j) + '/' + str(i)
                    res.append(t)
        print(res)
        # return res

    def largestNumber(self, cost: List[int], target: int) -> str:
        self.res = 0
        from collections import defaultdict
        d = defaultdict(str)
        for i in range(9):
            d[cost[i]] = 1 + i
        print(d)
        def helper(t, cost, lst, d):
            if t < 0:
                return
            if t == 0:
                self.res = max(self.res, lst)
                return
            for i in d:
                helper(t - i, cost, 10 * lst + d[i], d)
        helper(target, cost, 0, d)
        return self.res

    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        n = len(startTime)
        res = 0
        for i in range(n):
            if startTime[i] <= queryTime <= endTime[i]:
                res += 1
        return res

    def arrangeWords(self, text: str) -> str:
        res = ''
        string = text.lower().split(' ')
        # print(string)
        a = sorted(string, key=lambda x: len(x))
        return ' '.join(a).capitalize()

    def peopleIndexes(self, favoriteCompanies: List[List[str]]) -> List[int]:
        res = []
        n = len(favoriteCompanies)
        favoriteCompanies = list(map(set, favoriteCompanies))
        print(favoriteCompanies)
        for i in range(n):
            cur = favoriteCompanies[i]
            for j in range(n):
                if i == j:
                    continue
                if cur.issubset(favoriteCompanies[j]):
                    break
            else:
                res.append(i)

    def findTheLongestSubstring(self, s: str) -> int:
        D = {"a": 1, "e": 2, "i": 4, "o": 8, "u": 16}
        L = {0: 0}
        m = t = 0
        for i, c in enumerate(s, 1):
            t ^= D.get(c, 0)
            m = max(m, i - L.setdefault(t, i))
        return m

    def sequentialDigits(self, low: int, high: int) -> List[int]:
        string = ''.join(chr(i) for i in range(49, 58))
        nums = [int(string[:i]) for i in range(1, 10)]
        res = []
        for i, n in enumerate(nums):
            nine = 10 ** (i + 1) - 1
            eleven = nine // 9
            # print(n, eleven)
            while n <= high and n % 10:
                if n >= low:
                    res.append(n)
                n += eleven
        return res

    def maxVowels(self, s: str, k: int) -> int:
        res = 0
        n = len(s)
        aeiou = ('a', 'e', 'i', 'o', 'u')
        ori = s[:k]
        for i in ori:
            if i in aeiou:
                res += 1
        tmp = res
        print(ori, res)
        for i in range(1, n - k + 1):
            print(s[i: i + k], res, s[i - 1], s[i + k - 1])
            tmp += (s[i + k - 1] in aeiou) - (s[i - 1] in aeiou)
            # tmp += d.get(s[i + k], 0) - d.get(s[i - 1], 0)
            res = max(res, tmp)
        return res

    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        n1, n2 = map(len, (nums1, nums2))
        if not (n1 and n2):
            return 0
        res = nums1[0] * nums2[0]
        dp = [[float('-inf')] * (1 + n2) for _ in range(1 + n1)]

        for i in range(n1):
            for j in range(n2):
                tmp = nums1[i] * nums2[j]
                # if tmp >= 0:
                dp[i][j] = max(tmp, tmp + dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])
                # else:
                #     dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
                res = max(res, dp[i][j])
        print(dp)
        return res
    
    def decodeString(self, s: str) -> str:
        res = ''
        stack = []
        n = 0
        for i in s:
            # print(i)
            if i.isdigit():
                n = 10 * n + ord(i) - 48
                # print(n)
            elif i == '[':
                # stack.append(res)
                stack.append(n)
                stack.append(res)
                # print(res, n, stack, '[')
                n = 0
                res = ''
            elif i == ']':
                # print(res)
                res = stack.pop() + res * stack.pop()
                # print(res)
                # print(res, stack, ']')
                # stack.append(res)
            else:
                res += i
        return res

    def largestRectangleArea(self, heights: List[int]) -> int:
        ans, s, hs = 0, [0], [0, *heights, 0]
        for i, h in enumerate(hs):
            # print(i, h)
            while hs[s[-1]] > h:
                ans = max(ans, (i - s[-2] - 1) * hs[s.pop()])
            s.append(i)
            print(s, i)
        return ans

    def hasAllCodes(self, s: str, k: int) -> bool:
        dp = [False] * 2 ** k
        # print(dp)
        for i in range(len(s) - k + 1):
            t = s[i: i + k]
            m = int(t, 2)
            if 0 <= m < 2 ** k:
                dp[m] = True
        return all(dp)

    def checkIfPrerequisite(self, n: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        from collections import defaultdict
        d = defaultdict(set)
        dp = []
        for _from, _to in prerequisites:
            d[_from].add(_to)

        def dfs(d, x, y):
            if x not in d:
                return False
            if y in d[x]:
                return True
            else:
                for z in d[x]:
                    if dfs(d, z, y):
                        return True
                else:
                    return False
        
        for x, y in queries:
            # print(x, y)
            dp.append(dfs(d, x, y))

        return dp

    def maxProduct(self, nums: List[int]) -> int:
        nums.sort()
        return (nums[-1] - 1) * (nums[-2] - 1)

    def maxArea(self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]) -> int:
        horizontalCuts = [0] + [*sorted(horizontalCuts)] + [h]
        verticalCuts = [0] + [*sorted(verticalCuts)] + [w]
        def dp(lst):
            res = float('-inf')
            for i in range(1, len(lst)):
                res = max(res, lst[i] - lst[i - 1])
            return res
        print(horizontalCuts, verticalCuts)
        print(dp(horizontalCuts), dp(verticalCuts))
        return dp(horizontalCuts) * dp(verticalCuts)

    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        from collections import defaultdict
        self.res = 0
        d = defaultdict(list)
        for x, y in connections:
            d[x].append(y)
        edge = [[x, y] if x < y else [y, x] for x, y in connections]
        # edge = sorted(edge, key=lambda x: x[0])
        print(edge)
        cd = defaultdict(list)
        for x, y in edge:
            cd[x].append(y)
        print(cd)
        def dfs(start, next):
            if start not in d or start not in d[next]:
                self.res += 1
            for z in cd[start]:
                dfs(z, start)
        for i in cd[0]:
            dfs(0, i)

        return self.res

    def getStrongest(self, arr: List[int], k: int) -> List[int]:
        
        # res = []
        arr.sort()
        n = len(arr)
        mid = arr[(n - 1) >> 1]
        print(mid)
        arr.sort(key=lambda x: (abs(x - mid), x))
        print(arr)
        # res = sorted([abs(i - mid) for i in arr])[n - k:]
        # print(mid)
        return arr[n - k:]

    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        dp = [[[0] * m] for _ in range(n) for _ in range(target)]
        print(dp)
        for i in range(m):
            for j in range(n):
                for k in range(target):
                    dp[i][j]
        return 0

    def finalPrices(self, prices: List[int]) -> List[int]:
        n = len(prices)
        res = prices[:]
        for i in range(n - 1):
            for j in range(i + 1, n):
                if prices[j] <= prices[i]:
                    res[i] = prices[i] - prices[j]
                    break
        return res

    def minSumOfLengths(self, nums: List[int], k: int) -> int:
        res = -1
        s = 0
        t = []
        n = len(nums)
        d = {}
        for i in range(n):
            s += nums[i]
            d[s] = i
            if s - k in d:
                t.append((i, d[s - k]))
            print(t)
        return -1
    
    def minDistance(self, houses: List[int], k: int) -> int:
        res = 0
        houses.sort()
        n = len(houses)
        if n <= k:
            return 0
        dp = [[float('inf')] * (1 + k) for _ in range(1 + max(houses))]
        dp[houses[0]][0] = 0
        # print(dp)
        # for i in range(1, n):
        #     for j in range(k):
        #         dp[houses[i]][j] = min(       \
        #                 # 不放
        #                 houses[i] - houses[i - 1] + dp[houses[i - 1][j]],    \  
        #                 # 放
        #                                 )

        #         print(dp)
        
        return dp[max(houses)][k]

    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        from collections import Counter, defaultdict
        c = Counter(arr)
        # print(c)
        d = defaultdict(list)
        for _k in c:
            d[c[_k]].append(_k)
        print(d)
        keys = sorted(d.keys())
        print(keys)
        r = []
        for i in keys:
            for j in d[i]:
                r += i * [j]
        print(r)
        # arr.sort(key=lambda x: arr.count(x))
        
        return len(Counter(r[k:]))


    def reorderedPowerOf2(self, N: int) -> bool:
        from itertools import permutations
        from functools import reduce
        from operator import __add__

        f = lambda lst, n: [lst[i] * 10 ** (n - i - 1) for i in range(n)]
        g = lambda lst: (lst, len(lst))
        
        nums = [int(n) for n in str(N)]
        power = [2 ** i for i in range(30)]
        vis = set()
        print(power)
        for lst in permutations(*g(nums)):
            if lst in vis:
                continue
            vis.add(lst)
            a = reduce(__add__, f(*g(lst)), 0)
            print(a)
            if a in power:
                return True
        else:
            return False

    def isMatch(self, s: str, p: str) -> bool:
        """
        s: 字符串
        p: 正则表达式
        """
        m, n = map(len, (s, p))

        def match(i: int, j: int) -> bool:
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]
        
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] |= dp[i][j - 2]
                    if match(i, j - 1):
                        dp[i][j] |= dp[i - 1][j]
                else:
                    if match(i, j):
                        dp[i][j] |= dp[i - 1][j - 1]

        return dp[m][n]

    def xorOperation(self, n: int, start: int) -> int:
        from operator import __xor__
        from functools import reduce
        return reduce(__xor__, [2 * i + start for i in range(n)])

    def getFolderNames(self, names: List[str]) -> List[str]:
        from collections import defaultdict
        res = []
        vis = defaultdict(int)

        for file in names:
            if file in vis:
                n = vis[file]
                print(n)
                s = file + '({})'
                while s.format(n) in vis:
                    n += 1
                s = s.format(n)
                print(s)
                vis[s] = 1
                res.append(s)
            else:
                res.append(file)
                vis[file] = 1
            print()
            
        return res

    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        from collections import defaultdict
        d = defaultdict(int)
        for cpdomain in cpdomains:
            n, domain = cpdomain.split(' ')
            do = domain.split('.')
            tmp = ''
            for i in range(len(do) - 1, -1, -1):
                if i == len(do) - 1:
                    tmp = do[i]
                else:
                    tmp = do[i] + '.' + tmp
                d[tmp] += int(n)
        return [str(value) + ' ' + key for key, value in d.items()]

    def average(self, salary: List[int]) -> float:
        min_s = min(salary)
        max_s = max(salary)
        n = len(salary)
        s = sum(salary)
        return (s - min_s - max_s) / (n - 2)

    def kthFactor(self, n: int, k: int) -> int:
        res = []
        N = n
        part = int(n ** 0.5)
        if part ** 2 == n:
            s = 0
        else:
            s = 1
        n = 0
        for i in range(1, 1 + part):
            if N % i == 0:
                n += 1
                res.append(i)
                if n == k:
                    return i
        print(res)
        n = len(res)
        if n >= k:
            return res[k - 1]
        if s == 0:
            if 2 * n - 1 < k:
                return -1
            else:
                if n >= k:
                    return res[k - 1]
                else:
                    print(res[k - n], k - n)
                    return N // res[k - n]
        if s == 1:
            if 2 * n < k:
                return -1
            else:
                if n >= k:
                    return res[k - 1]
                else:   
                    # print(res[k - n], k - n)
                    return N // res[n - k]

    def longestSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return nums.count(1)
        dp = [0] * n
        dp[0] = int(nums[0] == 1)
        for i in range(1, n):
            if nums[i] == 1:
                dp[i] = 1 + dp[i - 1]
        print(dp)
        for i in range(n - 2, -1, -1):
            if nums[i] and dp[i + 1]:
                dp[i] = dp[i + 1]
        print(dp)
        res = max(dp[0], dp[-1])
        # if nums[0] == 0:
        #     res = dp[1]
        # if nums[-1] == 0:
        #     res = max(dp[-2], res)
        print(res)
        for i in range(1, n - 1):
            if nums[i] == 0:
                res = max(res, dp[i - 1] + dp[i + 1])
        return res - 1 if res == n else res


    def minNumberOfSemesters(self, n: int, dependencies: List[List[int]], k: int) -> int:
        from collections import defaultdict
        if not dependencies:
            return n // k + int(n % k > 0)
        # if len(dependencies) == 1:
        #     return n // k
        Benz = defaultdict(set)
        Tesla = defaultdict(int)
        DasAuto = defaultdict(int)
        BMW = 0
        for _from, _to in dependencies:
            Benz[_to].add(_from)
        print(Benz)
        def dfs(root):
            for node in Benz[root]:
                Tesla[node] = max(Tesla[root] + 1, Tesla[node])
                dfs(node)
        Tesla[n] = 0
        left = 0
        for i in range(1, n + 1):
            if i in Benz:
                dfs(i)
            else:
                left += 1
        print(Tesla)
        for key in Tesla:
            DasAuto[Tesla[key]] += 1
        print(DasAuto)
        for key in DasAuto:
            t = DasAuto[key] % k
            if t == 0:
                BMW += DasAuto[key] // k
            else:
                if left:
                    t -= min(k - DasAuto[key] % k, t)
                    BMW += DasAuto[key] // k + 1
                else:
                    BMW += DasAuto[key] // k
        return BMW

    def numSubmat(self, matrix: List[List[int]]) -> int:
        maxarea = 0
        res = 0

        dp = [[0] * len(matrix[0]) for _ in range(len(matrix))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if not matrix[i][j]: 
                    continue

                # compute the maximum width and update dp with it
                width = dp[i][j] = dp[i][j-1] + 1 if j else 1

                # compute the maximum area rectangle with a lower right corner at [i, j]
                for k in range(i, -1, -1):
                    width = min(width, dp[k][j])
                    maxarea = max(maxarea, dp[k][j] + (i-k+1))
                    res += maxarea
        return res

    def rangeSum(self, nums: List[int], n: int, left: int, right: int) -> int:
        BMW = []
        for i in range(n):
            tmp = 0
            for j in range(i, n):
                tmp += nums[j]
                BMW.append(tmp)
        BMW.sort()
        return sum(BMW[left - 1: right])

    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 4:
            return 0
        nums.sort()
        # 012
        # 0 1 -1
        # 0 -2 -1
        # -3 -2 -1
        _0 = nums[-1] - nums[3]
        _1 = nums[-2] - nums[2]
        _2 = nums[-3] - nums[1]
        _3 = nums[-4] - nums[0]
        return min(_0, _1, _2, _3)

    def winnerSquareGame(self, n: int) -> bool:
        BMW = [False] * (1 + n)
        for i in range(1, 1 + n):
            for j in range(1, 1 + i):
                if j ** 2 <= i and not BMW[i - j ** 2]:
                    BMW[i] = True
                    break
        return BMW[-1]

    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        from collections import defaultdict
        edge = len(edges)
        d = defaultdict(list)
        p = defaultdict(float)
        if start > end:
            start, end = end, start
        self.res = 0
        for i in range(edge):
            _from, _to = edges[i]
            d[_from].append(_to)
            if _from > _to:
                _from, _to = _to, _from
            p[(_from, _to)] = succProb[i]
        print(d)
        print(p)
        print()
        def dfs(i, pro):
            if i == end:
                self.res = max(self.res, pro)
                return
            if i in d:
                for j in d[i]:
                    if i > j:
                        i, j = j, i
                    # print(i, j)
                    dfs(j, pro * p[(i, j)])
        dfs(end, 1)
        return self.res

    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        from collections import defaultdict
        dp = [''] * n
        dp[0] = labels[0]
        d = defaultdict(list)
        for i in range(n - 1):
            _from, _to = edges[i]
            dp[i + 1] = labels[i + 1]
            d[_from].append(_to)
            d[_to].append(_from)
        print(d, dp)
        def dfs(node, str, vis):
            if node in vis:
                return 0
            if node not in d:
                return int(dp[node] == str)
            tmp = 0
            if dp[node] == str:
                tmp += 1
            for _node in d[node]:
                tmp += dfs(_node, str, vis + [node])
            return tmp
        # print(dfs(0, 'a'))
        vis = set()
        res = []
        for i in range(n):
            res.append(dfs(i, dp[i], []))
            vis.add(i)
            print(i, vis)
        return res

    # def countSubTrees(self, arr: List[int], target: int) -> int:
    #     def fun(arr, l, r):
    #         if l > r:
    #             return -1000000000000
    #         ans = arr[l]
    #         for i in range(l + 1, r + 1):
    #             ans &= arr[i]
    #         return ans
    #     res = float('inf')
    #     n = len(arr)
    #     for i in range(n):
    #         for j in range(i, n):
    #             res = min(res, abs(fun(arr, i, j) - target))
    #     return res

    def numOfSubarrays(self, arr: List[int]) -> int:
        res = set()
        n = len(arr)
        for i in range(n):
            tmp = 0
            for j in range(i, n):
                tmp += arr[j]
                print(arr[i], arr[j], tmp)
                if tmp % 2:
                    res.add(tmp)
            print()
        print(res)
        return len(res) % (10 * 9 + 7)

    def numSplits(self, s: str) -> int:
        n = len(s)
        if n == 1:
            return 0
        sss = s[::-1]
        dp1 = []
        dp2 = []
        for i in range(n):
            if not dp1:
                ss = set()
            else:
                ss = dp1[-1].copy()
            ss.add(s[i])
            dp1.append(ss)
            if not dp2:
                ss = set()
            else:
                ss = dp2[-1].copy()
            ss.add(sss[i])
            dp2.append(ss)
            print(dp1, dp2)
        count = 0
        for i in range(1, n - 1):
            print(dp1[i], dp2[n - 1 - i])
            if len(dp1[i]) == len(dp2[n - 2 - i]):
                count += 1
        return count


    def minNumberOperations(self, target: List[int]) -> int:
        peak = 0
        count = 0
        vally = []
        n = len(target)
        if n == 1:
            return target[0]
        if n == 2:
            return max(target)
        target = [0] + target + [0]
        for i in range(n + 1):
            if target[i] > target[i - 1] and target[i] > target[i + 1]:
                peak += target[i]
                count += 1
            if target[i] < target[i - 1] and target[i] < target[i + 1]:
                vally.append(target[i])
        print(peak, count, vally)
        if not count:
            return max(target)
        if count == 1:
            return peak
        return peak - 1

    def getWinner(self, arr: List[int], k: int) -> int:
        from collections import deque, defaultdict
        mm = max(arr)
        dq = deque(arr)
        d = defaultdict(int)
        while True:
            a1, a2 = dq.popleft(), dq.popleft()
            a_max, a_min = max(a1, a2), min(a1, a2)
            if a_max == mm:
                return mm
            dq.appendleft(a_max)
            dq.append(a_min)
            d[a_max] += 1
            d[a_min] = 0
            if d[a_max] == k:
                return a_max
        
    def minSwaps(self, grid: List[List[int]]) -> int:
        from collections import defaultdict
        d = defaultdict(list)
        n = len(grid)
        for j in range(n):
            line = grid[j]
            zero = 0
            for i in range(n - 1, -1, -1):
                if line[i] == 0:
                    zero += 1
                else:
                    break
            d[zero].append(j)
        print(d)
        if len(d) != n:
            return -1
        res = 0
        for i in range(n):
            if d[i] != n - i:
                for j in range(i + 1, n):
                    pass


    def maxSum(self, nums1: List[int], nums2: List[int]) -> int:
        from collections import defaultdict
        MOD = 10 ** 9 + 7
        h = lambda lst: [0] + lst + [-1]
        common = set(nums1) & set(nums2)
        nums1, nums2 = map(h, (nums1, nums2))
        s = 0
        def helper(lst):
            d = defaultdict(int)
            t = 0
            for n in lst:
                t += n
                if n in common:
                    d[n] = t
                    t = 0
            d[-1] += 1
            return d
        d1, d2 = map(helper, (nums1, nums2))

        for i in common:
            s += max(d1[i], d2[i])

        return s % MOD

    def findKthPositive(self, arr: List[int], k: int) -> int:
        n = len(arr)
        res = 1
        i = 0
        while k and i < n:
            if res != arr[i]:
                k -= 1
                if k == 0:
                    return res
                res += 1
                print(1, res, arr[i], k)
            else:
                res += 1
                print(2, res, arr[i], k)
                i += 1
        return res + k - 1

    def longestAwesome(self, s: str) -> int:
        dp = [0] * 10
        state = {
            0: dp
        }
        res_n = 0
        n = len(s)
        count = lambda ls1, ls2, n: [ls1[i] ^ ls2[i] for i in range(n)]
        vaild = lambda lst: lst.count(1) <= 1
        for i in range(n):
            char = s[i]
            dp[ord(char) - ord('0')] ^= 1
            state[i + 1] = dp
            if vaild(count(state[i + 1], state[i], n)):
                res_n = max(res_n, i)
        print(res_n)
        # for k in state:
        #     print(k, state[k])
        for i in range(n - 1):
            for j in range(i + 1, n):
                pass

    def makeGood(self, s: str) -> str:
        stack = []
        for char in s:
            if not stack:
                stack.append(char)
            else:
                tmp = stack[-1]
                if abs(ord(tmp) - ord(char)) == 32:
                    stack.pop()
                else:
                    stack.append(char)
        return ''.join(stack)

    def findKthBit(self, n: int, k: int) -> str:
        if n == 1 and k == 1:
            return '0'
        len_n = 2 ** n - 1
        mid = (len_n - 1) / 2 + 1
        # print(k, mid)
        if k == mid:
            return '1'
        if k > mid:
            tmp = self.findKthBit(n, len_n - k + 1)
            if tmp == '0':
                return '1'
            else:
                return '0'
        if k < mid:
            return self.findKthBit(n - 1, k)

    def maxNonOverlapping(self, nums: List[int], target: int) -> int:
        res = 0

        return res

    def minOperations(self, n: int) -> int:
        f = lambda x: x * (x + 1) // 2
        flag = n & 1
        if flag:
            left = (n - 1) // 2
            return 2 * f(left)
        else:
            left = n // 2
            return f(left) * 2 - left

    def minDays(self, n: int) -> int:
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):
            bmw = i % 6
            if bmw == 0:
                dp[i] = min(dp[i], 1 + dp[i - i // 2], 1 + dp[i - 2 * i // 3])
            elif bmw == 2:
                dp[i] = min(dp[i], 1 + dp[i - i // 2])
            elif bmw == 3:
                dp[i] = min(dp[i], 1 + dp[i - 2 * i // 3])
            else:
                dp[i] = 1 + dp[i - 1]
        print(dp)
        return dp[n]

    def maxDistance(self, position: List[int], m: int) -> int:
        position.sort()
        n = len(position)
        if n == 2:
            return position[-1] - position[0]
        return 1
    
    def mostVisited(self, n: int, rounds: List[int]) -> List[int]:
        dp = [0] * n
        dp[rounds[0] - 1] = 1
        for i in range(len(rounds) - 1):
            _1, _2 = rounds[i: i + 2]
            print(_1, _2)
            if _2 > _1:
                for j in range(_1, _2):
                    dp[j] += 1
            else:
                for j in range(_1, n):
                    dp[j] += 1
                for j in range(_2):
                    dp[j] += 1
            print(dp)
        m = max(dp)
        return [i + 1 for i in range(n) if dp[i] == m]

    def maxCoins(self, piles: List[int]) -> int:
        n = len(piles) // 3
        piles.sort()
        s = 0
        for i in range(n, 3 * n, 2):
            s += piles[i]
        return s

    def findLatestStep(self, arr: List[int], m: int) -> int:
        n = len(arr)
        res = []
        d = {}
        for _ in range(n):
            i = arr[_]
            if i not in d:
                d[i] = 1
            if i + 1 in d and i - 1 not in d:
                d[i] = d[i + 1] + 1
                d[i + 1] = d[i]
            if i - 1 in d and i + 1 not in d:
                d[i] = d[i - 1] + 1
                d[i - 1] = d[i]
            if i - 1 in d and i + 1 in d:
                d[i] = 1 + d[i - 1] + d[i + 1]
                d[i + 1] = d[i - 1] = d[i]
            if d[i] == m:
                res.append(i)
            while res and res[0] > m:
                res = res[1:]
            
            print(i, d, res)
        return 1

    def make_cancellation(self , content , bomb):
        import re
        p = re.findall(r'((.)\2{1})', content)
        bomb = str(bomb)
        while p:
            for b, n in p:
                if n != bomb:
                    content = content.replace(b, '')
            p = re.findall(r'((\d)\2{1})', content)
            tmp = []
            for b, n in p:
                if n != bomb:
                    tmp.append((b, n))
            p = tmp
        bb = bomb * 2
        if bb not in content:
            return content
        while bb in content:
            i = content.index(bb)
            content = content.replace(content[max(0, i - 1): i + 3], '')
        return content
    
a = Solution().make_cancellation(
    "111255aabbc", 1
)
p(a)
