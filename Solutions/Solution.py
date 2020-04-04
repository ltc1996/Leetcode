import re


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

    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        from collections import deque
        import re
        res = 0
        stack = []
        sign = 1
        n = len(s)
        index = 0
        while index < n:
            if s[index] == ' ':
                index += 1
            elif s[index] == '-':
                sign = -1
                index += 1
            elif s[index] == '+':
                sign = 1
                index += 1
            elif s[index] == '(':
                stack += [res, sign]
                res = 0
                sign = 1
                index += 1
            elif s[index] == ')':
                res = res * stack.pop() + stack.pop()
                index += 1
            elif s[index].isdigit():
                tmp = int(s[index])
                while index < n and s[index].isdigit():
                    tmp = 10 * tmp + int(s[index])
                    index += 1
                res += tmp * sign
        return res

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

    def findTheLongestSubstring(self, s):
        # vowel = ('a'，'e'，'i'，'o'，'u')
        return s

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

a = Solution().numRescueBoats(
    people = [3,5,3,4], limit = 5

)

p(a)
# def square(n, time):
#     try:
#         1 / time
#     except:
#         return n
#     return n + square(n - 1, time - 1)

# print(square(3, 3))


# while 1:
#     #每组第一行是N和M 
#     #每组第一行是N和M 
#     from collections import defaultdict
#     from itertools import combinations
#     d = defaultdict(int)
#     nm = list(map(int,input().split(" ")))
#     n, m = nm
#     #print(n, m)
#     res = list(map(int,input().split(" ")))
#     #print(res)
#     count = 0
#     for n in res:
#         d[n] += 1
#     # print(d)
#     max_d, min_d = max(d), min(d)
#     if max_d - min_d <= m:
#         print(0)