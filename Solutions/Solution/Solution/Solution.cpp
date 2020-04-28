#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <unordered_map>

using namespace std;
int question1(int, int);
int question2(int);
int question2(int, int);
vector<int> singleNumbers(vector<int>& nums);
int nthUglyNumber(int n);

int main()
{   
    int n = 5;
    cout << nthUglyNumber(n) << endl;
}

int question1(int n, int m){
    return 0;
}

int question2(int m) {
    int n, f = 0;
    cin >> n;
    for (int i = 1; i <= n; i++) {
        // cout << f << endl;
        f = (f + m) % i;
    }
    return 1 + f;
}

int question2(int n, int m) {
    vector<int> dp;
    int s = n, k = 0, count = 1;
    for (int i = 0; i < n; i++)
        dp.push_back(i + 1);

    while (s > 1) {
        int t = 0;
        cout << "第" << count++ << "轮" << endl;
        vector<int> tmp;
        //cout << "in" << endl;
        for (int i = 0; i < s; i++) {
            //cout << dp[i] << endl;
            if (((k + 1 + i) % 10) != m && ((k + 1 + i) % m != 0)) {
                cout << "剩下" << dp[i] << endl;
                tmp.push_back(dp[i]);
                t++;
            }
        }
        k += s % m;
        s = t;
        cout << "size = " << s << endl;
        dp = tmp;
        /*if (s == 1)
            return dp[0];
        for (int i : dp)
            cout << i << endl;*/
    }
    return dp[0];
}

vector<int> singleNumbers(vector<int>& nums)
{
    int res = 0;
    for (int i : nums)
        res ^= i;
    //int index = res & (-res);
    int index = 0;
    while (((res & 1) == 0) && (index < 8 * sizeof(int)))
    {
        res >>= 1;
        index++;
    }
    //cout << res << index << endl;
    int num1 = 0, num2 = 0;
    for (int i : nums) {
        if (((i >> index) & 1) == 0)
            num1 ^= i;
        else
            num2 ^= i;
    }
    return vector<int>{num1, num2};
}

int nthUglyNumber(int n)
{
    if (n <= 0)
        return 0;
    int* pUglyNumbers = new int[n];
    pUglyNumbers[0] = 1;
    int nextUglyIndex = 1;
    const int* pm2 = pUglyNumbers;
    const int* pm3 = pUglyNumbers;
    const int* pm5 = pUglyNumbers;
    while (nextUglyIndex < n)
    {
        int m = min(*pm2 * 2, min(*pm3 * 3, *pm5 * 5));
        pUglyNumbers[nextUglyIndex++] = m;
        while (*pm2 * 2 <= m) pm2++;
        while (*pm3 * 3 <= m) pm3++;
        while (*pm5 * 5 <= m) pm5++;
        cout << *pm2 << *pm3 << *pm5 << endl;
    }
    int res = pUglyNumbers[n - 1];
    delete[] pUglyNumbers;
    return res;
}
