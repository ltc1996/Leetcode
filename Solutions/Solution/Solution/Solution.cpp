#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <unordered_map>

using namespace std;
int question1(int, int);
int question2(int);
int question2(int, int);

int main()
{
    int N, M;
    // 每组第一行是2个整数，N和M，至于为啥用while，因为是多组。
    while (cin >> N >> M)
        cout << N << " " << M << endl;
    cout << question2(10, 3) << endl;
    return 0;
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