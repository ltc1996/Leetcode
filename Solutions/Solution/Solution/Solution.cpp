#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>

using namespace std;

class Solution {
public:
    void dfs(vector<int>& nums, int n, int curr, vector<int> lst, vector<bool>& marked) {
        if (lst.size() == n) {
            res.push_back(lst);
            return;
        }
        for (int i = 0; i < n; i++) {
            if (!marked[i]) {
                if ((i > 0) && (nums[i] == nums[i - 1]) && (!marked[i - 1]))
                    continue;
                marked[i] = true;
                lst.push_back(nums[i]);
                dfs(nums, n, ++curr, lst, marked);
                marked[i] = false;
                lst.pop_back();
            }
        }
        return;

    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return {};
        sort(nums.begin(), nums.end());
        vector<int> lst;
        vector<bool> marked(n, false);
        dfs(nums, n, 0, {}, marked);
        return res;
    }
private:
    vector<vector<int> > res;
};

class Derived
{
public:
    Derived();
    ~Derived();

protected:
    int a = 1;
private:
    int b = 1;
};

Derived::Derived()
{
}

Derived::~Derived()
{
}

int main()
{   
    char str[3] = {'h', 'w', '\0'};
    string s1 = "hw", s2 = "wh";
    string s3 = s1 + s2;
    cout << s3 << endl;
    getline(cin, s1);
}

