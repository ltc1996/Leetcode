import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Solution {
    public static final double PI = 3.14;

    public static void main(String[] args){
        double salary = 0;
        double x = 4;
        double y = Math.sqrt(x);
        String greeting = "hello world";
        String greeting_s = greeting.substring(0);
        int l = greeting.length();
        char c = greeting.charAt(2);
        int[] codePoints = greeting.codePoints().toArray();
        System.out.println(codePoints);
        boolean a = greeting.equalsIgnoreCase(greeting_s) || false;
        System.out.println("hw" + '8' + greeting_s);
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack_combinationSum(candidates, target, res, 0, new ArrayList<Integer>());
        return res;
    }

    private void backtrack_combinationSum(
            int[] candidates,
            int target,
            List<List<Integer>> res,
            int i,
            ArrayList<Integer> tmp_list
            ){
        if(target < 0)
            return;
        if(target == 0){
            res.add(new ArrayList<>(tmp_list));
            return;
        }
        for(int start = i; start < candidates.length; start++){
//            if(target < 0)
//                break;
            tmp_list.add(candidates[start]);
            backtrack_combinationSum(
                    candidates,
                    target - candidates[start],
                    res,
                    start,
                    tmp_list
                    );
            tmp_list.remove(tmp_list.size() - 1);
        }
    }


}




