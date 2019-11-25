## Git for GitHub

### change my own repo
1. download Git

2. open folder in PC, Shift + right click to open `Git Bash`

3. input:

    ```bash
    git init
    git remote add origin https://github.com/ltc1996/Leetcode.git
    git pull --rebase origin master
    git push -u origin master
    ```
4. open Atom, when folder become yellow

5. open folder, when `.git` and all file in PC


### fork others repo
1. click `fork` button in their repo

2. click `clone`,  such as url = `https://github.com/ltc1996/other-repo-name.git`

3. open folder and Bash `git clone url`

4. check if the branch is `master`? and create a new work repo `work` via
   
   > `git branch -a`
   
   > `git checkout -b work master`

5. make changes

6. create remote repo `work` via

    > `git push origin work`
    
    > `git branch -a`
    
    > `git add .` if add / rename files
    
7. the push `git push` + `first time to push` ? `--setup-stream oringin work` : `` ;

8. open THEIR repo and `new pull request`

### delete

1. `git branch -a` / `git branch -r`  checkout all branch in local / remote

2. `git branch -d local_branch1`  delete local branch

3. `git push origin --delete remote_branch1` delete remote branch

### html

1. https://shields.io/
2. https://simpleicons.org/
