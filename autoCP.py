import re
import random

import os
import git

import time
from datetime import datetime
from collections import defaultdict

LOCAL = os.path.dirname(__file__)

LANG = {
    'c': 'C',
    'cpp': 'C++',
    'csharp': 'C#',
    'java': 'Java',
    'javascript': 'JavaScript',
    'python': 'Python',
    'ruby': 'Ruby',
    'golang': 'Go',
    'swift': 'Swift',
}

TITLE = """
# LeetCode progress

[![](https://img.shields.io/badge/LeetCode-{}/1303-yellow.svg?style=social&logo=leetcode)](https://leetcode-cn.com/u/ltcccc/)

![](https://img.shields.io/github/commit-activity/m/ltc1996/leetcode)
![](https://img.shields.io/github/last-commit/ltc1996/leetcode)

| 序号 | 题目 |  语言 | 难度 |
|---:|:-----:| :-------:|:----------:|
"""

DIFF_SEP = '|---| ----| ---- |{}| --- |'        # .format(diff)

PATTERN = '| {} | {} | **{}** | {} |'       # .format(num, name, langs, diff)

DIFF_RANGE = (
    '困难',
    '中等',
    '简单',
    '空',
)


def file_change():
    """
    :return: list, list
        stands for:
            untrack, add files
    """
    print(LOCAL)
    repo = git.Repo(LOCAL)  # local git repo

    # untracked files if any else []
    untrack = repo.untracked_files

    # add files if any else []
    add = []
    hcommit = repo.head.commit
    # print(repo.untracked_files)
    for diff_added in hcommit.diff().iter_change_type('A'):
        d = diff_added.b_blob.path
        # print(d)   # , type(diff_added))
        # print(1)
        if os.path.basename(d)[-3:] == '.md':
            add.append(d)

    # res = untrack + add
    return untrack, add


def commit(file_list):
    repo = git.Repo(LOCAL)
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_str = 'AUTO push: {} @' + time_now
    print(commit_str)
    try:
        repo.commit(commit_str)
        print('commit √' + '\n')
    except IOError:
        print('commit ×' + '\n')


def push():
    """push and pull behaves similarly to `git push|pull`
    :return: None
    """
    repo = git.Repo(LOCAL)
    try:
        repo.remote().push('master')
        print('push √' + '\n')
    except IOError:
        print('push ×' + '\n')


def get_md_info(file_path):
    """
    :param file_path:
        str: .md文件名
    :return:
        dict:
            题号: 1.
            题名: 两数之和,
            难度: 简单,
            语言: cpp,
    """
    # print('opening file: {}'.format(file_path))
    if not os.path.exists(file_path):
        raise FileNotFoundError
    info = {
        'num': 0,
        'name': '',
        'diff': '简单',
        'lang': set(),
    }
    languages = set()
    f = True  # 难度只搜一次
    with open(file_path, encoding='utf-8') as f:
        if f:
            p = r'(\d+)\.\s*(.+)'  # 匹配题号和题名
            q = re.findall(p, f.readline())
            if q:
                num, name = q[0]
        # print(num, name)
        for line in f.readlines()[1:]:
            lang = r'```\s*(.+)'  # 语言(s)
            q = re.findall(lang, line)
            if q:
                languages.add(q[0])
            diff = r'- (.{2})'  # 难度
            q = re.findall(diff, line)
            if f and q:
                info['diff'] = q[0]
                f = False
    # print('file {} uses language(s):', ', '.join(languages))

    # 灌数据
    info['num'] = int(num)
    info['name'] = name
    info['lang'] = list(languages)
    return info


def get_readme():
    """
    :return:
        dict:
            困难: ...
            中等: ...
            简单: ...
        int:
            sum(1)
    """
    file_path = 'README.md'
    if not os.path.exists(file_path):
        raise FileNotFoundError
    progress = defaultdict(list)
    count = 0
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            p = r'\|\s+\d+\s+\|.*\|.*\|\s+(.*)\s+\|'
            q = re.match(p, line)
            if q:
                query, diff = (q.group(i) for i in range(2))
                print(query, diff)
                # diff = '简单' if diff == '空' else diff
                progress[diff].append(query)
                count += 1
    # print(count)
    # print(progress)

    return progress, count


def update_info(dict_old, dict_new):
    """update new .md into old .md
    and sort according to its diff(first) and num(second)
    :param dict_old:
    :param dict_new:
    :return:
    """
    for diff, statement in dict_new.items():
        dict_old[diff].append(statement)
    if '空' in dict_old:
        if '简单' in dict_old:
            dict_old['简单'] += dict_old['空']
            del dict_old['空']
    print(dict_old)


def set_readme(progress_dict, count):
    file_path = 'README_test.md'
    print('set README.md, add statement * {}'.format(str(count)))
    with open(file_path, 'w', encoding='utf-8') as f:
        title_now = TITLE.format(count)
        print(title_now)
        f.write(title_now)
        for diff in progress_dict:
            diff_sep = DIFF_SEP.format(diff)
            f.write(diff_sep)
            for statement in progress_dict[diff]:
                f.write(statement)

    print('set README.md √')


def make_num(num, indent=' '):
    """
    :param indent:
        str: _
    :param num:
        int: 5
    :return:
        str:
            '_' + '___5' + '_'
    """
    num_str = str(num)
    s = len(num_str)
    n_full = indent * (4 - s) + num_str
    return n_full


def make_name(num: int, name: str, diff: str) -> str:
    """
    :rtype: str
    :param num: int: 5
    :param name: str: 最长回文子串
    :param diff: str: 中等
    :return: str: [最长回文子串](中等/0005%20最长回文子串.md)
    """
    num = make_num(num, '0')
    file_name = '{}/{}%20{}.md'.format(diff, num, name)
    res = '[{}]({})'.format(name, file_name)
    return res


def make_lang(langs):
    """
    :param langs:
        list:
            [cpp, python]
    :return:
        str:
            **C++/Python**
    """
    sorted(langs)  # key=lambda x: LANG[x], reverse=True)
    res = [LANG[lang] if lang in langs else 'nil' for lang in langs]
    ls = '/'.join(res)
    return ls


def make_statement(info_dict):
    """
    :param info_dict:
        info:
            题号, 5
            题名, 最长回文子串
            难度, 中等
            语言, C++/Python
    :return:
        str:
            |    5 | [最长回文子串](中等/0005%20最长回文子串.md) | **C++/Python** | 中等 |
    """
    lang = make_lang(info_dict['lang'])
    print(lang)
    diff = info_dict['diff']
    num = info_dict['num']
    name = make_name(num, info_dict['name'], diff)
    num = make_num(num, ' ')
    # print(name, num, lang)

    pattern = PATTERN.format(num, name, lang, diff)
    return pattern


def main():
    untrack, add = file_change()
    files = untrack + add
    if len(files):
        print('发现文件:\n')
        for file in files:
            print(file)
        ans = input('\n是否提交? y/n\n').lower()
        if ans == 'y':
            progress_dict, progress_count = get_readme()
            print(progress_count, progress_dict)
            set_readme({}, 0)
            to_append = list()
            # for file in files:
            #     info = get_md_info(file)
            #     print(info)
            #     s = make_statement(info)
            #     print(s)

            # push()
        else:
            print('提交取消 ×')
    else:
        print('无文件修改 ⚪')

    input('任意键退出\n')
    return


if __name__ == '__main__':
    main()
    # print(file_change())

    # change_md()
    # l = get_md_info(r'D:\Charge\444\中等\0222 完全二叉树的节点个数.md')
    # print(l)
    # s = make_statement(l)
    # print(s)
