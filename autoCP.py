import re
import random

import os
import git

import time
from datetime import datetime
from collections import defaultdict

TIME_NOW = datetime.now().strftime("%Y-%m-%d %H:%M")

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
    'sql': 'SQL',
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


def log(string, sign_index):
    """ print format string in the screen startswith status
    sign_index Character Meaning
    --------- ---------------------------------------------------------------
    0:  '⚪',    prepare
    1:  '×',    fail
    2:  '√',    done
    3:  '＋',   something new
    """

    sign = (
        '⚪',
        '×',
        '√',
        '＋',
    )
    log_format = '[ {} ]: '.format(sign[sign_index])
    print(log_format + string)


def file_change():
    """
    :return: list, list
        stands for:
            untrack, add files
    """
    print('repo in ', LOCAL)
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


def get_num(file_list):
    nums = []
    for file_name in file_list:
        if file_name[-3:] != '.md':
            continue
        p = r'\d{4}'
        q = re.findall(p, file_name)
        nums.append(q[0])

    return ', '.join(nums)


def add_file(file_list):
    repo = git.Repo(LOCAL)
    index = repo.index
    nums = get_num(file_list)
    add_str = 'AUTO Add: {} @'.format(nums) + TIME_NOW

    try:
        index.add(file_list)
        log(add_str, 3)
    except Exception as e:
        log(e, 1)


def commit(file_list):
    nums = get_num(file_list)
    index = git.Repo(LOCAL).index
    commit_str = 'AUTO Commit: ' + 'add {} '.format(nums) + '@' + TIME_NOW
    # # print(commit_str)
    try:
        index.commit(commit_str)
        log(commit_str, 2)
        # print('commit √' + '\n')
    except IOError:
        log(commit_str, 1)
        # print('commit ×' + '\n')


def push():
    """push and pull behaves similarly to `git push|pull`
    :return: None
    """
    repo = git.Repo(LOCAL)
    # remote = repo.create_remote('origin', repo.remotes.origin.url)
    push_str = 'AUTO Push. @' + TIME_NOW
    try:
        repo.remote().push()
        log(push_str, 2)
        # print('push √' + '\n')
    except IOError:
        log(push_str, 1)
        # print('push ×' + '\n')


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
    if file_path[-3:] != '.md':
        return {}
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
            p = r'(\d+)\.\s*(\[(.*)\]|.+)'  # 匹配题号和题名
            q = re.findall(p, f.readline())
            if q:
                num, name, name_yes = q[0]
                name = name or name_yes
        # print(num, name)
        for line in f.readlines()[1:]:
            lang = r'```\s*(.+)'  # 语言(s)
            q = re.findall(lang, line)
            if q:
                languages.add(q[0])
            diff = r'- (.{2})'  # 难度
            q = re.findall(diff, line)
            if f and q:
                assert q[0] in DIFF_RANGE
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
                # print(query, diff)
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
    brand_dict = dict()
    for diff in dict_new:
        dict_old[diff] += dict_new[diff]

    if '空' in dict_old:
        if '简单' in dict_old:
            dict_old['简单'] += dict_old['空']
        del dict_old['空']

    for k in dict_old:
        brand_dict[k] = sorted(dict_old[k])

    # print(brand_dict)
    return brand_dict


def set_readme(progress_dict, count):
    # file_path = 'README_test.md'
    file_path = 'README.md'
    # new record
    s = '\nset README.md, add statement * {}\n'.format(str(count))
    log(s, 0)    # all records now
    count_now = sum([len(progress_dict[k]) for k in progress_dict])
    title_now = TITLE.format(count_now)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(title_now + '\n')
        for diff in progress_dict:
            diff_sep = DIFF_SEP.format(diff)
            f.write(diff_sep + '\n')
            for statement in progress_dict[diff]:
                f.write(statement + '\n')
    log('set README.md Done', 2)


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
    if diff == '空':
        res = name
    else:
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
    res = [LANG[lang] if lang in langs else lang for lang in langs]
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
    # print(lang)
    diff = info_dict['diff']
    num = info_dict['num']
    name = make_name(num, info_dict['name'], diff)
    num = make_num(num, ' ')
    # print(name, num, lang)

    pattern = PATTERN.format(num, name, lang, diff)
    log(pattern, 2)
    return pattern, diff


def main():
    untrack, add = file_change()
    print(untrack, add)
    # return
    files = untrack + add
    file_path = 'README_test.md'
    if os.path.exists(file_path):
        os.remove(file_path)
    if len(files):
        log('发现文件:\n', 0)
        # print('发现文件:\n')
        for count, file in enumerate(files):
            print(1 + count, ': ' + file)

        ans = input('\n是否提交? y/n\n').lower()
        if ans == 'y':
            progress_dict, progress_count = get_readme()
            # print(progress_count, progress_dict)

            to_append = defaultdict(list)
            for file in files:
                log(file, 3)
                info = get_md_info(file)
                if not info:
                    continue
                # print(info)
                s, diff = make_statement(info)      # all statement, its difficulty
                # print(s, diff)
                to_append[diff].append(s)
            # print(to_append)
            record_curr = update_info(progress_dict, to_append)
            # print(record_curr)
            # print(progress_dict)
            set_readme(record_curr, len(files))
            if untrack:
                # 添加暂存
                add_file(untrack)
            commit(files)
            push()
            log('done', 2)
        else:
            log('提交取消', 1)
            # print('提交取消 ×')
    else:
        log('无文件修改', 2)
        # print('无文件修改 ⚪')

    input('[  ]: 任意键退出\n')
    return


if __name__ == '__main__':
    main()

    # change_md()
    # l = get_md_info(r'D:\Charge\444\中等\0222 完全二叉树的节点个数.md')
    # print(l)
    # s = make_statement(l)
    # print(s)
