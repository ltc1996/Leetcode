import os
import re
import fileinput

def remove_blank(string):
    res = ''
    for char in string:
        if char != ' ':
            res += char
    return res

def maintain(filename):
    if not os.path.exists(filename):
        return 
    with fileinput.input(files=filename, openhook=fileinput.hook_encoded("utf-8")) as f:        # inplace=True, 
        last = ''
        for line in f:
            tmp_line = line
            p = re.findall(r'\|.*(\d)+.*\|', tmp_line)
            if p:
                # [滑动谜题](中等/0071%20简化路径.md) 
                # print(tmp_line, tmp_line.split('|'), len(tmp_line.split('|')))
                # if len(tmp_line.split('|')) != 6:
                #     print(tmp_line)
                num, name, lang, diff = tmp_line.split('|')[1: -1]
                num = remove_blank(num)
                new_num = '0' * (4 - len(num)) + num
                name = remove_blank(name)
                new_diff = remove_blank(diff)
                if new_diff == '空':
                    new_diff = '简单'
                new_name = '[' + name + ']' + '(' + new_diff + r'/' + new_num + '%20' + name + '.md)'
                # print(new_name)
                if new_diff != last:
                    print()
                    last = new_diff
                print('|' + '|'.join([
                    ' ' + ' ' * (4 - len(num)) + remove_blank(num) + ' ',
                    ' ' + new_name + ' ',
                    ' ' + lang + ' ',
                    ' ' + diff + ' ',
                ]) + '|'
                )
    

if __name__ == "__main__":
    maintain('readme'.upper() + '.md')