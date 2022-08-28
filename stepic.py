# -*- coding: utf-8 -*-
import numpy
import copy


def card_calc():
    fst_card, sec_card = input().replace('J', '11').replace('Q', '12').replace('K', '13').replace('A', '14').split()
    trump = input()
    fst_card_val = fst_card[:-1]
    fst_card_suit = fst_card[-1]
    sec_card_val = sec_card[:-1]
    sec_card_suit = sec_card[-1]
    if fst_card_suit == sec_card_suit:
        if fst_card_val > sec_card_val:
            return 'First'
        elif fst_card_val < sec_card_val:
            return 'Second'
        else:
            return 'Error'
    elif trump == fst_card_suit:
        return 'First'
    elif trump == sec_card_suit:
        return 'Second'
    else:
        return 'Error'


def miner():
    pole = '4 4'
    mines = """..*.
**..
..*.
...."""
    n, m = map(int, pole.split())
    ish_mines = mines
    mines = mines.replace('.', '0').replace('*', '1').split('\n')
    mines.insert(0, '0' * (m + 2))
    mines.append('0' * (m + 2))
    preobr_mines = []
    i = 0
    while i < len(mines) - 0:
        if 0 < i < len(mines) - 1:
            mines[i] = '0' + mines[i] + '0'
        mines[i] = list(map(int, list(mines[i])))
        i += 1
    for j in range(1, m + 1):
        preobr_mines.append([])
        for i in range(1, n + 1):
            preobr_mines[j - 1].append(
                sum(mines[j - 1][i - 1:i + 2] + mines[j][i - 1:i + 2] + mines[j + 1][i - 1:i + 2]))
        preobr_mines[j - 1] = ''.join(map(str, preobr_mines[j - 1]))
    preobr_mines = '\n'.join(preobr_mines)

    for i in range(len(ish_mines)):
        if ish_mines[i] == '.':
            ish_mines = ish_mines[:i] + preobr_mines[i] + ish_mines[i + 1:]
    return ish_mines


def len_stats():
    mystring = input().split()
    result = {len(i): 0 for i in mystring}
    for i in mystring:
        result[len(i)] += 1
    for i in sorted(result):
        print(f'{i}: {result[i]}')


def vivod_povtorov():
    stroka = '4 8 0 3 4 2 0 3'
    stroka = list(stroka.split())
    result = []
    for i in stroka:
        if stroka.count(i) > 1:
            result.append(i)
    print(' '.join(sorted(set(result))))


def modify_list(lst):
    lst = [1, 2, 3, 4, 5, 6]
    i = 0
    while i < len(lst):
        if lst[i] % 2 > 0:
            lst.pop(i)
            # if i > 0:
            #     i -= 1
        else:
            lst[i] = int(lst[i] / 2)
            i += 1


def find_pos():
    list = '5 8 2 7 8 8 2 4'
    what_find = '8'
    result = []
    for i, v in enumerate(list.split()):
        if v == what_find:
            result.append(str(i))
    print(' '.join(result))


def posledovat_chisel():
    n = int(input())
    result = ' '.join([' '.join(map(str, [i] * i)) for i in range(n + 1)[1:]])[:n * 2 - 1]
    print(result)


def spiral_matrix():
    n = 30
    matrix = numpy.zeros((n, n), int)

    number = 1
    for x in range(n):
        for y in range(x, n - x):
            matrix[x][y] = number
            number += 1
        matrix = numpy.rot90(matrix)
        for k in range(2):
            for y in range(x + 1, n - x):
                matrix[x][y] = number
                number += 1
            matrix = numpy.rot90(matrix)
        for y in range(x + 1, n - x - 1):
            matrix[x][y] = number
            number += 1
        matrix = numpy.rot90(matrix)
    matrix.astype(int)
    for k in matrix:
        print(' '.join(map(str, k)))


def converter():
    info = {'mile': 1609,
            'yard': 0.9144,
            'foot': 0.3048,
            'inch': 0.0254,
            'km': 1000,
            'm': 1,
            'cm': 0.01,
            'mm': 0.001}
    perevod = input()
    value, from_whitch, nona, to_whitch = perevod.split()
    return_value = float(value) * info[from_whitch] / info[to_whitch]
    print("{:.2e}".format(return_value))


def find_indexes():
    where_find = "aaaa"
    what_find = "aa"
    if what_find not in where_find:
        return -1
    return ' '.join(map(str, [i for i in range(len(where_find)) if where_find[i:i + len(what_find)] == what_find]))


def decToRoman():
    coding = zip(
        [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1],
        ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    )
    num = int(input())
    if num <= 0 or num >= 4000 or int(num) != num:
        raise ValueError('Input should be an integer between 1 and 3999')
    result = []
    for d, r in coding:
        while num >= d:
            result.append(r)
            num -= d
    return ''.join(result)


def anagrams():
    sl1 = 'AbaCa'.lower()
    sl2 = 'AcaBa'.lower()
    if sorted(sl1) == sorted(sl2):
        print('Anagram!')
    else:
        print('Not anagram!')


def jolly_jumper():
    # put your python code here

    l = input()
    # l = '4'
    l = list(map(int, l.split()))

    # for i in range(len(l) - 1):
    n = len(l) - 1
    somelist = [l[i] - l[i + 1] for i in range(n)]
    if (len(somelist) > 0 and len(somelist) == len(sorted(somelist)) and min(somelist) > -1 * n and max(
            somelist) <= n) or len(somelist) == 0:
        print('Jolly')
    else:
        print('Not jolly')


def lcd_calc():
    a1 = ' -- '
    a2 = '|  |'
    a3 = '    '
    a4 = '   |'
    a5 = '|   '

    d = {'0': [a1, a2, a2, a3, a2, a2, a1],
         '1': [a3, a4, a4, a3, a4, a4, a3],
         '2': [a1, a4, a4, a1, a5, a5, a1],
         '3': [a1, a4, a4, a1, a4, a4, a1],
         '4': [a3, a2, a2, a1, a4, a4, a3],
         '5': [a1, a5, a5, a1, a4, a4, a1],
         '6': [a1, a5, a5, a1, a2, a2, a1],
         '7': [a1, a4, a4, a3, a4, a4, a3],
         '8': [a1, a2, a2, a1, a2, a2, a1],
         '9': [a1, a2, a2, a1, a4, a4, a1]}
    nums = '024'
    first_line = f'x{"-" * (len(nums) * 4 + len(nums) - 1)}x'
    print(first_line)
    for line in range(len(d['0'])):
        vivod_line = '|'
        for num in nums:
            vivod_line += f'{d[num][line]} '
        print(f'{vivod_line[:-1]}|')
        # print('')
    print(first_line)


def game_of_life():
    pole = '5 5'
    life_or_not = """.....
..X..
...X.
.XXX.
....X"""
    n, m = map(int, pole.split())
    ish_mines = life_or_not
    life_or_not = life_or_not.replace('.', '0').replace('X', '1').split('\n')
    for i in range(len(life_or_not)):
        life_or_not[i] = list(map(int, list(life_or_not[i])))

    new_life = copy.deepcopy(life_or_not)

    for y in range(n):
        for x in range(m):
            s = return_kol_sosed(life_or_not, x, y)
            new_life[y][x] = s

    for y in range(len(new_life)):
        for x in range(len(new_life[y])):
            if (life_or_not[y][x] == 0 and new_life[y][x] == 3) or (
                    life_or_not[y][x] == 1 and new_life[y][x] in [2, 3]):
                life_or_not[y][x] = 1
            elif new_life[y][x] not in [2, 3]:
                life_or_not[y][x] = 0

    itog = ''
    for i in life_or_not:
        itog += f"{''.join(map(str, i))}" + "\n"
    itog = itog.replace('0', '.').replace('1', 'X')

    return itog


def return_kol_sosed(matrix, x, y):
    x_plus_1, y_plus_1 = x + 1, y + 1
    if x + 1 == len(matrix[0]):
        x_plus_1 = 0
    if y + 1 == len(matrix):
        y_plus_1 = 0
    return sum([matrix[y - 1][x - 1], matrix[y - 1][x], matrix[y - 1][x_plus_1], matrix[y][x - 1],
                matrix[y][x_plus_1], matrix[y_plus_1][x - 1], matrix[y_plus_1][x], matrix[y_plus_1][x_plus_1]])


def line_of_Koh():
    n = 2  # int(input())
    if n > 0:
        itog_list = [60, -120, 60]
    for i in range(n - 1):
        itog_list = ('60, -120, 60, ' + ', 60, -120, 60, '.join(map(str, itog_list)) + ', 60, -120, 60').split(', ')

    for i in itog_list:
        print(f'turn {i}')


def fizz_buzz(i):
    if i % 15 == 0:
        return 'FizzBuzz'
    if i % 5 == 0:
        return 'Buzz'
    if i % 3 == 0:
        return 'Fizz'
    return str(i)


# n, m = map(int, '8 16'.split(' '))
# for i in range(n, m + 1):
#     print(fizz_buzz(i))


def continued_fraction():
    nums = '239/30'
    a, b = map(int, nums.split('/'))
    itog_list = []
    while b != 0:
        itog_list.append(a // b)
        a, b = b, a % b
    print(' '.join(map(str, itog_list)))


def transpose():
    n, m = map(int, '2 3'.split())
    lines = """1 2 3
4 5 6"""
    # matrix = list(map(list, lines.splitlines()))
    matrix = []
    for line in lines.splitlines():
        matrix.append([int(i) for i in line.split(' ')])
    new_matrix = []
    for i in zip(*matrix):
        new_matrix.append(list(i))
        print(' '.join(map(str, i)))
    # print(new_matrix)


# transpose()

def hanoy():
    n = int(input())
    t1, buf, t2 = 1, 2, 3
    moves = []
    m1, m2, m3 = [], [], []
    for i in range(n, 0, -1):
        m1.append(i)
    need_to_be = m1.copy()
    while m3 != need_to_be:
        if n % 2 != 0:
            if len(m1) > 0 and (len(m3) == 0 or m3[-1] > m1[-1]):
                m3.append(m1.pop(-1))
                moves.append('1 - 3')
            elif len(m3) > 0 and (len(m1) == 0 or m3[-1] < m1[-1]):
                m1.append(m3.pop(-1))
                moves.append('3 - 1')
            if m3 == need_to_be:
                continue

            if len(m1) > 0 and (len(m2) == 0 or m2[-1] > m1[-1]):
                m2.append(m1.pop(-1))
                moves.append('1 - 2')
            elif len(m2) > 0 and (len(m1) == 0 or m2[-1] < m1[-1]):
                m1.append(m2.pop(-1))
                moves.append('2 - 1')
            if m3 == need_to_be:
                continue

            if len(m2) > 0 and (len(m3) == 0 or m3[-1] > m2[-1]):
                m3.append(m2.pop(-1))
                moves.append('2 - 3')
            elif len(m3) > 0 and (len(m2) == 0 or m3[-1] < m2[-1]):
                m2.append(m3.pop(-1))
                moves.append('3 - 2')
        else:
            if len(m1) > 0 and (len(m2) == 0 or m2[-1] > m1[-1]):
                m2.append(m1.pop(-1))
                moves.append('1 - 2')
            elif len(m2) > 0 and (len(m1) == 0 or m2[-1] < m1[-1]):
                m1.append(m2.pop(-1))
                moves.append('2 - 1')
            if m3 == need_to_be:
                continue

            if len(m1) > 0 and (len(m3) == 0 or m3[-1] > m1[-1]):
                m3.append(m1.pop(-1))
                moves.append('1 - 3')
            elif len(m3) > 0 and (len(m1) == 0 or m3[-1] < m1[-1]):
                m1.append(m3.pop(-1))
                moves.append('3 - 1')
            if m3 == need_to_be:
                continue

            if len(m2) > 0 and (len(m3) == 0 or m3[-1] > m2[-1]):
                m3.append(m2.pop(-1))
                moves.append('2 - 3')
            elif len(m3) > 0 and (len(m2) == 0 or m3[-1] < m2[-1]):
                m2.append(m3.pop(-1))
                moves.append('3 - 2')

    print('\n'.join(moves))


def poker():
    cards = '10C JC QC KC AC'.replace('J', '11').replace('Q', '12').replace('K', '13').replace('A', '14')
    # cards_tuple = cards.split()
    spades = [i[-1] for i in cards.split()]
    cards = [int(i[:-1]) for i in cards.split()]
    flush = func_flush(spades)
    street = func_street(cards)
    pairs_thirds = func_pairs_thirds(cards)
    if flush and street and 14 in cards:
        return 'Royal Flush'
    elif flush and street:
        return 'Straight Flush'
    elif flush:
        return 'Flush'
    elif street:
        return 'Straight'
    return pairs_thirds

    pass


def func_pairs_thirds(cards):
    cards_nums = [cards.count(i) for i in cards]
    if cards_nums.count(2) == 4:
        return 'Two Pairs'
    if 4 in cards_nums:
        return 'Four of a Kind'
    if 2 in cards_nums and 3 in cards_nums:
        return 'Full House'
    if 3 in cards_nums:
        return 'Three of a Kind'
    if 2 in cards_nums:
        return 'Pair'
    return 'High Card'
    pass


def func_flush(cards):
    if len(set(cards)) == 1:
        return True
    return False


def func_street(cards):
    cards = sorted(set(cards))
    if len(cards) == 5 and cards[-1] - cards[0] == 4:
        return True
    return False


# print(poker())

def amerik_zpt():
    number = input()
    number = number[::-1]
    n = 3
    text = [number[i:i + n] for i in range(0, len(number), n)]
    print(','.join(text)[::-1])


# amerik_zpt()

def iosif_flavii():
    n, k = int(input()), int(input())
    peoples = list(range(1, n + 1))
    i = 1
    abs_num = 1
    while len(peoples) > 1:
        if abs_num % k == 0:
            peoples.pop(i - 1)
        else:
            i += 1
        if i > len(peoples):
            i = 1
        abs_num += 1
    print(peoples[0])


def iosif_flaviy_2():
    n = int(input())
    k = int(input())

    res = 0
    for i in range(1, n + 1):
        res = (res + k) % i
    print(res + 1)


# iosif_flaviy_2()
# iosif_flavii()

def chetvert():
    n = int('4')
    points = ['0 -1', '1 2', '0 9', '-9 -5']
    points_pb = []
    for i in range(n):
        points_pb.append(tuple(map(int, points[i].split())))
    chet = [0, 0, 0, 0]
    for x, y in points_pb:
        if x > 0:
            if y > 0:
                chet[0] += 1
            elif y < 0:
                chet[3] += 1
        elif x < 0:
            if y > 0:
                chet[1] += 1
            elif y < 0:
                chet[2] += 1
    names = ['Первая', 'Вторая', 'Третья', 'Четвертая']
    for i in range(4):
        print(f'{names[i]} четверть: {chet[i]}')

    pass


# chetvert()

def uvelichen_list():
    nums_list = list(map(int, '1 1 3 2 2 1 1 1 1'.split()))
    nums = 0
    for i in range(1, len(nums_list)):
        if nums_list[i] > nums_list[i - 1]:
            nums += 1
    print(nums)


# uvelichen_list()

def sdvig_stroki():
    stroka = '1 2 3 4 5'.split()
    sdvig_stroka = [stroka[-1]] + stroka[:-1]
    print(' '.join(sdvig_stroka))

    # sdvig_stroka = stroka.insert(0, stroka.pop(-1))
    # print(sdvig_stroka)


# sdvig_stroki()

def podschet_el():
    stroka = '1 1 1 2 2 2 2 3 3 3'
    print(len(set('1 1 1 2 2 2 2 3 3 3'.split())))


# podschet_el()

def proverka_na_mnozhitel():
    n = 3
    nums = list(map(int, '''33
33
17
35'''.splitlines()))
    summa = 999
    for i in range(len(nums)):
        for k in range(len(nums)):
            if i != k and nums[i] * nums[k] == summa:
                return "ДА"
    return "НЕТ"


# print(proverka_na_mnozhitel())

def kamen_nozh_bum():
    timur = ['камень-ножницы', 'ножницы-бумага', 'бумага-камень']
    first = input()
    second = input()
    if first == second:
        exit(print('ничья'))
    round = f"{first}-{second}"
    if round in timur:
        exit(print('Тимур'))
    print("Руслан")


# kamen_nozh_bum()

def super_kamen():
    graf = {'ножницы': ['бумага', 'ящерица'], 'бумага': ['камень', 'Спок'], 'камень': ['ножницы', 'ящерица'],
            'ящерица': ['Спок', 'бумага'], 'Спок': ['ножницы', 'камень']}
    timur = input()
    ruslan = input()
    if timur == ruslan:
        exit(print('ничья'))
    if ruslan in graf[timur]:
        exit(print("Тимур"))
    print('Руслан')


# super_kamen()

def orel_reshka():
    stroka = input().split('О')
    max_o = max(map(len, stroka))
    print(max_o)


# orel_reshka()

def holodos():
    l = int(input())
    shifr = 'anton'
    holod_nums = list(range(1, l + 1))
    stroki = """0000a0000n00t00000o000000n
gylfole
richard
ant0n""".splitlines()
    n = 1
    for i in stroki:
        for bukv in shifr:
            if bukv in i:
                i = i[i.find(bukv) + 1:]
            else:
                holod_nums.remove(n)
                break
        n += 1
    print(*zip(map(str, holod_nums)))


# holodos()

def pesenka():
    b = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х',
         'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
    slovo = input()
    message = f"{slovo} запретил букуву"
    for sym in b:
        if sym in message:
            message += f" {sym}"
            print(message)
            message = message.replace(sym, '').replace('  ', ' ').strip()


# pesenka()

def pascal():
    n = int(input())
    pascal_list = [[1]]
    for i in range(1, n + 1):
        pascal_list.append([1])
        for j in range(i - 1):
            pascal_list[i].append(pascal_list[i - 1][j] + pascal_list[i - 1][j + 1])
        pascal_list[i].append(1)
    print(pascal_list[n])


# pascal()

def upak_center():
    word = 'a b c d'.split()
    vivod_list = [[word[0]]]
    for i in range(1, len(word)):
        if vivod_list[-1][-1] == word[i]:
            vivod_list[-1].append(word[i])
        else:
            vivod_list.append([word[i]])
    print(vivod_list)


# upak_center()

def chunked():
    list1 = 'a b c d e f r g b'.split()
    n = int('2')
    vivod_list = []
    for i in range(0, len(list1), n):
        vivod_list.append(list1[i:i + n])
    print(vivod_list)


# chunked()

def podspisok():
    # list1 = input().split()
    list1 = 's t e p i k r o c k s'.split()
    vivod_list = [[]]
    for i in range(len(list1)):
        for j in range(len(list1)):
            dop = list1[j:j + i + 1]
            if len(dop) == i + 1:
                vivod_list.append(dop)
    print(vivod_list)


# podspisok()


def vvod_matrix():
    n = int(input())
    matrix = []
    for i in range(n):
        temp = [int(num) for num in input().split()]
        matrix.append(temp)
    return matrix


def print_matrix(matrix, n=0, width=1):
    if n == 0:
        n = len(matrix[0])
    for r in range(n):
        for c in range(n):
            print(str(matrix[r][c]).ljust(width), end=' ')
        print()


def vvod_vivod_matrix():
    rows = int(input())
    cols = int(input())

    matrix = [[0] * cols for _ in range(rows)]

    words = """язык
болтает
а
голова
не
знает""".splitlines()
    k = 0
    for i in range(rows):
        for j in range(cols):
            matrix[i][j] = words[k]
            k += 1

    for i in range(rows):
        for j in range(cols):
            print(matrix[i][j], end=' ')
        print()
    print()
    for i in range(cols):
        for j in range(rows):
            print(matrix[j][i], end=' ')
        print()


# vvod_vivod_matrix()

def sled_matrix():
    n = int(input())
    matrix = []
    for i in range(n):
        matrix.append(list(map(int, input().split())))
    print(sum([matrix[i][i] for i in range(n)]))

    # print(matrix)


# sled_matrix()

def matrix_count_more_mid_arifm():
    n = int(input())
    matrix = []
    for _ in range(n):
        matrix.append(list(map(int, input().split())))

    for i in range(n):
        row_mid = sum(matrix[i]) / len(matrix[i])
        sums = 0
        for j in range(n):
            if matrix[i][j] > row_mid:
                sums += 1
        print(sums)


# matrix_count_more_mid_arifm()

def max_treyg_matrix():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)

    need_max = []
    for i in range(n):
        for j in range(n):
            if i >= j:
                need_max.append(matrix[i][j])
    print(max(need_max))


# max_treyg_matrix()

def max_treyg_matrix():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)

    need_max = []
    for i in range(n):
        for j in range(n):
            if i >= j and i <= n - j - 1 or i <= j and i >= n - j - 1:
                need_max.append(matrix[i][j])

    print(max(need_max))


# max_treyg_matrix()

def four_chetvert():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)

    max_nizh, max_verh, max_left, max_right = 0, 0, 0, 0
    for i in range(n):
        for j in range(n):
            el = matrix[i][j]
            if i > j:
                if i + j + 1 > n and el > max_nizh:
                    max_nizh = el
                elif i + j + 1 < n and el > max_left:
                    max_left = el
            elif i < j:
                if i + j + 1 > n and el > max_right:
                    max_right = el
                elif i + j + 1 < n and el > max_verh:
                    max_verh = el

    print(f'Верхняя четверть: {max_verh}')
    print(f'Правая четверть: {max_right}')
    print(f'Нижняя четверть: {max_nizh}')
    print(f'Левая четверть: {max_left}')


# four_chetvert()

def muliplication():
    n, m = int(input()), int(input())
    mult = [[0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            mult[i][j] = i * j
            print(str(mult[i][j]).ljust(3), end=' ')
        print()


# muliplication()

def max_index_matrix():
    n, m = int(input()), int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)
    # print(matrix)
    max, max_i, max_j = matrix[0][0], 0, 0
    for i in range(n):
        for j in range(m):
            el = matrix[i][j]
            if el > max:
                max_i, max_j, max = i, j, el

    print(max_i, max_j)


def change_cols():
    n, m = int(input()), int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)
    print(matrix)
    n1, n2 = map(int, input().split())
    for i in range(n):
        matrix[i][n1], matrix[i][n2] = matrix[i][n2], matrix[i][n1]
        for j in range(m):
            print(matrix[i][j], end=' ')
        print()


def symmetric_matrix():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)
    # print(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                exit(print('NO'))
    print('YES')


def change_main_sec_diag():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)
    for i in range(n):
        prom = matrix[i][i]
        matrix[i][i], matrix[n - i - 1][i] = matrix[n - i - 1][i], matrix[i][i]

    for i in range(n):
        for j in range(n):
            print(matrix[i][j], end=' ')
        print()
    # print()
    # print_matrix(matrix, n, 1)


def mirror_matrix():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)
    # print_matrix(matrix, n)
    # print()
    for i in range(n // 2):
        matrix[i], matrix[n - i - 1] = matrix[n - i - 1], matrix[i]

    for i in range(n):
        for j in range(n):
            print(matrix[i][j], end=' ')
        print()


def rotate_matrix():
    n = int(input())
    matrix = []
    rotated = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        rotated.append([0] * n)
        matrix.append(row)
    for i in range(n):
        for j in range(n):
            rotated[i][j] = matrix[n - j - 1][i]

    for i in range(n):
        for j in range(n):
            print(rotated[i][j], end=' ')
        print()


def horse_chess():
    xy = input()
    y = int('87654321'.index(xy[1]))
    x = int('abcdefgh'.index(xy[0]))
    n = 8
    matrix = []
    for _ in range(n):
        matrix = [['.'] * n for _ in range(n)]

    matrix[y][x] = 'N'
    print_matrix(matrix)
    for i in range(n):
        for j in range(n):
            if abs(x - j) == 2 and abs(y - i) == 1:
                matrix[i][j] = '*'
            if abs(x - j) == 1 and abs(y - i) == 2:
                matrix[i][j] = '*'


    for i in range(n):
        for j in range(n):
            print(matrix[i][j], end=' ')
        print()


def magic_square():
    n = int(input())
    matrix = []
    numbers_in_matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)
        numbers_in_matrix.extend(row)

    test_list = [i for i in range(1, n**2+1)]
    for i in test_list:
        if i not in numbers_in_matrix:
            exit(print('NO'))
    magic_number = sum(matrix[0])
    for i in range(n):
        if sum(matrix[i]) != magic_number or sum(matrix[j][i] for j in range(n)) != magic_number:
            exit(print('NO'))
    if sum(matrix[i][n - i - 1] for i in range(n)) != magic_number or sum(matrix[i][i] for i in range(n)) != magic_number:
        exit(print('NO'))


    print('YES')

def chess_desk():
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        matrix.append([['.'] for i in range(m)])

    for i in range(n):
        start_sym = '*.'
        if i % 2 == 0:
            start_sym = '.*'
        for j in range(m):
            print(start_sym[j % 2], end=' ')
        print()

def secondary_diag():
    n = int(input())
    matrix = []

    for _ in range(n):
        row = [[0] for i in range(n)]
        matrix.append(row)

    for i in range(n):
        for j in range(n):
            el = 0
            if i == n - j - 1:
                el = 1
            elif i > n - j - 1:
                el = 2
            print(el, end=' ')
        print()

    # print_matrix(matrix)

print('print hello my friend')
secondary_diag()