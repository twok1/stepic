# -*- coding: utf-8 -*-
import math
import random
from decimal import Decimal
from fractions import Fraction
from string import ascii_letters, digits, ascii_uppercase, ascii_lowercase

import numpy
from numpy.ma import copy


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
    l_1 = list(map(int, input().split()))

    n = len(l_1) - 1
    somelist = [l_1[i] - l_1[i + 1] for i in range(n)]
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
    n, m = map(int, input().split())
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
    print(len(set(input().split())))


# podschet_el()

def proverka_na_mnozhitel():
    n = 3
    nums = list(map(int, '''33
33
17
35'''.splitlines()))
    summa = 999
    for i in range(n):
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
    what_a_round = f"{first}-{second}"
    if what_a_round in timur:
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
    l_1 = int(input())
    shifr = 'anton'
    holod_nums = list(range(1, l_1 + 1))
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

def max_treyg_matrix_2():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)

    need_max = []
    for i in range(n):
        for j in range(n):
            if j <= i <= n - j - 1 or j >= i >= n - j - 1:
                need_max.append(matrix[i][j])

    print(max(need_max))


# max_treyg_matrix_2()

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
    max_val, max_i, max_j = matrix[0][0], 0, 0
    for i in range(n):
        for j in range(m):
            el = matrix[i][j]
            if el > max_val:
                max_i, max_j, max_val = i, j, el

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

    test_list = [i for i in range(1, n ** 2 + 1)]
    for i in test_list:
        if i not in numbers_in_matrix:
            exit(print('NO'))
    magic_number = sum(matrix[0])
    for i in range(n):
        if sum(matrix[i]) != magic_number or sum(matrix[j][i] for j in range(n)) != magic_number:
            exit(print('NO'))
    if sum(matrix[i][n - i - 1] for i in range(n)) != magic_number or sum(
            matrix[i][i] for i in range(n)) != magic_number:
        exit(print('NO'))

    print('YES')


def chess_desk():
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        matrix.append([['.'] for _ in range(m)])

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
        row = [[0] for _ in range(n)]
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


def n_to_m_matrix_by_rows():
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        matrix.append([[0] for _ in range(m)])

    num = 1
    for i in range(n):
        for j in range(m):
            matrix[i][j] = num
            num += 1
            print(str(matrix[i][j]).ljust(3), end=' ')
        print()


def n_to_m_matrix_by_cols():
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        matrix.append([[0] for _ in range(m)])

    num = 1
    for i in range(m):
        for j in range(n):
            matrix[j][i] = num
            num += 1

    for i in range(n):
        for j in range(m):
            print(str(matrix[i][j]).ljust(3), end=' ')
        print()


def primary_secondary_diag():
    n = int(input())
    matrix = []

    for _ in range(n):
        row = [[0] for _ in range(n)]
        matrix.append(row)

    for i in range(n):
        for j in range(n):
            el = 0
            if i == n - j - 1 or i == j:
                el = 1
            print(str(el).ljust(3), end=' ')
        print()


def primary_secondary_diag_with_treug():
    n = int(input())
    matrix = []

    for _ in range(n):
        row = [[0] for _ in range(n)]
        matrix.append(row)

    for i in range(n):
        for j in range(n):
            el = 0
            if i <= n - j - 1 and i <= j:
                el = 1
            if i >= n - j - 1 and i >= j:
                el = 1
            print(str(el).ljust(3), end=' ')
        print()


def beautiful_matrix():
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        matrix.append([[0] for _ in range(m)])
    for i in range(n):
        for j in range(m):
            matrix[i][j] = (i + j) % m + 1
            print(str(matrix[i][j]).ljust(3), end=' ')
        print()


def zmeika_matrix():
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        matrix.append([0 for _ in range(m)])
    el = 1
    for i in range(n):
        for j in range(m):
            matrix[i][j] = el
            el += 1
        if i % 2:
            matrix[i].reverse()

    for i in range(n):
        for j in range(m):
            print(str(matrix[i][j]), end=' ')
        print()


def diag_matrix_beautiful():
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        matrix.append([0 for _ in range(m)])
    el = 1
    mn = 0
    for q in range(n * m + 1):
        for i in range(n):
            for j in range(m):
                if i + j == mn:
                    matrix[i][j] = el
                    el += 1
        mn += 1

    for i in range(n):
        for j in range(m):
            print(str(matrix[i][j]), end=' ')
        print()


def spiral_matrix_2():
    n, m = map(int, input().split())
    matrix = []
    for _ in range(n):
        matrix.append([0 for _ in range(m)])
    directions = ['r', 'd', 'l', 'u']
    el = 1
    i, j = 0, 0
    direction = 'r'
    while el <= m * n:
        if i < n and j < m and matrix[i][j] == 0:
            matrix[i][j] = el
            if direction == 'r':
                j += 1
            elif direction == 'd':
                i += 1
            elif direction == 'l':
                j -= 1
            elif direction == 'u':
                i -= 1
            el += 1
        elif direction == 'r':
            direction = directions[(directions.index(direction) + 1) % len(directions)]
            i += 1
            j -= 1
        elif direction == 'd':
            direction = directions[(directions.index(direction) + 1) % len(directions)]
            j -= 1
            i -= 1
        elif direction == 'l':
            direction = directions[(directions.index(direction) + 1) % len(directions)]
            i -= 1
            j += 1
        elif direction == 'u':
            direction = directions[(directions.index(direction) + 1) % len(directions)]
            j += 1
            i += 1

    for row in matrix:
        print(*row)


def sum_of_matrix():
    n, m = map(int, input().split())
    matrix1 = []
    sum_matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix1.append(row)
        sum_matrix.append([0 for _ in range(m)])
    input()
    matrix2 = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix2.append(row)

    for i in range(n):
        for j in range(m):
            sum_matrix[i][j] = matrix1[i][j] + matrix2[i][j]
            print(sum_matrix[i][j], end=' ')
        print()


def multiply_matrix():
    n, m = map(int, input().split())
    matrix1 = []
    res_matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix1.append(row)
    input()

    m, k = map(int, input().split())
    matrix2 = []
    for _ in range(m):
        row = [int(i) for i in input().split()]
        matrix2.append(row)
    for _ in range(n):
        res_matrix.append([0 for _ in range(k)])

    for i in range(n):
        for j in range(k):
            for z in range(m):
                res_matrix[i][j] += matrix1[i][z] * matrix2[z][j]

    for i in range(n):
        for j in range(k):
            print(res_matrix[i][j], end=' ')
        print()


def degree_matrix():
    n = int(input())
    matrix1 = []
    res_matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix1.append(row)
        res_matrix.append([0 for _ in range(n)])
    m = int(input())

    for step in range(m - 1):
        now_matrix = copy.deepcopy(res_matrix)
        res_matrix = []
        for _ in range(n):
            res_matrix.append([0 for _ in range(n)])
        for i in range(n):
            for j in range(n):
                for z in range(n):
                    if step == 0:
                        res_matrix[i][j] += matrix1[i][z] * matrix1[z][j]
                    else:
                        res_matrix[i][j] += now_matrix[i][z] * matrix1[z][j]

    for i in range(n):
        for j in range(n):
            print(res_matrix[i][j], end=' ')
        print()


def stroka_to_list():
    list1 = input().split()
    m = int(input())
    res_list = [[] for _ in range(m)]
    for i in range(len(list1)):
        res_list[i % m].append(list1[i])
    print(res_list)


def max_from_treug():
    n = int(input())
    matrix = []

    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)

    max_el = 0
    for i in range(n):
        for j in range(n):
            if i >= n - j - 1 and matrix[i][j] > max_el:
                max_el = matrix[i][j]
            print(str(matrix[i][j]).ljust(3), end=' ')
        print()
    print(max_el)


def transpose2():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)

    new_matrix = []
    for i in zip(*matrix):
        new_matrix.append(list(i))
        print(' '.join(map(str, i)))


# transpose2()

def snezhinka():
    n = int(input())
    matrix = []

    for _ in range(n):
        row = ['.' for _ in range(n)]
        matrix.append(row)

    for i in range(n):
        for j in range(n):
            el = '.'
            if i == n - j - 1 or i == j or i == (n - 1) / 2 or j == (n - 1) / 2:
                el = '*'
            print(str(el).ljust(3), end=' ')
        print()


# snezhinka()

def symmetric_matrix_secondary():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)
    # print(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[n - j - 1][n - i - 1]:
                exit(print('NO'))
    print('YES')


# symmetric_matrix_secondary()


def latin_squad():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [int(i) for i in input().split()]
        matrix.append(row)

    new_matrix = []

    for i in zip(*matrix):
        new_matrix.append(list(i))
    for i in range(n):
        for k in range(1, n + 1):
            if k not in matrix[i] or k not in new_matrix[i]:
                exit(print('NO'))
    print('YES')


# latin_squad()


def queen_chess():
    xy = input()
    y = int('87654321'.index(xy[1]))
    x = int('abcdefgh'.index(xy[0]))
    n = 8
    matrix = []
    for _ in range(n):
        matrix = [['.'] * n for _ in range(n)]

    # print_matrix(matrix)
    for i in range(n):
        for j in range(n):
            if abs(j - x) == abs(i - y):
                matrix[i][j] = '*'
            if x == j or y == i:
                matrix[i][j] = '*'

    matrix[y][x] = 'Q'
    for i in range(n):
        for j in range(n):
            print(matrix[i][j], end=' ')
        print()


# queen_chess()

def main_diag_plus_one():
    n = int(input())
    matrix = []
    for _ in range(n):
        row = [0 * n for _ in range(n)]
        matrix.append(row)

    for i in range(n):
        for j in range(n):
            matrix[i][j] = abs(j - i)
            print(matrix[i][j], end=' ')
        print()


# main_diag_plus_one()

def non_empty_tuples():
    tuples = [(), (), ('',), ('a', 'b'), (), ('a', 'b', 'c'), (1,), (), (), ('d',), ('', ''), ()]
    non_empty_tuples = [i for i in tuples if i]

    print(non_empty_tuples)


# non_empty_tuples()

def convert_list_to_tuple():
    tuples = [(10, 20, 40), (40, 50, 60), (70, 80, 90), (10, 90), (1, 2, 3, 4), (5, 6, 10, 2, 1, 77)]
    new_tuples = tuple([(*i[:-1], 100) for i in tuples])
    print(new_tuples)


# convert_list_to_tuple()

# poet_data = ('Пушкин', 1799, 'Санкт-Петербург')
# poet_data = tuple((*poet_data[:-1], "Москва"))
# print(poet_data)


def verh_parabola():
    a = int(input())
    b = int(input())
    c = int(input())

    x = -b / (2 * a)
    y = (4 * a * c - b ** 2) / (4 * a)
    print((x, y))


# verh_parabola()

def print_ranks():
    m = int(input())

    ranks = []
    for _ in range(m):
        i = input().split()
        row = (i[0], int(i[1]))
        ranks.append(row)
        print(*row)
    print()
    for i in ranks:
        if i[1] in [4, 5]:
            print(*i)


# print_ranks()
def tribonachi():
    n = int(input())
    trib = [1, 1, 1]
    for i in range(3, n):
        trib.append(sum(trib[i - 3:i + 1]))
    print(*trib[:n])


# tribonachi()

def mnozhestvo():
    n, m, k, x, y, z = int(input()), int(input()), int(input()), int(input()), int(input()), int(input()),
    t = n + m + k + z - x - y
    print(t)


# mnozhestvo()


def mnozhestvo_10_klass():
    n, m, k, x, y, z, t, a = int(input()), int(input()), int(input()), int(input()), int(input()), int(input()), int(
        input()), int(input())

    x_nm = (m + n - x - t)
    x_mk = (m + k - y - t)
    x_nk = (n + k - z - t)
    m1 = m - x_mk - x_nm - t
    k1 = k - x_mk - x_nk - t
    n1 = n - x_nk - x_nm - t
    print(m1 + k1 + n1)
    print(x_nm + x_mk + x_nk)
    print(a - (m1 + k1 + n1 + x_nm + x_mk + x_nk + t))


# mnozhestvo_10_klass()

def unique_symbols():
    length = []
    for _ in range(int(input())):
        length.append(len(set(input().lower())))
    print('\n'.join(map(str, length)))


# unique_symbols()

def mass_unique_symbols():
    length_str = ''
    for _ in range(int(input())):
        length_str += input().lower()
    print(len(set(length_str)))


# mass_unique_symbols()

def unique_words():
    del_syms = '.,;:-?!'
    sentence = input().lower()
    for sym in del_syms:
        sentence = sentence.replace(sym, '')
    print(len(set(sentence.split())))


# unique_words()

def unique_numbers():
    stdin = list(map(int, input().split()))
    was_set = set()
    for i in stdin:
        if i in was_set:
            print('YES')
        else:
            print('NO')
        was_set.add(i)


# unique_numbers()

def length_all_numbers():
    num1, num2 = set(input().split()), set(input().split())
    print(len(num1 & num2))


# length_all_numbers()

def length_interc_numbers():
    num1, num2 = set([int(i) for i in input().split()]), set([int(i) for i in input().split()])
    print(*sorted(num1 & num2))


# length_interc_numbers()

def length_diff_numbers():
    num1, num2 = set([int(i) for i in input().split()]), set([int(i) for i in input().split()])
    print(*sorted(num1 - num2))


# length_diff_numbers()

def intersection_numbers():
    i = int(input())
    itog_set = set(map(int, input()))
    for _ in range(i - 1):
        itog_set.intersection_update(map(int, set(input())))
    print(*sorted(itog_set))


# intersection_numbers()

def check_for_disjoint():
    set1 = set(input())
    set2 = set(input())
    if set1.isdisjoint(set2):
        print('NO')
    else:
        print('YES')


# check_for_disjoint()

def check_for_subset():
    if set(input()).issuperset(set(input())):
        print('YES')
    else:
        print('NO')


# check_for_subset()
def sravn_ranks():
    st1 = set(map(int, input().split()))
    st2 = set(map(int, input().split()))
    st3 = set(map(int, input().split()))
    print(*sorted((st1 & st2) - st3, reverse=True))


# sravn_ranks()
def sravn_ranks_no_more_that_two():
    st1 = set(map(int, input().split()))
    st2 = set(map(int, input().split()))
    st3 = set(map(int, input().split()))
    print(*sorted((st1 | st2) - st3 | (st2 | st3) - st1 | (st1 | st3) - st2))


# sravn_ranks_no_more_that_two()
def sravn_ranks_of_third_without_first_two():
    st1 = set(map(int, input().split()))
    st2 = set(map(int, input().split()))
    st3 = set(map(int, input().split()))
    print(*sorted(st3 - (st1 | st2)))


# sravn_ranks_of_third_without_first_two()

def ne_poluchil_nikto():
    st1 = set(map(int, input().split()))
    st2 = set(map(int, input().split()))
    st3 = set(map(int, input().split()))
    print(*sorted(set(range(11)) - st1 - st2 - st3))


# ne_poluchil_nikto()

def razbor_sentence():
    sentence = '''My very photogenic mother died in a freak accident (picnic, lightning) when I was three, and, 
    save for a pocket of warmth in the darkest past, nothing of her subsists within the hollows and dells of memory, 
    over which, if you can still stand my style (I am writing under observation), the sun of my infancy had set: 
    surely, you all know those redolent remnants of day suspended, with the midges, about some hedge in bloom or 
    suddenly entered and traversed by the rambler, at the bottom of a hill, in the summer dusk; a furry warmth, 
    golden midges. '''
    for i in '.,;:-?!)(':
        sentence = sentence.replace(i, '')
    sentence = sentence.lower().split()
    unique_sent = sorted(set(sentence))
    print(*unique_sent)


def razbor_sentence_2():
    files = ['python.png', 'qwerty.py', 'stepik.png', 'beegeek.org', 'windows.pnp', 'pen.txt', 'phone.py', 'book.txT',
             'board.pNg', 'keyBoard.jpg', 'Python.PNg', 'apple.jpeg', 'png.png', 'input.tXt', 'split.pop',
             'solution.Py', 'stepik.org', 'kotlin.ko', 'github.git']
    print(*sorted({i.lower() for i in files if i.lower()[i.find('.') + 1:] == 'png'}))


# razbor_sentence_2()

def home_work_school():
    n = int(input())
    m = int(input())
    k = int(input())
    p = int(input())
    print(n - m - k + p)


# home_work_school()

def voshod_func():
    i = input().split()
    print(len(i) - len(set(i)))


# voshod_func()

def goroda():
    goroda = set()
    for _ in range(int(input())):
        goroda.add(input())
    if input() in goroda:
        exit(print('REPEAT'))
    print('OK')


# goroda()
def books_list():
    n = int(input())
    m = int(input())
    have_books = set()
    for _ in range(n):
        have_books.add(input())
    for _ in range(m):
        if input() in have_books:
            print('YES')
        else:
            print('NO')


# books_list()

def bad_day_good_day():
    set1 = set(input().split())
    set2 = set(input().split())
    if set1 & set2:
        print(*sorted(map(int, set1 & set2), reverse=True))
    else:
        print('BAD DAY')


# bad_day_good_day()
def test_candidate():
    if set(input().split()) == set(input().split()):
        exit(print('YES'))
    print('NO')


# test_candidate()
def beegeek_inf_math():
    m = int(input())
    n = int(input())
    math_set = set()
    inf_set = set()
    for _ in range(m):
        math_set.add(input())
    for _ in range(n):
        inf_set.add(input())
    # print(len(math_set - inf_set))
    if math_set ^ inf_set:
        print(len(math_set ^ inf_set))
    else:
        print('NO')


# beegeek_inf_math()
def beegeek_lesson_5():
    m = int(input())
    n = int(input())
    all_set = set()
    for _ in range(m + n):
        all_set.add(input())
    two_less = m + n - 2 * (m + n - len(all_set))
    if two_less:
        print(two_less)
    else:
        print('NO')


# beegeek_lesson_5()

def beegeek_was_on_all_lessons():
    m = int(input())
    all_lessons = set()
    for n in range(m):
        i = int(input())
        one_lesson = []
        for _ in range(i):
            k = input()
            if n == 0:
                all_lessons.add(k)
                one_lesson.append(k)
            else:
                one_lesson.append(k)
        all_lessons.intersection_update(one_lesson)
    print(*sorted(all_lessons), sep='\n')
    pass


# beegeek_was_on_all_lessons()
def number_to_string():
    conver = {'0': 'zero',
              '1': 'one',
              '2': 'two',
              '3': 'three',
              '4': 'four',
              '5': 'five',
              '6': 'six',
              '7': 'seven',
              '8': 'eight',
              '9': 'nine'}
    itog_num = []
    num = input()
    for i in num:
        itog_num.append(conver[i])
    print(*itog_num)


# number_to_string()

def nokia_3310_message():
    message = input().upper()
    d = {
        "0": " ",
        "1": ".,?!:",
        "2": "ABC",
        "3": "DEF",
        "4": "GHI",
        "5": "JKL",
        "6": "MNO",
        "7": "PQRS",
        "8": "TUV",
        "9": "WXYZ"
    }
    num_message = []
    for letter in message:
        for syms in d.items():
            if letter in syms[1]:
                num_letter = list(d.values()).index(syms[1])
                multi_letter = syms[1].find(letter) + 1
                num_message.append(str(num_letter) * multi_letter)
                break
    print(''.join(num_message))


# nokia_3310_message()

def morse_function():
    message = input().upper()
    morse_code = {"A": ".-", "J": ".---", "S": "...", "1": ".----", "B": "-...", "K": "-.-", "T": "-", "2": "..---",
                  "C": "-.-.", "L": ".-..", "U": "..-", "3": "...--", "D": "-..", "M": "--", "V": "...-", "4": "....-",
                  "E": ".", "N": "-.", "W": ".--", "5": ".....", "F": "..-.", "O": "---", "X": "-..-", "6": "-....",
                  "G": "--.", "P": ".--.", "Y": "-.--", "7": "--...", "H": "....", "Q": "--.-", "Z": "--..",
                  "8": "---..", "I": "..", "R": ".-.", "0": "-----", "9": "----."}
    result_message = []
    for letter in message:
        result_message.append(morse_code[letter])
    print(' '.join(result_message))


# morse_function()
def summary_dicts():
    dict1 = {'a': 100, 'z': 333, 'b': 200, 'c': 300, 'd': 45, 'e': 98, 't': 76, 'q': 34, 'f': 90, 'm': 230}
    dict2 = {'a': 300, 'b': 200, 'd': 400, 't': 777, 'c': 12, 'p': 123, 'w': 111, 'z': 666}
    result = {}
    for i in dict1.keys() | dict2.keys():
        result[i] = dict1.get(i, 0) + dict2.get(i, 0)


# summary_dicts()

def words_count():
    s = 'orange strawberry barley gooseberry apple apricot barley currant orange melon pomegranate banana banana ' \
        'orange barley apricot plum grapefruit banana quince strawberry barley grapefruit banana grapes melon ' \
        'strawberry apricot currant currant gooseberry raspberry apricot currant orange lime quince grapefruit barley ' \
        'banana melon pomegranate barley banana orange barley apricot plum banana quince lime grapefruit strawberry ' \
        'gooseberry apple barley apricot currant orange melon pomegranate banana banana orange apricot barley plum ' \
        'banana grapefruit banana quince currant orange melon pomegranate barley plum banana quince barley lime ' \
        'grapefruit pomegranate barley'.split()

    result = {}
    for word in set(s):
        if s.count(word) in result and word < result[s.count(word)] or s.count(word) not in result:
            result[s.count(word)] = word
    print(result[max(result)])


# words_count()
def pet_dict():
    pets = [('Hatiko', 'Parker', 'Wilson', 50),
            ('Rusty', 'Josh', 'King', 25),
            ('Fido', 'John', 'Smith', 28),
            ('Butch', 'Jake', 'Smirnoff', 18),
            ('Odi', 'Emma', 'Wright', 18),
            ('Balto', 'Josh', 'King', 25),
            ('Barry', 'Josh', 'King', 25),
            ('Snape', 'Hannah', 'Taylor', 40),
            ('Horry', 'Martha', 'Robinson', 73),
            ('Giro', 'Alex', 'Martinez', 65),
            ('Zooma', 'Simon', 'Nevel', 32),
            ('Lassie', 'Josh', 'King', 25),
            ('Chase', 'Martha', 'Robinson', 73),
            ('Ace', 'Martha', 'Williams', 38),
            ('Rocky', 'Simon', 'Nevel', 32)]

    result = {}
    for pet in pets:
        result.setdefault(pet[1:], []).append(pet[0])


# pet_dict()
def the_most_unpopular():
    sentence = input().lower()
    for i in '.,!?:;-':
        sentence = sentence.replace(i, '')
    result = {}
    sent_list = sentence.split()
    for word in set(sent_list):
        if sent_list.count(word) not in result or word < result[sent_list.count(word)]:
            result[sent_list.count(word)] = word
    print(result[min(result)])


# the_most_unpopular()
def the_most_unpopular_v2():
    result = {}
    sentence = [i.strip('.,!?:;-') for i in input().lower().split()]
    for word in sentence:
        if sentence.count(word) not in result or word < result[sentence.count(word)]:
            result[sentence.count(word)] = word
    print(result[min(result)])


# the_most_unpopular_v2()
def rename_duplicates():
    result = {}
    sentence = [i for i in input().split()]
    for i in range(len(sentence)):
        word = sentence[i]
        result[word] = result.setdefault(word, -1) + 1
        if result[word] > 0:
            sentence[i] += f'_{result[word]}'
    print(*sentence)
    # i am i_1 r o n m a n_1


# rename_duplicates()
def dict_of_proger():
    n = int(input())
    p_dict = {}
    for _ in range(n):
        k, v = input().split(': ')
        p_dict[k.lower()] = v
    for _ in range(int(input())):
        print(p_dict.get(input().lower(), 'Не найдено'))


# dict_of_proger()
def anagramm_func():
    # d1, d2 = {}
    in1 = input()
    in2 = input()
    for letter in in1:
        if in1.count(letter) != in2.count(letter):
            exit(print('NO'))
    print('YES')


def anagramm_func_v2():
    print('YES' if sorted(input()) == sorted(input()) else 'NO')


# anagramm_func_v2()
def anagramm_sentences():
    s1, s2 = input().lower(), input().lower()
    for i in '.,!?:;- ':
        s1, s2 = s1.replace(i, ''), s2.replace(i, '')
    print('YES' if sorted(s1) == sorted(s2) else 'NO')


# anagramm_sentences()
def sinonyms():
    d1 = {}
    for _ in range(int(input())):
        sin = input().split()
        d1[sin[0]] = sin[1]
    word = input()
    for item in d1.items():
        if word in item:
            print(item[item.index(word) - 1])


def sinonyms_v2():
    d1 = {}
    for _ in range(int(input())):
        a, b = input().split()
        d1[a], d1[b] = b, a
    print(d1[input()])


# sinonyms_v2()
def countries_cities():
    d = {}
    for _ in range(int(input())):
        words = input().split()
        for word in words[1:]:
            d[word] = words[0]
    for _ in range(int(input())):
        print(d[input()])


# countries_cities()
def telephone_book():
    tb = {}
    for _ in range(int(input())):
        pos = input().split()
        tb.setdefault(pos[1].lower(), []).append(pos[0])
    for _ in range(int(input())):
        print(*tb.get(input().lower(), ('абонент не найден',)))


# telephone_book()
def thats_shifr():
    word = input()
    d1 = {}
    for letter in set(word):
        d1[str(word.count(letter))] = letter
    d = {}
    for _ in range(int(input())):
        line = input().split(': ')
        d[line[1]] = line[0]
    was = set()
    for letter in word:
        if letter not in was:
            word = word.replace(letter, d[str(word.count(letter))])
            was.add(letter)
    print(word)


# thats_shifr()
def dict_generator():
    s = '1:men 2:kind 90:number 0:sun 34:book 56:mountain 87:wood 54:car 3:island 88:power 7:box 17:star 101:ice'

    result = {int(k): v for k, v in [line.split(':') for line in s.split()]}
    print(result)


# dict_generator()

def dict_delimeters():
    numbers = [34, 10, 4, 6, 10, 23, 90, 100, 21, 35, 95, 1, 36, 38, 19, 1, 6, 87, 1000, 13456, 360]

    result = {i: [k for k in range(1, i + 1) if i % k == 0] for i in numbers}
    print(result)


# dict_delimeters()
def gen_dict_ords():
    words = ['hello', 'bye', 'yes', 'no', 'python', 'apple', 'maybe', 'stepik', 'beegeek']

    result = {i: [ord(k) for k in i] for i in words}
    print(result)


# gen_dict_ords()
def gen_dict_with_remove_1():
    letters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 26: 'Z'}

    remove_keys = [1, 5, 7, 12, 17, 19, 21, 24]

    result = {k: v for k, v in letters.items() if k not in remove_keys}
    print(result)


# gen_dict_with_remove_1()
def gen_dict_2():
    students = {'Timur': (170, 75), 'Ruslan': (180, 105), 'Soltan': (192, 68), 'Roman': (175, 70), 'Madlen': (160, 50),
                'Stefani': (165, 70), 'Tom': (190, 90), 'Jerry': (180, 87), 'Anna': (172, 67), 'Scott': (168, 78),
                'John': (186, 79), 'Alex': (195, 120), 'Max': (200, 110), 'Barak': (180, 89), 'Donald': (170, 80),
                'Rustam': (186, 100), 'Alice': (159, 59), 'Rita': (170, 80), 'Mary': (175, 69), 'Jane': (190, 80)}

    result = {k: v for k, v in students.items() if v[0] > 167 and v[1] < 75}
    print(result)


# gen_dict_2()
def gen_dict_3():
    tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18), (19, 20, 21), (22, 23, 24),
              (25, 26, 27), (28, 29, 30), (31, 32, 33), (34, 35, 36)]

    result = {k[0]: k[1:] for k in tuples}
    print(result)


# gen_dict_3()
def gen_list():
    student_ids = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010', 'S011', 'S012',
                   'S013']
    student_names = ['Camila Rodriguez', 'Juan Cruz', 'Dan Richards', 'Sam Boyle', 'Batista Cesare', 'Francesco Totti',
                     'Khalid Hussain', 'Ethan Hawke', 'David Bowman', 'James Milner', 'Michael Owen', 'Gary Oldman',
                     'Tom Hardy']
    student_grades = [86, 98, 89, 92, 45, 67, 89, 90, 100, 98, 10, 96, 93]

    result = [{i: {nm: wt}} for i, nm, wt in zip(student_ids, student_names, student_grades)]
    print(result)


# gen_list()
def bingo_game():
    matrix = []

    for i in random.sample(range(1, 76), k=25):
        matrix.append(i)

    matrix[12] = 0
    for i in range(len(matrix)):
        print(str(matrix[i]).ljust(3), end=' ')
        if i in [4, 9, 14, 19, 24]:
            print()
        i += 1


# bingo_game()


def secret_santa():
    m = int(input())
    names = []
    for _ in range(m):
        names.append(input())
    friend_names = names.copy()
    for i in names:
        print(i, end=' - ')
        name = random.choice(friend_names)
        while name == i:
            name = random.choice(friend_names)
        friend_names.remove(name)
        print(name)


# secret_santa()
def dict_update():
    my_dict = {'C1': [10, 20, 30, 7, 6, 23, 90], 'C2': [20, 30, 40, 1, 2, 3, 90, 12], 'C3': [12, 34, 20, 21],
               'C4': [22, 54, 209, 21, 7], 'C5': [2, 4, 29, 21, 19], 'C6': [4, 6, 7, 10, 55], 'C7': [4, 8, 12, 23, 42],
               'C8': [3, 14, 15, 26, 48], 'C9': [2, 7, 18, 28, 18, 28]}
    for i in my_dict:
        my_dict[i] = [k for k in my_dict[i] if k <= 20]


# dict_update()
def mail_domen():
    emails = {'nosu.edu': ['timyr', 'joseph', 'svetlana.gaeva', 'larisa.mamuk'],
              'gmail.com': ['ruslan.chaika', 'rustam.mini', 'stepik-best'],
              'msu.edu': ['apple.fruit', 'beegeek', 'beegeek.school'],
              'yandex.ru': ['surface', 'google'],
              'hse.edu': ['tomas-henders', 'cream.soda', 'zivert'],
              'mail.ru': ['angel.down', 'joanne', 'the.fame.moster']}
    form_mails = set()
    for domain in emails:
        for name in emails[domain]:
            form_mails.add(f'{name}@{domain}')
    for mail in sorted(form_mails):
        print(mail)


# mail_domen()
def dna_rna():
    dna = input()
    dna_to_rna = {'G': 'C', 'C': 'G', 'T': 'A', 'A': 'U'}
    rna = ''
    for letter in dna:
        rna += dna_to_rna[letter]
    print(rna)


# dna_rna()
def count_of_inputs():
    sentence = input().split()
    result_counts = {}
    count_sentence = []
    # result_sentence = []
    for word in sentence:
        result_counts[word] = result_counts.setdefault(word, 0) + 1
        count_sentence.append(result_counts[word])
    print(*count_sentence)


# count_of_inputs()
def scrabble():
    d = {
        1: "AEILNORSTU",
        2: "DG",
        3: "BCMP",
        4: "FHVWY",
        5: "K",
        8: "JX",
        10: "QZ"
    }
    sentence = input()
    result = 0
    for letter in sentence:
        for k in d:
            if letter in d[k]:
                result += k
    print(result)


# scrabble()
def build_query_string(params):
    return '&'.join([f'{k}={params[k]}' for k in sorted(params)])


# print(build_query_string({'sport': 'hockey', 'game': 2, 'time': 17}))

def merge(values):  # values - это список словарей
    res_dict = {}
    for d in values:
        for k in d:
            res_dict.setdefault(k, set()).add(d[k])
    return res_dict


# print(merge([{'a': 1, 'b': 2}, {'b': 10, 'c': 100}, {'a': 1, 'b': 17, 'c': 50}, {'a': 5, 'd': 777}]))
# print(merge([{}, {}]))
def super_virus():
    files_dict = {}
    for _ in range(int(input())):
        line = input().split()
        for letter in line[1:]:
            files_dict.setdefault(line[0], set()).add(letter)
    for _ in range(int(input())):
        right, name = input().split()
        for k, v in [('X', 'execute'), ('W', 'write'), ('R', 'read')]:
            right = right.replace(v, k)
        if right in files_dict.get(name, set()):
            print('OK')
        else:
            print('Access denied')


# super_virus()
def internet_shop():
    shop_dict = {}
    for _ in range(int(input())):
        name, position, c = input().split()
        c = int(c)
        shop_dict[name][position] = shop_dict.setdefault(name, {}).setdefault(position, 0) + c
    for name in sorted(shop_dict):
        print(f'{name}:')
        for position in sorted(shop_dict[name]):
            print(position, shop_dict[name][position])


# internet_shop()


def generate_password():
    letter = ''.join((set(ascii_letters) | set(digits)) - set('lI1oO0'))
    n, m = int(input()), int(input())
    for _ in range(n):
        password = ''.join(random.choices(letter, k=m))
        print(password)


# generate_password()
def generate_hard_password():
    letter_dict = {'EN': {x for x in ascii_uppercase if x not in 'OI'},
                   'en': {x for x in ascii_lowercase if x not in 'ol'},
                   'dig': {x for x in digits if x not in '01'}}
    letter_all = ''.join((set(ascii_letters) | set(digits)) - set('lI1oO0'))
    n, m = int(input()), int(input())
    for _ in range(n):

        enough = False
        while not enough:
            password = random.choices(letter_all, k=m)
            if bool(set(password) & letter_dict['EN']) & bool(set(password) & letter_dict['en']) & bool(
                    set(password) & letter_dict['dig']):
                enough = True
        print(''.join(password))


# generate_hard_password()


def method_monte_carlo(n=10):
    # n = 100000
    k = 0
    s0 = 1
    for _ in range(n):
        x = random.uniform(0, 1)  # случайное число с плавающей точкой от 0 до 1
        y = random.uniform(0, 1)  # случайное число с плавающей точкой от 0 до 1

        if y <= x ** 2:  # если попадает в нужную область
            k += 1

    print(n, (k / n) * s0, sep=': ')


# method_monte_carlo(50000000)


def method_monte_carlo_2():
    n = 10 ** 6  # количество испытаний
    k = 0
    s0 = 16
    for _ in range(n):
        x = random.uniform(-2, 2)  # случайное число с плавающей точкой от 0 до 1
        y = random.uniform(-2, 2)  # случайное число с плавающей точкой от 0 до 1

        if x ** 3 + y ** 4 + 2 >= 0 and 3 * x + y ** 2 <= 2:  # если попадает в нужную область
            k += 1

    print((k / n) * s0)


# method_monte_carlo_2()

def method_monte_carlo_for_pi():
    n = 10 ** 6  # количество испытаний
    k = 0
    r = 1
    s0 = 4
    for _ in range(n):
        x = random.uniform(-1, 1)  # случайное число с плавающей точкой от 0 до 1
        y = random.uniform(-1, 1)  # случайное число с плавающей точкой от 0 до 1

        if x ** 2 + y ** 2 <= 1:  # если попадает в нужную область
            k += 1

    # print(s0 = pi * r**2)

    pi = (k / n) * s0 / r ** 2
    print(pi)


# method_monte_carlo_for_pi()


def func_decimal():
    s = list(map(Decimal,
                 '9.73 8.84 8.92 9.60 9.32 8.97 8.53 1.26 6.62 9.85 1.85 1.80 0.83 6.75 9.74 9.11 9.14 5.03 5.03 1.34 '
                 '3.52 8.09 7.89 8.24 8.23 5.22 0.30 2.59 1.25 6.24 2.14 7.54 5.72 2.75 2.32 2.69 9.32 8.11 4.53 0.80 '
                 '0.08 9.36 5.22 4.08 3.86 5.56 1.43 8.36 6.29 5.13'.split()))
    print(sum(s))
    print(*map(str, sorted(s)[-5:][::-1]))


# func_decimal()


def func_decimal_2():
    num = Decimal(input())
    print(num.as_tuple())
    if -1 < num < 1:
        print(min(num.as_tuple().digits.__add__((0,))) + max(num.as_tuple().digits))
    else:
        print(min(num.as_tuple().digits) + max(num.as_tuple().digits))


# func_decimal_2()

def result_of_formula():
    d = Decimal(input())
    result = Decimal.exp(d) + Decimal.ln(d) + Decimal.log10(d) + Decimal.sqrt(d)
    print(result)


# result_of_formula()


def func_fractions():
    numbers = ['6.34', '4.08', '3.04', '7.49', '4.45', '5.39', '7.82', '2.76', '0.71', '1.97', '2.54', '3.67', '0.14',
               '4.29', '1.84', '4.07', '7.26', '9.37', '8.11', '4.30', '7.16', '2.46', '1.27', '0.29', '5.12', '4.02',
               '6.95', '1.62', '2.26', '0.45', '6.91', '7.39', '0.52', '1.88', '8.38', '0.75', '0.32', '4.81', '3.31',
               '4.63', '7.84', '2.25', '1.10', '3.35', '2.05', '7.87', '2.40', '1.20', '2.58', '2.46']
    for num in numbers:
        print(num, Fraction(num), sep=' = ')


# func_fractions()


def func_fractions_2():
    s = list(map(Fraction,
                 '0.78 4.3 9.6 3.88 7.08 5.88 0.23 4.65 2.79 0.90 4.23 2.15 3.24 8.57 0.10 8.57 1.49 5.64 3.63 8.36 '
                 '1.56 6.67 ' \
                 '1.46 5.26 4.83 7.13 1.22 1.02 7.82 9.97 5.40 9.79 9.82 2.78 2.96 0.07 1.72 7.24 7.84 9.23 1.71 6.24 '
                 '5.78 ' \
                 '5.37 0.03 9.60 8.86 2.73 5.83 6.50 0.123 0.00021'.split()))
    print(min(s) + max(s))


# func_fractions_2()


def func_fractions_3():
    m, n = int(input()), int(input())
    print(Fraction(m, n))


# func_fractions_3()


def func_fractions_4():
    num1 = input()
    num2 = input()
    f_num1 = Fraction(num1)
    f_num2 = Fraction(num2)
    print(f'{num1} + {num2} = {f_num1 + f_num2}')
    print(f'{num1} - {num2} = {f_num1 - f_num2}')
    print(f'{num1} * {num2} = {f_num1 * f_num2}')
    print(f'{num1} / {num2} = {f_num1 / f_num2}')


# func_fractions_4()


def func_fractions_5():
    n = int(input())
    result = Fraction(0)
    for i in range(1, n + 1):
        result += Fraction(1, i ** 2)
    print(result)


# func_fractions_5()
def func_fractions_6():
    n = int(input())
    print(sum([Fraction(1, math.factorial(i)) for i in range(1, n + 1)]))


# func_fractions_6()


def super_fraction():
    n = int(input())
    for i in range(n // 2, 0, -1):
        if math.gcd(i, n - i) == 1:
            exit(print(Fraction(i, n - i)))


# super_fraction()
def super_fraction_2():
    n = int(input())
    i = 1
    res = set()
    while Fraction(i, n) < 1:
        if math.gcd(i, n) == 1:
            res.add(Fraction(i, n))
        i += 1
        if i == n:
            i = 1
            n -= 1
    print(*sorted(res), sep='\n')


# super_fraction_2()

def complex_func_1():
    z1 = complex(input())
    z2 = (complex(input()))
    for op in '+-*':
        print(f'{z1} {op} {z2} = {eval(f"{z1} {op} {z2}")}')


# complex_func_1()
def complex_func_2():
    numbers = [3 + 4j, 3 + 1j, -7 + 3j, 4 + 8j, -8 + 10j, -3 + 2j, 3 - 2j, -9 + 9j, -1 - 1j, -1 - 10j, -20 + 15j,
               -21 + 1j, 1j, -3 + 8j, 4 - 6j, 8 + 2j, 2 + 3j]
    result = 0
    for num in numbers:
        if abs(num) > abs(result):
            result = num
    print(result, abs(result), sep='\n')


# complex_func_2()
def complex_func_3():
    n = int(input())
    z1 = complex(input())
    z2 = complex(input())
    print(z1 ** n + z2 ** n + z1.conjugate() ** n + z2.conjugate() ** (n + 1))


# complex_func_3()
def matrix(n=1, m=None, value=0):
    if m is None:
        m = n
    matrix = []
    for _ in range(n):
        row = [value for _ in range(m)]
        matrix.append(row)
    return matrix


# print(matrix(3,4,9))
def mean(*args):
    # [i for i in args if str(i).isdigit]
    l_1 = [i for i in args if type(i) is float or type(i) is int]
    if len(l_1) == 0:
        return 0.0
    return sum(l_1) / len(l_1)


# print(mean())
# print(mean(7))
# print(mean(1.5, True, ['stepik'], 'beegeek', 2.5, (1, 2)))
# print(mean(True, ['stepik'], 'beegeek', (1, 2)))
# print(mean(-1, 2, 3, 10, ('5')))
# print(mean(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))


def sorting_meth():
    i = int(input())

    def mean(point):
        return point[i - 1]

    athletes = [('Дима', 10, 130, 35), ('Тимур', 11, 135, 39), ('Руслан', 9, 140, 33), ('Рустам', 10, 128, 30),
                ('Амир', 16, 170, 70), ('Рома', 16, 188, 100), ('Матвей', 17, 168, 68), ('Петя', 15, 190, 90)]

    for p in athletes:
        print(*p)


# sorting_meth()
def double(x):
    return x ** 2


def cube(x):
    return x ** 3


def square(x):
    return math.sqrt(x)


def absolute_val(x):
    return abs(x)


def sinus_func(x):
    return math.sin(x)


def main_function_of_many():
    n = int(input())
    func = input()
    funcs = {'квадрат': double, 'куб': cube, 'корень': square, 'модуль': absolute_val, 'синус': sinus_func}
    print(funcs[func](n))


# main_function_of_many()

def non_decreasing_sum_nums():
    line = input().split()
    print(*sorted(line, key=non_decreasing_sum_nums_mean))


def non_decreasing_sum_nums_mean(point):
    return sum(map(int, list(point)))


# non_decreasing_sum_nums()


def non_decreasing_2():
    line = input().split()
    print(*sorted(line, key=non_decreasing_mean_2))


def non_decreasing_mean_2(point):
    return (sum(map(int, list(point))), int(point))


# non_decreasing_2()
n = 2


# def map(function, items):
#     result = []
#     for item in items:
#         result.append(function(item, n))
#     return result


# numbers = [3.56773, 5.57668, 4.00914, 56.24241, 9.01344, 32.12013, 23.22222, 90.09873, 45.45, 314.1528, 2.71828,
#            1.41546]
# print(*map(round, numbers))

# def map(function, items):
#     result = []
#     for item in items:
#         result.append(function(item))
#     return result


# def filter(function, items):
#     result = []
#     for item in items:
#         if function(item):
#             result.append(item)
#     return result


def lenny(num):
    return len(str(num)) == 3 and num % 5 == 2


def cubes(num):
    return num ** 3


# numbers = [1014, 1321, 675, 1215, 56, 1386, 1385, 431, 1058, 486, 1434, 696, 1016, 1084, 424, 1189, 475, 95, 1434, 1462,
#            815, 776, 657, 1225, 912, 537, 1478, 1176, 544, 488, 668, 944, 207, 266, 1309, 1027, 257, 1374, 1289, 1155,
#            230, 866, 708, 144, 1434, 1163, 345, 394, 560, 338, 232, 182, 1438, 1127, 928, 1309, 98, 530, 1013, 898, 669,
#            105, 130, 1363, 947, 72, 1278, 166, 904, 349, 831, 1207, 1496, 370, 725, 926, 175, 959, 1282, 336, 1268, 351,
#            1439, 186, 273, 1008, 231, 138, 142, 433, 456, 1268, 1018, 1274, 387, 120, 340, 963, 832, 1127]
#
# numbers = filter(lenny, numbers)
# numbers = map(cubes, numbers)
# pass
# def reduce(operation, items, initial_value):
#     acc = initial_value
#     for item in items:
#         acc = operation(acc, item)
#     return acc


# def sqr(acc, item):
#     return acc + item**2
#
# numbers = [97, 42, 9, 32, 3, 45, 31, 77, -1, 11, -2, 75, 5, 51, 34, 28, 46, 1, -8, 84, 16, 51, 90, 56, 65, 90, 23, 35, 11, -10, 70, 90, 90, 12, 96, 58, -8, -4, 91, 76, 94, 60, 72, 43, 4, -6, -5, 51, 58, 60, 30, 38, 67, 62, 36, 72, 34, 82, 62, -1, 60, 82, 87, 81, -7, 57, 26, 36, 17, 43, 80, 40, 75, 94, 91, 64, 38, 72, 29, 84, 38, 35, 7, 54, 31, 95, 78, 27, 82, 1, 64, 94, 31, 29, -8, 98, 24, 61, 7, 73]
# print(reduce(sqr, numbers, 0))

def map_2(function, items):
    result = []
    for item in items:
        result.append(function(item))
    return result


def sqr(item):
    return item ** 2


def filter_2(function, items):
    result = []
    for item in items:
        if function(item):
            result.append(item)
    return result


# def sorting(item):
#     if len(str(abs(item))) == 2 and item % 7 == 0:
#         return item


# numbers = [77, 293, 28, 242, 213, 285, 71, 286, 144, 276, 61, 298, 280, 214, 156, 227, 228, 51, -4, 202, 58, 99, 270, 219, 94, 253, 53, 235, 9, 158, 49, 183, 166, 205, 183, 266, 180, 6, 279, 200, 208, 231, 178, 201, 260, -35, 152, 115, 79, 284, 181, 92, 286, 98, 271, 259, 258, 196, -8, 43, 2, 128, 143, 43, 297, 229, 60, 254, -9, 5, 187, 220, -8, 111, 285, 5, 263, 187, 192, -9, 268, -9, 23, 71, 135, 7, -161, 65, 135, 29, 148, 242, 33, 35, 211, 5, 161, 46, 159, 23, 169, 23, 172, 184, -7, 228, 129, 274, 73, 197, 272, 54, 278, 26, 280, 13, 171, 2, 79, -2, 183, 10, 236, 276, 4, 29, -10, 41, 269, 94, 279, 129, 39, 92, -63, 263, 219, 57, 18, 236, 291, 234, 10, 250, 0, 64, 172, 216, 30, 15, 229, 205, 123, -105]
# # numbers = [int(i) for i in numbers if len(str(abs(i))) == 2 and i % 7 == 0]
# numbers = filter_2(sorting, numbers)
# print(sum(map_2(sqr, numbers)))

from functools import reduce


def less_15_7_1():
    floats = [4.35, 6.09, 3.25, 9.77, 2.16, 8.88, 4.59, 34.23, 12.12, 4.67, 2.45, 9.32]
    words = ['racecar', 'akinremi', 'deed', 'temidayo', 'omoseun', 'civic', 'TATTARRATTAT', 'malayalam', 'nun']
    numbers = [4, 6, 9, 23, 5]

    # Исправьте этот код
    map_result = list(map(lambda num: round(num ** 2, 1), floats))
    filter_result = list(filter(lambda name: len(name) > 4 and name == name[::-1], words))
    reduce_result = reduce(lambda num1, num2: num1 * num2, numbers, 1)

    print(map_result)
    print(filter_result)
    print(reduce_result)


# less_15_7_1()
def less_15_7_2():
    data = [['Tokyo', 35676000, 'primary'],
            ['New York', 19354922, 'nan'],
            ['Mexico City', 19028000, 'primary'],
            ['Mumbai', 18978000, 'admin'],
            ['Sao Paulo', 18845000, 'admin'],
            ['Delhi', 15926000, 'admin'],
            ['Shanghai', 14987000, 'admin'],
            ['Kolkata', 14787000, 'admin'],
            ['Los Angeles', 12815475, 'nan'],
            ['Dhaka', 12797394, 'primary'],
            ['Buenos Aires', 12795000, 'primary'],
            ['Karachi', 12130000, 'admin'],
            ['Cairo', 11893000, 'primary'],
            ['Rio de Janeiro', 11748000, 'admin'],
            ['Osaka', 11294000, 'admin'],
            ['Beijing', 11106000, 'primary'],
            ['Manila', 11100000, 'primary'],
            ['Moscow', 10452000, 'primary'],
            ['Istanbul', 10061000, 'admin'],
            ['Paris', 9904000, 'primary']]
    data = filter(lambda x: x[1] > 10000000 and x[2] == 'primary', data)
    data = sorted(data)
    data = map(lambda x: x[0], data)
    data = reduce(lambda x, y: x + ', ' + y, data)
    print('Cities:', data)


# less_15_7_2()

def less_15_8_1():
    func = lambda x: True if (x % 13 == 0 or x % 19 == 0) else False
    print(func(19))
    print(func(13))
    print(func(20))
    print(func(15))
    print(func(247))


# less_15_8_1()

def less_15_8_2():
    func = lambda x: True if x.lower()[0] == x.lower()[-1] == 'a' else False
    print(func('abcd'))
    print(func('bcda'))
    print(func('abcda'))
    print(func('Abcd'))
    print(func('bcdA'))
    print(func('abcdA'))


# less_15_8_2()
def less_15_8_3():
    is_non_negative_num = lambda x: x.replace('.', '', 1).isdigit()
    print(is_non_negative_num('10.34ab'))
    print(is_non_negative_num('10.45'))
    print(is_non_negative_num('-18'))
    print(is_non_negative_num('-34.67'))
    print(is_non_negative_num('987'))
    print(is_non_negative_num('abcd'))
    print(is_non_negative_num('123.122.12'))
    print(is_non_negative_num('123.122'))


# less_15_8_3()
def less_15_8_4():
    is_num = lambda x: '-' not in x[1:] and x.replace('.', '', 1).replace('-', '', 1).isdigit()
    print(is_num('10.34ab'))
    print(is_num('10.45'))
    print(is_num('-18'))
    print(is_num('-34.67'))
    print(is_num('987'))
    print(is_num('abcd'))
    print(is_num('123.122.12'))
    print(is_num('-123.122'))
    print(is_num('--13.2'))


# less_15_8_4()
def less_15_8_5():
    words = ['beverage', 'monday', 'abroad', 'bias', 'abuse', 'abolish', 'abuse', 'abuse', 'bid', 'wednesday', 'able',
             'betray', 'accident', 'abduct', 'bigot', 'bet', 'abandon', 'besides', 'access', 'friday', 'bestow',
             'abound', 'absent', 'beware', 'abundant', 'abnormal', 'aboard', 'about', 'accelerate', 'abort', 'thursday',
             'tuesday', 'sunday', 'berth', 'beyond', 'benevolent', 'abate', 'abide', 'bicycle', 'beside', 'accept',
             'berry', 'bewilder', 'abrupt', 'saturday', 'accessory', 'absorb']
    words = filter(lambda x: len(x) == 6, words)

    print(*list(sorted(words)))

# less_15_8_5()
def less_15_8_6():
    numbers = [46, 61, 34, 17, 56, 26, 93, 1, 3, 82, 71, 37, 80, 27, 77, 94, 34, 100, 36, 81, 33, 81, 66, 83, 41, 80,
               80, 93, 40, 34, 32, 16, 5, 16, 40, 93, 36, 65, 8, 19, 8, 75, 66, 21, 72, 32, 41, 59, 35, 64, 49, 78, 83,
               27, 57, 53, 43, 35, 48, 17, 19, 40, 90, 57, 77, 56, 80, 95, 90, 27, 26, 6, 4, 23, 52, 39, 63, 74, 15, 66,
               29, 88, 94, 37, 44, 2, 38, 36, 32, 49, 5, 33, 60, 94, 89, 8, 36, 94, 46, 33]
    # numbers = filter(lambda x: x % 2 == 0 or (x < 47 and x % 2 == 1), numbers)
    # print(list(numbers))
    numbers = map(lambda x: x//2 if x % 2 == 0 else x, filter(lambda x: x % 2 == 0 or (x < 47 and x % 2 == 1), numbers))
    print(*numbers)


# less_15_8_6()
def less_15_8_7():
    data = [(19542209, 'New York'), (4887871, 'Alabama'), (1420491, 'Hawaii'), (626299, 'Vermont'),
            (1805832, 'West Virginia'), (39865590, 'California'), (11799448, 'Ohio'), (10711908, 'Georgia'),
            (10077331, 'Michigan'), (10439388, 'Virginia'), (7705281, 'Washington'), (7151502, 'Arizona'),
            (7029917, 'Massachusetts'), (6910840, 'Tennessee')]
    data = sorted(data, key=lambda x: x[1][-1], reverse=True)
    for k, v in data:
        print(v, k, sep=': ')

# less_15_8_7()
def less_15_8_8():
    data = ['год', 'человек', 'время', 'дело', 'жизнь', 'день', 'рука', 'раз', 'работа', 'слово', 'место', 'лицо',
            'друг', 'глаз', 'вопрос', 'дом', 'сторона', 'страна', 'мир', 'случай', 'голова', 'ребенок', 'сила', 'конец',
            'вид', 'система', 'часть', 'город', 'отношение', 'женщина', 'деньги']
    print(*sorted(sorted(data), key=lambda x: len(x)))

# less_15_8_8()
def less_15_8_9():
    mixed_list = ['tuesday', 'abroad', 'abuse', 'beside', 'monday', 'abate', 'accessory', 'absorb', 1384878, 'sunday',
                  'about', 454805, 'saturday', 'abort', 2121919, 2552839, 977970, 1772933, 1564063, 'abduct', 901271,
                  2680434, 'bicycle', 'accelerate', 1109147, 942908, 'berry', 433507, 'bias', 'bestow', 1875665,
                  'besides', 'bewilder', 1586517, 375290, 1503450, 2713047, 'abnormal', 2286106, 242192, 701049,
                  2866491, 'benevolent', 'bigot', 'abuse', 'abrupt', 343772, 'able', 2135748, 690280, 686008, 'beyond',
                  2415643, 'aboard', 'bet', 859105, 'accident', 2223166, 894187, 146564, 1251748, 2851543, 1619426,
                  2263113, 1618068, 'berth', 'abolish', 'beware', 2618492, 1555062, 'access', 'absent', 'abundant',
                  2950603, 'betray', 'beverage', 'abide', 'abandon', 2284251, 'wednesday', 2709698, 'thursday', 810387,
                  'friday', 2576799, 2213552, 1599022, 'accept', 'abuse', 'abound', 1352953, 'bid', 1805326, 1499197,
                  2241159, 605320, 2347441]
    print(max(mixed_list, key=lambda x: x if type(x) == int else 0))

# less_15_8_9()
def less_15_8_10():
    mixed_list = ['beside', 48, 'accelerate', 28, 'beware', 'absorb', 'besides', 'berry', 15, 65, 'abate', 'thursday',
                  76, 70, 94, 35, 36, 'berth', 41, 'abnormal', 'bicycle', 'bid', 'sunday', 'saturday', 87, 'bigot', 41,
                  'abort', 13, 60, 'friday', 26, 13, 'accident', 'access', 40, 26, 20, 75, 13, 40, 67, 12, 'abuse', 78,
                  10, 80, 'accessory', 20, 'bewilder', 'benevolent', 'bet', 64, 38, 65, 51, 95, 'abduct', 37, 98, 99,
                  14, 'abandon', 'accept', 46, 'abide', 'beyond', 19, 'about', 76, 26, 'abound', 12, 95, 'wednesday',
                  'abundant', 'abrupt', 'aboard', 50, 89, 'tuesday', 66, 'bestow', 'absent', 76, 46, 'betray', 47,
                  'able', 11]
    print(*sorted(mixed_list, key=lambda x: str(x)))


# less_15_8_10()
from operator import sub, add


def less_15_8_11():
    print(*map(sub, (255,255,255), map(int, input().split())))

# less_15_8_11()


def less_15_8_12():
    coeff = list(map(int, input().split()))
    x = int(input())
    result = list(map(lambda c, i: c * (x ** i), coeff, range(len(coeff)-1, -1, -1)))
    # print(result)
    print(reduce(add, result))
    # print(evaluate(coeff, x))


# less_15_8_12()
def less_15_9_1():
    def ignore_command(command):
        ignore = ['alias', 'configuration', 'ip', 'sql', 'select', 'update', 'exec', 'del', 'truncate']

        return any(map(lambda x: x in command, ignore))

    print(ignore_command('get ip'))
    print(ignore_command('select all'))
    print(ignore_command('delete'))
    print(ignore_command('trancate'))

# less_15_9_1()
def less_15_9_2():
    countries = ['Russia', 'USA', 'UK', 'Germany', 'France', 'India']
    capitals = ['Moscow', 'Washington', 'London', 'Berlin', 'Paris', 'Delhi']
    population = [145_934_462, 331_002_651, 80_345_321, 67_886_011, 65_273_511, 1_380_004_385]
    for cap, countr, pop in zip(capitals, countries, population):
        print(f'{cap} is the capital of {countr}, population equal {pop} people.')

# less_15_9_2()
def less_15_9_3():
    abscissas = list(map(float, input().split()))
    ordinates = list(map(float, input().split()))
    applicates = list(map(float, input().split()))
    print(all([2 ** 2 >= x ** 2 + y ** 2 + z ** 2 for x, y, z in zip(abscissas, ordinates, applicates)]))

# less_15_9_3()
def less_15_9_4():
    print(all(map(lambda x: x.isdigit() and 0 <= int(x) <= 255, input().split('.'))))

# less_15_9_4()
def less_15_9_5():
    n, m = int(input()), int(input())
    nums = list(filter(lambda x: True if '0' not in str(x)  else False, range(n, m+1)))
    nums = list(filter(lambda x: True if all([1 if (x % int(i) == 0) else 0 for i in str(x)]) else False, nums))
    print(*nums)


# less_15_9_5()
def less_15_9_6():
    password = input()
    check = all([len(password) >= 7,
                 any([i.isupper() for i in password]),
                 any([i.islower() for i in password]),
                 any([i.isdigit() for i in password])])
    if check:
        print('YES')
    else:
        print('NO')

# less_15_9_6()
def less_15_9_7():
    stud = []
    for i in range(int(input())):
        stud.append(any([1 if 5 % int(input().split()[1]) == 0 else 0 for _ in range(int(input()))]))
    print('YES' if all(stud) else 'NO')

less_15_9_7()