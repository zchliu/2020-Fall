import numpy as np
from collections import defaultdict


class itemtype:
    def __init__(self, distance, position):
        self.distance = distance
        self.position = position
        self.last_item = []


def edit(str1, str2):
    l1 = len(str1)
    l2 = len(str2)

    D = np.array([[itemtype(0, (i, j)) for j in range(l2 + 10)] for i in range(l1 + 10)], dtype=itemtype)

    for i in range(0, l1 + 1):
        D[i, 0].distance = i

    for j in range(0, l2 + 1):
        D[0, j].distance = j

    # 加上从边缘到D[0,0]的边，防止后面path出错
    for i in range(1, l1 + 1):
        D[i, 0].last_item.append(D[i - 1, 0])

    for j in range(1, l2 + 1):
        D[0, j].last_item.append(D[0, j - 1])

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if str1[i - 1] == str2[j - 1]:
                D[i, j].distance = min(D[i - 1, j].distance + 1, D[i, j - 1].distance + 1, D[i - 1, j - 1].distance)
                if (D[i, j].distance == D[i - 1, j].distance + 1): D[i, j].last_item.append(D[i - 1, j])
                if (D[i, j].distance == D[i, j - 1].distance + 1): D[i, j].last_item.append(D[i, j - 1])
                if (D[i, j].distance == D[i - 1, j - 1].distance): D[i, j].last_item.append(D[i - 1, j - 1])
            else:
                D[i, j].distance = min(D[i - 1, j].distance + 1, D[i, j - 1].distance + 1, D[i - 1, j - 1].distance + 2)
                if (D[i, j].distance == D[i - 1, j].distance + 1): D[i, j].last_item.append(D[i - 1, j])
                if (D[i, j].distance == D[i, j - 1].distance + 1): D[i, j].last_item.append(D[i, j - 1])
                if (D[i, j].distance == D[i - 1, j - 1].distance + 2): D[i, j].last_item.append(D[i - 1, j - 1])

    return D[l1, l2].distance, D


def bfs(s, end):
    queue = []
    path = defaultdict(list)
    queue.append(s)
    seen = set()
    seen.add(s)
    while (len(queue) > 0):
        vertex = queue.pop(0)
        for i in vertex.last_item:
            # print(i.distance)
            if i not in seen:
                queue.append(i)
                path[i] = vertex
                seen.add(i)
    return path


def output(str1, str2, distance, D, path):
    last = D[0, 0]
    end = D[len(str1), len(str2)]
    lst = ""
    pos = 0

    print(str1 + " " + str2)
    while (1):
        if (last == end):
            break

        next = path[last]

        if (next.distance != last.distance):
            # 在下一个位置的距离不相等的时候才做变化
            if (next.position[0] == last.position[0] + 1 and next.position[1] == last.position[1] + 1):
                # 替换
                j = next.position[1] - 1
                lst = lst + (str2[j])
                pos = pos + 1
                print("".join(lst + str1[pos:]), " 替换", " ", distance - next.distance)

            if (next.position[0] == last.position[0] + 1 and next.position[1] == last.position[1]):
                # 删除
                pos = pos + 1
                print("".join(lst + str1[pos:]), " 删除", " ", distance - next.distance)

            if (next.position[0] == last.position[0] and next.position[1] == last.position[1] + 1):
                # 增加
                j = next.position[1] - 1
                lst = lst + (str2[j])
                print("".join(lst + str1[pos:]), " 增加", " ", distance - next.distance)

        else:
            pos = pos + 1
            j = next.position[1] - 1
            lst = lst + (str2[j])
        last = next


if __name__ == "__main__":

    str1 = input("Enter a string:")
    str2 = input("Enter another string:")

    distance, D = edit(str1, str2)
    path = bfs(D[len(str1), len(str2)], D[0, 0])
    print("The minimun editting distance is: ", distance)
    output(str1, str2, distance, D, path)
