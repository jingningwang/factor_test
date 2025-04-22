from itertools import zip_longest
import multiprocessing
from fun import process_rows

list1 = [[1, 2], [3, 4], [5, 6]]
list2 = [[7, 8], [9, 10], [11, 12]]
list3 = [[13, 14], [15, 16], [17, 18]]

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        rows = zip_longest(list1, list2, list3, fillvalue=[])
        results = pool.map(process_rows, rows)
    # 输出结果
    # for result in results:
    #     for row in result:
    #         sum = row[0] + row[1] + row[2]
    print(results)