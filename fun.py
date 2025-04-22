def process_rows(row_data):
    row1, row2, row3 = row_data
    result = []
    # 对每一行中的每列并行处理
    for val1, val2, val3 in zip(row1, row2, row3):
        result.append([val1, val2, val3])
    return result