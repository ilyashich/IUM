import csv
parse_data = []
with open('../data/data_with_categories.csv') as f:
    reader = csv.reader(f, delimiter=',')
    for line in reader:
        parse_data.append(line)
print(parse_data[0][2]) # user_id index = 3
print(parse_data[0][38]) # category index = 12 do 39 lacznie 28
users = 200
categories = 28
used_area = []
index = 0
for session in parse_data:
    if index > 0:
        for x in range(28):
            if (parse_data[index][2], parse_data[index][11 + x]) not in used_area:
                used_area.append((parse_data[index][2], parse_data[index][11 + x]))
    index += 1

full_area = users * categories
print(f"Used data level: {(len(used_area) - 1) / full_area}")
print(f"used area = {len(used_area)}")
print(f"full area = {full_area}")
