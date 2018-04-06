# def read_car():
#     f=open('data/cars.csv', 'r', encoding='utf-8')
#     rows=[]
#     for row in f:
#         #print(row, end='')
#         #print(row.strip())
#         #print(row.strip().split(','))
#     rows.append(row.strip().split(','))
#     f.close()
#     return rows

import csv

def write_tree(rows):
    f=open('data/treeout2.csv', 'w', encoding='utf-8', newline='')
    writer=csv.writer(f, delimiter=':')
    for row in rows:
        writer.writerows(row)

    # csv.writer(f).writerows(rows)
    f.close()

def read_tree():
    f=open('data/trees.csv', 'r', encoding='utf-8')
    rows=[]
    for row in csv.reader(f):
        #print(row)
        rows.append(row)
    f.close()
    return rows

rows=read_tree()
print(rows)
write_tree(rows)

# rows=read_car()
# print(rows)

















