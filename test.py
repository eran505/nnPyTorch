

str_l="0: 1: ;0:{(0, 10, 0, ),(0, 0, 0, )} ;1:{(1, 10, 0, ),(1, 0, 0, )} ;2:{(3, 11, 1, ),(2, 1, 1, )} ;3:{(7, 12, 2, ),(2, 0, -1, )} ;4:{(9, 13, 1, ),(2, 1, -1, )} ;5:{(11, 14, 1, ),(2, 1, 0, )} ;6:{(13, 16, 2, ),(2, 2, 1, )} ;7:{(15, 18, 3, ),(0, 0, 0, )} ;8:{(15, 17, 3, ),(0, -1, 0, )} ;9:{(16, 15, 3, ),(1, -2, 0, )} ;10:{(18, 13, 2, ),(2, -2, -1, )} ;11:{(20, 11, 0, ),(2, -2, -2, )} ;"

arr = str(str_l).split(";")
arr=arr[1:]
l=[]
for item in arr:
    z = str(item).split("{")[-1][:-2]
    if len(z)<3:
        continue
    l.append(eval(z))

x = (19,9,1)

for item in l:
    pos = item[0]
    diff = (abs(pos[0]-x[0]),abs(pos[1]-x[1]),abs(pos[2]-x[2]))
    diff_max = max(diff)
    print(diff_max)

x = (19,10,1)
print("-"*10)
for item in l:
    pos = item[0]
    diff = (abs(pos[0]-x[0]),abs(pos[1]-x[1]),abs(pos[2]-x[2]))
    diff_max = max(diff)
    print(diff_max)