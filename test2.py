
a = "morduspordus"

list_str = [a[i:] for i in range(len(a) + 1)]
print(list_str)

for a in [i + '$' for i in sorted(list_str)]:
    print(a)


