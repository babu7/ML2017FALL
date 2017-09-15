import sys
wordlist = []
with open(sys.argv[1], "r") as infile:
    for line in infile:
        for i in line.split():
            wordlist.append(i)

out = []
for word in wordlist:
    for obj in out:
        if obj['word'] == word:
            obj['count'] += 1
            break
    else:
        out.append({'word': word, 'count': 1})

with open("Q1.txt", "w") as fout:
    for idx, obj in enumerate(out):
        if idx != len(out)-1:
            print("%s %d %d" % (obj['word'], idx, obj['count']), file=fout)
        else:
            # Seems not necessary
            print("%s %d %d" % (obj['word'], idx, obj['count']), file=fout, end='')
