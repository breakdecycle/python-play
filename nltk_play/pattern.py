from pattern.en import parsetree

s = parsetree('The cat sat on the mat.')
for sentence in s:
    for chunk in sentence.chunks:
        print (chunk.type, [(w.string, w.type) for w in chunk.words])