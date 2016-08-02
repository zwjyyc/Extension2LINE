import sys


def filter(rawfile, unfilteredfile, filteredfile):
    
    pairss = {}
    
    with open(rawfile, 'r') as f:
        for x in f.readlines():
            items = x.strip().split()
            pairss[items[0] + '_' + items[1]] = ''
            pairss[items[1] + '_' + items[2]] = ''
            pairss[items[2] + '_' + items[0]] = ''
    
    sys.stdout.write('train set has ' + str(len(pairss)) + ' pairs!\n')    
    sys.stdout.flush()

    with open(unfilteredfile, 'r') as fin:
        with open(filteredfile, 'w') as fout:
            for x in fin.readlines():
                items = x.strip().split()
                if pairss.has_key(items[0] + '_' + items[1]) or pairss.has_key(items[1] + '_' + items[0]):
                    continue;
                else:
                    #sys.stdout.write('ok! find it!')
                    #sys.stdout.flush()
                    fout.write(x)
                    fout.flush()
    

if __name__ == "__main__":
    raw_file = sys.argv[1]
    un_filtered_file = sys.argv[2]
    filtered_file = sys.argv[3]
    filter(raw_file, un_filtered_file, filtered_file)
    
