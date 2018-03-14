import sys
if(len(sys.argv)!=3):
    print("Usage: transfrom_data.py input_file output_file")
    sys.exit(0)
open(sys.argv[2], 'w').write('\n'.join(map(lambda line: '0 '+' '.join(map(lambda a:'{}:{}'.format(*a),enumerate(map(lambda x:x/255,map(float,line.rstrip().split(','))),1))),open(sys.argv[1], 'r').read().strip().split('\n'))))