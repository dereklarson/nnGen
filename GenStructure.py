#!/home/derek/anaconda/bin/python
import os, sys, getopt

def usage():
    print "<exec> [ -h -l <layers> -s <neurons> -d <dropout> -b <i_drop> -r <decay>]"
    sys.exit()

try:
    opts, args = getopt.getopt(sys.argv[1:], "hl:s:d:b:r:")
except getopt.GetoptError as err:
    print str(err)
    usage()
#default values
n_layers = 2
neurons = 500
dropout = 0.5
i_dropout = 0.5
decay = 0.0
for opt, val in opts:
    if opt == '-l':
        n_layers = int(val)
    elif opt == '-s':
        neurons = int(val)
    elif opt == '-d':
        dropout = float(val)
    elif opt == '-b':
        i_dropout = float(val)
    elif opt == '-r':
        decay = float(val)
    elif opt in ("-h", "--help"):
        usage()
    else:
        assert False, "unhandled option"
print n_layers, neurons, dropout

#Write input layer
S_file = open("Structure", 'w')
jitter = 0
flipX = flipY = 0
S_file.write("#HDR:  Structure for CNN\n##layer 0\ntype:\t\tInput\n")
S_file.write("jitter:\t\t" + str(jitter) + "\n")
S_file.write("flipX:\t\t" + str(flipX) + "\n")
S_file.write("flipY:\t\t" + str(flipY) + "\n")
S_file.write("dropout:\t" + str(i_dropout) + "\n\n")

#Write middle layers
for i in range(n_layers):
    S_file.write("##layer " + str(i+1) + "\ntype:\t\tFC\n")
    S_file.write("neurons:\t" + str(neurons) + "\n")
    S_file.write("dropout:\t" + str(dropout) + "\n")
    S_file.write("decayW:\t\t" + str(decay) + "\n")
    S_file.write("activation:\tReLU\n\n")

#Write the output layer
S_file.write("##layer out\ntype:\t\tOutput\nneurons:\t2\n")
S_file.write("activation:\tRectScaleTanh\n")

