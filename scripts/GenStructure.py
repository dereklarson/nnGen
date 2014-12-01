#!/home/derek/anaconda/bin/python
""" Generates a feed-forward MLP structure, setting a few parameters:
    number of layers
    neurons per layer
    output neurons
    dropout
    input dropout
    jitter
    L2 decay
"""
import os, sys, getopt

def usage():
    print "<exec> [ -h -l <layers> -s <neurons> -d <dropout> -b <i_drop> -r <decay>]"
    sys.exit()

try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:D:j:l:L:n:o:")
except getopt.GetoptError as err:
    print str(err)
    usage()
#default values
n_layers = 2
neurons = 800
o_neurons = 10
dropout = 0.5
i_dropout = 0.2
decay = 0.0
jitter = 0
for opt, val in opts:
    if opt == '-d':
        dropout = float(val)
    elif opt == '-D':
        i_dropout = float(val)
    elif opt == '-j':
        jitter = int(val)
    elif opt == '-l':
        n_layers = int(val)
    elif opt == '-L':
        decay = float(val)
    elif opt == '-n':
        neurons = int(val)
    elif opt == '-o':
        o_neurons = int(val)
    elif opt in ("-h", "--help"):
        usage()
    else:
        assert False, "unhandled option"

print 'Input dropout:', i_dropout
print 'Layers:', n_layers, 'by', neurons, 'dropout:', dropout, 'L2:', decay
print 'Output neurons:', o_neurons

#Write input layer
S_file = open("structure", 'w')
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
S_file.write("##layer out\ntype:\t\tOutput\nneurons:\t"+ str(o_neurons) +"\n")
#S_file.write("activation:\tRectScaleTanh\n")

