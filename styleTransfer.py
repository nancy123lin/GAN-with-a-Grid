import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import sigmoid
import numpy
import matplotlib.pyplot as plt
import time


def show_floorplan(np_floorplan, hold=True):   # one-liner for showing pic WITH/WITHOUT holding the process
    # np_floorplan should be a 3D numpy array
    plt.imshow(np_floorplan)
    if hold:
        plt.show(block=True)
    else:
        plt.show(block=False)
        time.sleep(10)
        plt.close()

bln_hold2show = False   # define whether hold to show for the whole program


# get data input, place holder
# should allow number of channels to be other than 3
#floorplan_original = numpy.random.uniform(low=0., high=1., size=(1,4,30,50))
#floorplan_style = numpy.random.uniform(low=0., high=1., size=(1,4,30,50))


def populate_sections(shape=(1,4,30,50)):       # pass a dimension of 4
    np_floorplans = numpy.ndarray(shape=shape)
    for pic in range(np_floorplans.shape[0]):
        for layer in range(np_floorplans.shape[1]):
            if layer % 2 == 0:
                int_div = numpy.random.randint(low=1, high=np_floorplans.shape[2]-1)
                np_floorplans[pic,layer,:int_div,:] = numpy.random.rand()
                np_floorplans[pic,layer,int_div:,:] = numpy.random.rand()
            else:
                int_div = numpy.random.randint(low=1, high=np_floorplans.shape[3]-1)
                np_floorplans[pic,layer,:,:int_div] = numpy.random.rand()
                np_floorplans[pic,layer,:,int_div:] = numpy.random.rand()
    return np_floorplans

floorplan_original = populate_sections()
floorplan_style = populate_sections()


# convert a tensor variable to a displayable picture
def pic_convert(floorplan, to='pic'):   # place holder
    floorplan3D = T.flatten(floorplan.dimshuffle(2,3,1,0),ndim=3).eval()
    converted = floorplan3D[:,:,:3]
    return converted


# show input images
ts_original = theano.shared(value=floorplan_original,name='ts_original')
show_floorplan(pic_convert(ts_original), hold=bln_hold2show)
ts_style = theano.shared(value=floorplan_style,name='ts_style')
show_floorplan(pic_convert(ts_style), hold=bln_hold2show)



# model construction
# a handful convolutional layers, off of which the cost function is calculated
input_original = T.tensor4(name='input_original')
input_style = T.tensor4(name='input_style')  # can expand this to be a "summary" of style features
output_ub = max([numpy.max(floorplan_original,axis=None), numpy.max(floorplan_style,axis=None)])
output_lb = max([numpy.min(floorplan_original,axis=None), numpy.min(floorplan_style,axis=None)])
# synthesized floor plan, initialized at white noise
#output_init = numpy.random.uniform(low=output_lb, high=output_ub, size=floorplan_original.shape)
output_init = populate_sections()
#output_init = floorplan_original
output = theano.shared(value=output_init,name='output')     # initialize output at white noise
show_floorplan(pic_convert(output), hold=bln_hold2show)
w1_init = numpy.ones((16,4,5,5)) * 0.01   # first layer filters
W1 = theano.shared(value=w1_init,name='W1')
w2_init = numpy.ones((32,16,3,3)) * 0.001    # second layer filters
W2 = theano.shared(value=w2_init,name='W2')
# "hard"-coding each layer, since I'm not expecting many layers
conv_wn1 = sigmoid(conv2d(input=output,filters=W1))
conv_wn2 = sigmoid(conv2d(input=conv_wn1,filters=W2))
conv_og1 = sigmoid(conv2d(input=input_original,filters=W1))
conv_og2 = sigmoid(conv2d(input=conv_og1,filters=W2))
conv_st1 = sigmoid(conv2d(input=input_style,filters=W1))
conv_st2 = sigmoid(conv2d(input=conv_st1,filters=W2))
# "hard"-coding the cost as well
wn2_flat = T.flatten(conv_wn2, ndim=1)
og2_flat = T.flatten(conv_og2, ndim=1)
loss_content = T.sum((wn2_flat-og2_flat)*(wn2_flat-og2_flat)) / 2
grammatrix_wn1 = T.tensordot(conv_wn1,conv_wn1,[[2,3],[2,3]])
grammatrix_wn2 = T.tensordot(conv_wn2,conv_wn2,[[2,3],[2,3]])
grammatrix_st1 = T.tensordot(conv_st1,conv_st1,[[2,3],[2,3]])
grammatrix_st2 = T.tensordot(conv_st2,conv_st2,[[2,3],[2,3]])
(n1, m1, _, _) = T.shape(grammatrix_wn1)
(n2, m2, _, _) = T.shape(grammatrix_wn2)
gm_wn1_flat = T.flatten(grammatrix_wn1, ndim=1)
gm_wn2_flat = T.flatten(grammatrix_wn2, ndim=1)
gm_st1_flat = T.flatten(grammatrix_st1, ndim=1)
gm_st2_flat = T.flatten(grammatrix_st2, ndim=1)
loss_style1 = T.sum((gm_wn1_flat-gm_st1_flat)*(gm_wn1_flat-gm_st1_flat)) / (4*m1*m1*n1*n1)
loss_style2 = T.sum((gm_wn2_flat-gm_st2_flat)*(gm_wn2_flat-gm_st2_flat)) / (4*m2*m2*n2*n2)
weights_stls = [1., 1.]
loss_style = weights_stls[0] * loss_style1 + weights_stls[1] * loss_style2
weights_lstp = [1., 10.]
loss_total = weights_lstp[0] * loss_content + weights_lstp[1] * loss_style
# backward propagation
[grad_W1, grad_W2, grad_output] = T.grad(loss_total, [W1,W2,output])
learning_rate = 0.001     # learning rate
W1_updated = W1 - learning_rate * grad_W1
#W1_updated = W1 - learning_rate * numpy.sqrt(numpy.absolute(grad_W1)) * numpy.sig(grad_W1) / 10
W2_updated = W2 - learning_rate * grad_W2
#W2_updated = W2 - learning_rate * numpy.sqrt(numpy.absolute(grad_W2)) * numpy.sig(grad_W2) / 10
output_updated = output - learning_rate * grad_output
#updates = [(W1, W1_updated), (W2, W2_updated), (output, output_updated)]
#f = theano.function([input_original,input_style],loss_total, updates=updates)
f = theano.function([input_original,input_style],loss_total, updates=[(W1, W1_updated)])
g = theano.function([input_original,input_style],loss_total, updates=[(W2, W2_updated)])
h = theano.function([input_original,input_style],loss_total, updates=[(output, output_updated)])



# training
max_iter = 1000     # number of iterations
# consider something like simulated annealing
for i in range(max_iter):
    loss_iter = f(floorplan_original, floorplan_style)
    loss_iter = g(floorplan_original, floorplan_style)
    loss_iter = h(floorplan_original, floorplan_style)
    if i % 100 == 0:
        print("iteration %d" % i)
        print("current loss: %.2E" % loss_iter)
        show_floorplan(pic_convert(output), hold=bln_hold2show)
        #plt.imshow(pic_convert(output))
        #plt.show(block=False)
        #plt.pause(10)
        #plt.close()









