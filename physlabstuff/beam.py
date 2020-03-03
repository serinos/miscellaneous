"""
Feb 17 2020 - OS
Numerical Calculation Tool for Gaussian Beams Masked by Semi-ablated Absorbant Surface

***Functions:
-- beam_initialize(res, threshold, Ep, w, over_est)  outputs a np.array matrix with I values
wrt x,y coordinates with desired resolution, calculates only first quadrant
values under y=x, then extends by symmetry to the rest of cartesian plane
over_est flag determines over/underestimation, which can be used for error calcs
Equation1: J(x,y) = (2*Ep/pi*w^2)*exp((-2*x^2 - 2*y^2)/w^2)
-- mask_initialize(beam, <shape params>, thickness, Is, a0)  outputs mask with
desired shape for a given beam, for now only straight lines are to be implemented
-- mask_apply(beam, mask)  Applies the following eqn:
Equation2: Jnew := J - deltaJ where deltaJ := J*(aS + a0/(1 + J/Js))
-- integrate_for_power(beam)  Adds up J values in a beam matrix, finds Ep
-- multi_integrate_for_Ep (beamlist)  Yields a list of tuples in format (index, power)
-- plot_heat(beam or mask)  Plots heat graph of beam/mask
-- plot_power(beamlist)  Plots power graph of beams in a list
-- mask_slide(beam, mask, stepsX, stepsY)  Slides mask on beam, returns a tuple of beams.
-- mask_draw(pad, dim, crop)  Draws a mask by using the pad repetitively to achieve square matrix,
edges have at least 1 and at most 2 extra pads to ensure proper working of mask_slide(), crop
equals 1 returns cropped matrix to match dim

TODO: Add different mask shapes after zebra pattern is understood satisfactorily
TODO: Multiprocessing cannot join threads without timeout for some reason, fix it by rewrite?
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Process, Queue, cpu_count


class DimensionMismatch(Exception): pass
class UnsufficientParameters(Exception): pass

class Beam:
    def __init__(self, res, Ep, w, dim, matrix, over_est):
        self.res = res
        self.Ep = Ep
        self.w = w
        self.dim = dim
        self.matrix = matrix
        self.over_est = over_est

class Mask:
    def __init__(self, shape, width, thickness, dim, matrix, pad, Js, a0, aS):
        self.shape = shape
        self.width = width
        self.thickness = thickness
        self.dim = dim
        self.matrix = matrix
        self.pad = pad
        self.Js = Js
        self.a0 = a0
        self.aS = aS


def beam_initialize(res=1, threshold=(10**-5), Ep=1, w=0, over_est=True):
    if(w==0):
        raise DimensionMismatch
    w2 = w**2
    #  By threshold, calculate how many cells will be traversed, assign to
    #  variable "cut", uses Eqn1 to find x and assign int wrt res. (y=0)
    #  For cut, over_est=True always to ensure comparability with over_est=False matrices
    cut = int(np.ceil(res*np.sqrt(((w2)*np.log(2*Ep/(np.pi*(w2)*threshold)))/2)))
    #  Iterate cut times for first line, then use sin function to truncate
    #  after threshold and fill in zeros, will traverse the first quadrant
    #  below y=x and used for filling above y=x,first quad data is put in a list
    #  Over/underestimation fix is done after first quadrant is calculated
    #  with overestimation
    const = np.float32(2*Ep/(np.pi * w2))  # For later use in value assigning with Eqn1
    y_counter = 0
    first_quad = []
    for i in range(cut):
        x_traversal = []
        fix = int(np.sqrt(cut**2 - y_counter**2))  # Used for zeroing out values below thrsh
        for k in range(y_counter):
            x_traversal.append(first_quad[k][y_counter])
        for j in range(y_counter, fix):
            #  Here comes Eqn1 to be assigned to variable "value"
            value = const*np.exp(-2*((j/res)**2 + (y_counter/res)**2)/w2, dtype="float32")
            x_traversal.append(value)
        for l in range(max(fix, y_counter), cut):
            x_traversal.append(0)

        first_quad.append(x_traversal)
        y_counter += 1

    # Fixing for over_est flag:
    if(over_est == False):
        _ = first_quad.pop(0)
        first_quad.append([0 for i in range(cut)])
        for i in range(cut):
            _ = first_quad[i].pop(0)
            first_quad[i].append(0)
    # Now, time for extending to other quadrants
    total_matrix = []
    for i in range(cut):  # 1st quad used for 2nd quad
        total_line = []
        for j in reversed(range(cut)):
            total_line.append(first_quad[i][j])
        total_matrix.append(total_line + first_quad[i])
    total_matrix.reverse()  # Fixing rotation problem
    for i in reversed(range(cut)):  # Upper quads used for lower quads
        total_matrix.append(total_matrix[i])

    return Beam(res, Ep, w, cut*2, np.array(total_matrix), over_est)


def mask_initialize(Js=0.015, a0=0.01725, aS=0.00575, **kwargs):  # Js = 0.015 J/cm2
    mask = []
    try:
        shape = kwargs.pop("shape")
        beam = kwargs.pop("beam")
    except:
        print("Parameters not sufficient")
        raise UnsufficientParameters
    try:
        crop_flag = kwargs.pop("crop")
    except:
        crop_flag = True

    if(shape=="lines"):
        width = kwargs.pop("width")
        thickness = kwargs.pop("thickness")
        digital_thickness = int(np.ceil(thickness * beam.res))  # Note: Ceiling thickness
        digital_width = int(np.ceil(width * beam.res))
        square_len = digital_thickness + digital_width
        pad = np.vstack((np.zeros((digital_thickness, square_len)), np.ones((digital_width, square_len))))
        mask = mask_draw(pad=pad, dim=beam.dim, crop=crop_flag)

    elif(shape=="dots"):
        pad = np.array([[1,0],[0,1]])
        mask = mask_draw(pad=pad, dim=beam.dim, crop=crop_flag)

    else:
        return 0

    return Mask(shape, width, thickness, mask.shape[0], mask, pad, Js, a0, aS)


def mask_apply(beam: Beam, mask: Mask):
    if(beam.dim != mask.dim):
        raise DimensionMismatch
    Js = mask.Js
    a0 = mask.a0
    aS = mask.aS
    new_matrix = []
    # Eqn2 will be iterated for cells that are passing through absorbant medium
    for i in range(beam.dim):  # Traverses y coord.
        line = []
        for j in range(beam.dim):  # Traverses x coord
            value = beam.matrix[i][j]
            if(bool(mask.matrix[i][j]) & bool(value)):  # Only work on filled cells
               line.append(value*(1-(aS+(a0/(1+(value/Js))))))
            else:
               line.append(value)

        new_matrix.append(line)

    return Beam(beam.res, beam.Ep, beam.w, beam.dim, new_matrix, beam.over_est)


def mask_slide(beam: Beam, mask: Mask, stepsX=0, stepsY=0):  # Pass cropless masks only
    #  Multiprocessing does not work for some reason
    pad_Y = mask.pad.shape[0]
    pad_X = mask.pad.shape[1]
    if((beam.dim > mask.dim) | ((stepsX == 0) & (stepsY == 0))):
        raise DimensionMismatch
    try:
        step_sizeY = pad_Y//(stepsY)
        if(step_sizeY == 0):
            step_sizeY = 1
    except:
        step_sizeY = 0
    try:
        step_sizeX = pad_X//(stepsX)
        if(step_sizeX == 0):
            step_sizeX = 1
    except:
        step_sizeX = 0
    try:
        slope = stepsY//stepsX
    except:
        slope = np.inf

    configs = []  #  Config elements are tuples: (Y axis, X axis)
    if(step_sizeX == 0):
        configs = [(i,0) for i in range(0, pad_Y, step_sizeY)]
    elif(step_sizeY == 0):
        configs = [(0,i) for i in range(0, pad_X, step_sizeX)]
    elif(step_sizeY > step_sizeX):
        for i in range(0, pad_X, step_sizeX):
            if(i*slope <= pad_Y):
                configs.append((i*slope, i))
            else:
                break
    else:
        for i in range(0, pad_Y, step_sizeY):
            if(i//slope <= pad_X):
                configs.append((i, int(i//slope)))
            else:
                break

    processes = []
    q = Queue()
    cpu = cpu_count()
    ranger = (len(configs)//cpu)+1
    #Timeout fix for process.join, processes will terminate after sufficient time
    timeout = time()
    _ = mask_apply(beam=beam, mask=Mask(mask.shape, mask.width, mask.thickness, beam.dim,\
    mask.matrix[0:beam.dim, 0:beam.dim], mask.pad, mask.Js, mask.a0, mask.aS))
    timeout = time() - timeout + 0.02  #  Too much time, try to lower it
    print(f"Best case: {timeout * ranger} seconds")  #  Not precise at all...
    del(_)
    ###

    for i in range(ranger):
        for j in range(cpu):
            try:
                config = configs.pop()
            except:
                break
            process = Process(target=_mask_apply, args=(q, config, beam, mask))
            processes.append(process)
            process.start()
        for process in processes:
            process.join(timeout)  # Dirty fix to force calculations

    returnee = []
    for i in range(q.qsize()):
        returnee.append(q.get())
    returnee.sort()
    return returnee


def _mask_apply(q, config, beam, mask):  #Note that config[0]: Y axis
    mask.matrix = mask.matrix[config[0]:(beam.dim + config[0]), config[1]:(beam.dim+config[1])]
    mask.dim = beam.dim
    beam = mask_apply(beam, mask)
    index = config[0] + config[1]
    beam_tuple = (index, beam)
    q.put(beam_tuple)


def mask_draw(pad: np.ndarray, dim: int, crop=True):
    if(pad.shape[0] > dim):
        raise DimensionMismatch
    pad = np.vstack(tuple(pad for i in range((dim//pad.shape[0])+2)))
    pad = np.hstack(tuple(pad for i in range((dim//pad.shape[1])+2)))
    if(crop):
        return pad[0:dim,0:dim]
    else:
        return pad


def plot_heat(beam: Beam):
    plt.imshow(beam.matrix, cmap='viridis')
    plt.colorbar()
    plt.show()


def integrate_for_power(beam: Beam):
    # TODO: Add a flag for calculating error range, also add support for lists of
    # beams cooked up by mask_slide()
    dA = 1 / (beam.res**2)  # dA for integration by adding up squares
    power = 0
    for i in beam.matrix:
        for j in i:
            power += (dA * j)
    return np.float32(power)


def _integrate_for_power(q, index, beam):
    dA = 1 / (beam.res**2)
    power = 0
    for i in beam.matrix:
        for j in i:
            power += (dA * j)
    q.put((index, np.float32(power)))


def multi_integrate_for_power(beamlist):
    #Timeout fix for process.join, processes will terminate after sufficient time
    timeout = time()
    integrate_for_power(beam=beamlist[0][1])
    timeout = time() - timeout + 0.02
    print(f"Best case: {timeout * ((len(beamlist)//cpu_count())+1)} seconds")
    ###
    beamlist = beamlist.copy()
    processes = []
    q = Queue()
    cpu = cpu_count()
    ranger = (len(beamlist)//cpu)+1

    for i in range(ranger):
        for j in range(cpu):
            try:
                config = beamlist.pop()
            except:
                break
            process = Process(target=_integrate_for_power, args=(q, config[0], config[1]))
            processes.append(process)
            process.start()
        for process in processes:
            process.join(timeout)  # Dirty fix to force calculations

    returnee = []
    for i in range(q.qsize()):
        returnee.append(q.get())
    returnee.sort()
    return returnee

def plot_power(powerlist):
    plt.plot([i for i in range(len(powerlist))],[j[1] for j in powerlist])
    plt.show()
