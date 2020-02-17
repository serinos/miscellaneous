"""
Feb 17 2020 - OS
Numerical Calculation Tool for Gaussian Beams Masked by Absorbant Surface

***Functions:

--beam_initialize(res, threshold, P0, w, est)  outputs a np.array matrix with I values
wrt x,y coordinates with desired resolution, calculates only first quadrant
values under y=x, then extends by symmetry to the rest of cartesian plane
est flag determines over/underestimation, which can be used for error calcs
Equation1: I(x,y) = (2*P0/pi*w^2)*exp((-2*x^2 - 2*y^2)/w^2)
--mask_initialize(beam, <shape params>, thickness, Is, a0)  outputs mask with
desired shape for a given beam, for now only straight lines are to be implemented
--mask_apply(beam, mask)  Applies the following eqn:
Equation2: Inew := I - deltaI where deltaI := I*a0/(1 + I/Is)
--integrate_for_power(beam)  Adds up I values in a beam matrix, finds P
--change_P0(matrix)  Creates a new beam instance with different P0
--plot_heat(matrix)  Plots heat graph of beam/mask
--mask_slide(beam, mask, steps)  Slides mask on beam, returns a tuple of beams.
--mask_draw(pad, dim, crop)  Draws a mask by using the pad repetitively to achieve square matrix,
edges have at least 1 and at most 2 extra pads to ensure proper working of mask_slide(), crop
equals 1 returns cropped matrix to match dim

***Data structures:

--beam: is a tuple (res, P0, w, dimensions, matrix, est)
Note that matrix is in np.ndarray type
--mask: is a tuple (shape, width, thickness, dimensions,  matrix, pad, Is, a0)
Note that pad, matrix in np.ndarray type
--res: is int; e.g. res=2: 4 values per 1 unit square of Intensity plane

**could refactor the structures to be more obj oriented; low priority
also tuples are easier to follow than naming stuff

TODO: After implementing mask initializer, write a sliding function and
some analysis tools for it. Also code some fft funtion for some fun comparisons?

TODO: Initiating masks with pandas datasheet of pad makes sense
for sliding function, pad can be used to generate on-the-fly masks with
sled config

TODO: Add import&export function for data
"""

import numpy as np
import matplotlib.pyplot as plt

class DimensionMismatch(Exception): pass


def beam_initialize(res=1, threshold=(10**-5), P0=1, w=0, est=1):
    if(w == 0):
        return 0
    w2 = w**2
    #  By threshold, calculate how many cells will be traversed, assign to
    #  variable "cut", uses Eqn1 to find x and assign int wrt res. (y=0)
    #  For cut, est=1 always to ensure comparability with est=0 matrices
    cut = int(np.ceil(res*np.sqrt(((w2)*np.log(2*P0/(np.pi*(w2)*threshold)))/2)))
    #  Iterate cut times for first line, then use sin function to truncate
    #  after threshold and fill in zeros, will traverse the first quadrant
    #  below y=x and used for filling above y=x,first quad data is put in a list
    #  Over/underestimation fix is done after first quadrant is calculated
    #  with overestimation
    const = np.float32(2*P0/(np.pi * w2))  # For later use in value assigning with Eqn1
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

    # Fixing for est flag:
    if(est == 0):
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

    return (res, P0, w, cut*2, np.array(total_matrix), est)


def mask_initialize(Is=1.0, a0=1.0, **kwargs):
    mask = []
    try:
        shape = kwargs.pop("shape")
        beam = kwargs.pop("beam")
    except:
        print("Parameters not sufficient")
    dim = beam[3]

    if(shape=="lines"):
        width = kwargs.pop("width")
        thickness = kwargs.pop("thickness")
        digital_thickness = int(np.ceil(thickness * beam[0]))  # Note: Ceiling thickness
        digital_width = int(np.ceil(width * beam[0]))
        pad = np.vstack((np.zeros((digital_thickness, dim)), np.ones((digital_width, dim))))
        mask = mask_draw(pad=pad, dim=dim, crop=True)

    elif(shape=="dots"):
        pad = np.array([[1,0],[0,1]])
        mask = mask_draw(pad=pad, dim=dim, crop=True)

    else:
        return 0

    return((shape, width, thickness, dim, mask, pad, Is, a0))


def mask_apply(beam: tuple, mask: tuple):
    if(beam[3] != mask[3]):
        raise DimensionMismatch
    dim = beam[3]
    Is = mask[6]
    a0 = mask[7]
    new_matrix = []
    # Eqn2 will be iterated for cells that are passing through absorbant medium
    for i in range(dim):  # Traverses y coord.
        line = []
        for j in range(dim):  # Traverses x coord
            value = beam[4][i][j]
            if(bool(mask[4][i][j]) & bool(value)):  # Only work on filled cells
               line.append(value*(1-(a0/(1+(value/Is)))))
            else:
               line.append(value)

        new_matrix.append(line)

    return((beam[0], beam[1], beam[2], dim, new_matrix, beam[5]))


def change_P0(P0: int, beam: tuple):
    pass  # TODO


def mask_slide(beam: tuple, mask: tuple, steps: int):  # NOT IMPLEMENTED YET
    if(beam[3] != mask[3]):
        raise DimensionMismatch
    pad = mask[5]
    dim = mask[3]
    mask = mask_draw(pad=pad, dim=dim, crop=False)
    # This surface will then be shifted wrt steps given, implemented for  x=y case
    try:
        step_size = pad.shape[0]//(steps-1)
        if(step_size == 0):
            step_size = 1
    except ZeroDivisionError:
        step_size = 1

    return 0


def mask_draw(pad: np.ndarray, dim: int, crop=True):
    if(pad.shape[0] > dim):
        raise DimensionMismatch
    pad = np.vstack(tuple(pad for i in range((dim//pad.shape[0])+2)))
    pad = np.hstack(tuple(pad for i in range((dim//pad.shape[1])+2)))
    if(crop):
        return pad[0:dim,0:dim]
    else:
        return pad


def plot_heat(beam: tuple):  # Mind that beam_init. returns a tuple
    plt.imshow(beam[4], cmap='viridis')
    plt.colorbar()
    plt.show()


def integrate_for_power(beam: tuple):
    # TODO: Add a flag for calculating error range, also add support for lists of
    # beams cooked up by mask_slide()
    dA = 1 / (beam[0]**2)  # dA for integration by adding up squares
    power = 0
    for i in beam[4]:
        for j in i:
            power += (dA * j)
    return np.float32(power)
