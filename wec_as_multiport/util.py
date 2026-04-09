# © 2024 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.

from scipy.constants import golden
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.optimize import fsolve
import numpy as np

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.sankey import Sankey
from typing import Tuple

def figsize(wf=1, hf=1, columnwidth=250):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    https://stackoverflow.com/questions/29187618/matplotlib-and-latex-beamer-correct-size/30170343
    """

    hf = hf/golden
    fig_width_pt = columnwidth*wf
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]

def find_maximum_with_interpolation(x_data, y_data, method='cubic', bounds=None):
    """
    Sandia AI
    Finds the maximum value of a function defined by data points using interpolation.

    Parameters:
    - x_data: array-like, the x-coordinates of the data points.
    - y_data: array-like, the y-coordinates of the data points.
    - method: str, the type of interpolation ('linear', 'quadratic', 'cubic', etc.).
    - bounds: tuple, the bounds for the x-values to search for the maximum (min_x, max_x).

    Returns:
    - max_x: the x-coordinate of the maximum point.
    - max_y: the maximum value of the interpolated function.
    """
    
    # Step 1: Interpolate the data
    interpolator = interp1d(x_data, y_data, kind=method, fill_value="extrapolate")

    # Step 2: Define a function for optimization
    def interpolated_function(x):
        return interpolator(x)

    # Step 3: Find the maximum using optimization
    if bounds is None:
        bounds = (min(x_data), max(x_data))
    
    result = minimize_scalar(lambda x: -interpolated_function(x), bounds=bounds, method='bounded')

    # Get the maximum value and corresponding x
    max_x = result.x
    max_y = -result.fun

    return max_x, max_y
        
        
def find_zero_crossings(x, y):
    """Generated from Google AI: Finds zero crossings in a 1D signal.

    Args:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.

    Returns:
        array: The x-coordinates of the interpolated zero crossings.
    """

    # Find indices where the sign of y changes
    sign_changes = np.where(np.diff(np.sign(y)))[0]

    # Interpolate the zero crossings
    zero_crossings = []
    for i in sign_changes:
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]

        # Use linear interpolation to find the zero crossing
        zero_crossing = x1 - y1 * (x2 - x1) / (y2 - y1)
        zero_crossings.append(zero_crossing)

    return np.array(zero_crossings)

def depth_function(k, h=None):
    if h is None:
        h = np.infty
        D = 1
    else:
        D = (1 + 2*k*h/np.sinh(2*k*h))*np.tanh(k*h)
    return D

def w2k(w, h=None, g=9.81):
    """Radial frequency to wave number"""
    if h is None:
        h = np.infty
    func = lambda k: __dispersion__(k=k, w=w, h=h, g=g)
    x0 = w**2/g
    return fsolve(func, x0=x0)[0]

def k2w(k, h=None, g=9.81):
    """Wave number to radial frequency"""
    if h is None:
        h = np.infty
    func = lambda w: __dispersion__(k=k, w=w, h=h, g=g)
    x0 = np.sqrt(g*k)
    return fsolve(func, x0=x0)[0]

def __dispersion__(k, w, h=None, g=9.81):
    """Dispersion relationship"""
    if h is None:
        h = np.infty
    return w**2 - g * k * np.tanh(k * h)

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.sankey import Sankey
from typing import Tuple
def power_flow_colors():
    """
    Define and return a dictionary of colors to represent different stages of the power flow through a WEC.

    The function creates a dictionary where each key corresponds to a specific stage of power flow,
    and each value is a tuple representing an RGBA color. The colors are derived from the 'viridis' 
    colormap, which is perceptually uniform.

    Returns:
        dict: A dictionary containing the following stages of power flow and their associated colors:
            - 'exc': Color for excitation power (RGBA: (0.267004, 0.004874, 0.329415, 1.0))
            - 'rad': Color for radiated power (RGBA: (0.229739, 0.322361, 0.545706, 1.0))
            - 'abs': Color for absorbed power (RGBA: (0.127568, 0.566949, 0.550556, 1.0))
            - 'use': Color for useful power (RGBA: (0.369214, 0.788888, 0.382914, 1.0))
            - 'elec': Color for electrical power (RGBA: (0.974417, 0.90359, 0.130215, 0.5))

    Example:
        colors = power_flow_colors()
        print(colors['exc'])  # Output: (0.267004, 0.004874, 0.329415, 1.0)
    """
    clrs = {'exc':        (0.267004, 0.004874, 0.329415, 1.0), #viridis(0.0)
        'rad':   (0.229739, 0.322361, 0.545706, 1.0), #viridis(0.25)
        'abs':         (0.127568, 0.566949, 0.550556, 1.0), #viridis(0.5)
        'use':    (0.369214, 0.788888, 0.382914, 1.0), #viridis(0.75)
        'elec':         (0.974417, 0.90359, 0.130215, 0.5), #viridis(0.99)
        }
    return clrs
def plot_power_flow(power_flows: dict[str, float], 
                    plot_reference: bool = True,
                    axes_title: str = '', 
                    axes: Axes = None,
                    return_fig_and_axes: bool = False
    )-> Tuple[Figure, Axes]:
    """Plot power flow through a Wave Energy Converter (WEC) as a Sankey diagram.

    This function visualizes the power flow through a WEC by creating a Sankey diagram.
    If the model does not include mechanical and electrical components, customization of this function will be necessary.

    Parameters
    ----------
    power_flows : dict[str, float]
        A dictionary containing power flow values produced by, for example,
        :py:func:`wecopttool.utilities.calculate_power_flows`.
        Required keys include:
            - 'Optimal Excitation'
            - 'Deficit Excitation'
            - 'Excitation'
            - 'Deficit Radiated'
            - 'Deficit Absrobed'
            - 'Radiated'
            - 'Absorbed'
            - 'Electrical'
            - 'Useful'
            - 'PTO Loss Mechanical'
            - 'PTO Loss Electrical'

    plot_reference : bool, optional
        If True, the optimal absorbed reference powers will be plotted. Default is True.
    
    axes_title : str, optional
        A string to display as the title over the Sankey diagram. Default is an empty string.
    
    axes : Axes, optional
        A Matplotlib Axes object where the Sankey diagram will be drawn. If None, a new figure and axes will be created. Default is None.
    
    return_fig_and_axes : bool, optional
        If True, the function will return the Figure and Axes objects. Default is False.

    Returns
    -------
    tuple[Figure, Axes] or None
        A tuple containing the Matplotlib Figure and Axes objects if `return_fig_and_axes` is True.
        Otherwise, returns None.

    Example
    -------
    power_flows = {
        'Optimal Excitation': 100,
        'Deficit Excitation': 30,
        'Excitation': 70,
        'Deficit Radiated': 20,
        'Deficit Absorbed': 10,
        'Radiated': 30,
        'Absorbed': 40,
        'Electrical': 30,
        'Useful': 35,
        'PTO Loss Mechanical': 5,
        'PTO Loss Electrical': 5
    }
    plot_power_flow(power_flows, axes_title='Power Flow Diagram')
    """

    if axes is None:
        fig, axes = plt.subplots(nrows = 1, ncols= 1,
                tight_layout=True, 
                figsize= [8, 4])
    clrs = power_flow_colors()
    len_trunk = 1.0
    if plot_reference:
        sankey = Sankey(ax=axes, 
                        scale= 1/power_flows['Optimal Excitation'],
                        offset= 0,
                        format = '%.1f',
                        shoulder = 0.02,
                        tolerance=1e-03*power_flows['Optimal Excitation'],
                        unit = 'W')
        sankey.add(flows=[power_flows['Optimal Excitation'],
                    -1*power_flows['Deficit Excitation'],
                    -1*power_flows['Excitation']], 
            labels = [' Optimal \n Excitation ', 
                    'Deficit \n Excitation', 
                    'Excitation'], 
            orientations=[0, 0,  0],#arrow directions,
            pathlengths = [0.15,0.15,0.15],
            trunklength = len_trunk,
            edgecolor = 'None',
            facecolor = clrs['exc'],
                alpha = 0.1,
            label = 'Reference',
                )
        n_diagrams = 1
        init_diag  = 0
        if power_flows['Deficit Excitation'] > 0.1:
            sankey.add(flows=[power_flows['Deficit Excitation'],
                        -1*power_flows['Deficit Radiated'],
                        -1*power_flows['Deficit Absorbed'],], 
                labels = ['XX Deficit Exc', 
                        'Deficit \n Radiated',
                            'Deficit \n Absorbed', ], 
                prior= (0),
                connect=(1,0),
                orientations=[0, 1,  0],#arrow directions,
                pathlengths = [0.15,0.01,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['rad'],
                alpha = 0.3, #viridis(0.2)
                label = 'Reference',
                    )
            n_diagrams = n_diagrams +1
    else:
        sankey = Sankey(ax=axes, 
                        scale= 1/power_flows['Excitation'],
                        offset= 0,
                        format = '%.1f',
                        shoulder = 0.02,
                        tolerance=1e-03*power_flows['Excitation'],
                        unit = 'W')
        n_diagrams = 0
        init_diag = None

    sankey.add(flows=[power_flows['Excitation'],
                        -1*(power_flows['Absorbed'] 
                           + power_flows['Radiated'])], 
                labels = ['Excitation', 
                        'Excitation'], 
                prior = init_diag,
                connect=(2,0),
                orientations=[0,  -0],#arrow directions,
                pathlengths = [.15,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['exc'] #viridis(0.9)
        )
    sankey.add(flows=[
                (power_flows['Absorbed'] + power_flows['Radiated']),
                -1*power_flows['Radiated'],
                -1*power_flows['Absorbed']], 
                labels = ['Excitation', 
                        'Radiated', 
                        'Absorbed'], 
                # prior= (0),
                prior= (n_diagrams),
                connect=(1,0),
                orientations=[0, -1,  -0],#arrow directions,
                pathlengths = [0.15,0.2,0.15],
                trunklength = len_trunk-0.2,
                edgecolor = 'None', 
                facecolor = clrs['rad'] #viridis(0.5)
        )
    sankey.add(flows=[power_flows['Absorbed'],
                        -1*power_flows['PTO Loss Mechanical'],                      
                        -1*power_flows['Useful']], 
                labels = ['Absorbed', 
                        'PTO-Loss Mechanical' ,                           
                        'Useful'], 
                prior= (n_diagrams+1),
                connect=(2,0),
                orientations=[0, -1, -0],#arrow directions,
                pathlengths = [.15,0.2,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['abs'] #viridis(0.9)
        )
    sankey.add(flows=[(power_flows['Useful']),
                        -1*power_flows['PTO Loss Electrical'],
                        -1*power_flows['Electrical']], 
                labels = ['Useful', 
                        'PTO-Loss Electrical' , 
                        'Electrical'], 
                prior= (n_diagrams+2),
                connect=(2,0),
                orientations=[0, -1,  -0],#arrow directions,
                pathlengths = [.15,0.2,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['use'] #viridis(0.9)
        )
    sankey.add(flows=[(power_flows['Electrical']),
                        -1*power_flows['Electrical']], 
                labels = ['', 
                        'Electrical'], 
                prior= (n_diagrams+3),
                connect=(2,0),
                orientations=[0,  -0],#arrow directions,
                pathlengths = [.15,0.15],
                trunklength = len_trunk,
                edgecolor = 'None',
                facecolor = clrs['elec'] #viridis(0.9)
        )


    sankey.ax.axis([sankey.extent[0] - sankey.margin,
                      sankey.extent[1] + sankey.margin,
                      sankey.extent[2] - sankey.margin,
                      sankey.extent[3] + sankey.margin])
    sankey.ax.set_aspect('equal', adjustable='box') 
    diagrams = sankey.diagrams
    for diagram in diagrams:
        for text in diagram.texts:
            text.set_fontsize(8)

    #Remvove labels that are double
    len_diagrams = len(diagrams)

    diagrams[len_diagrams-4].texts[0].set_text('') #remove exciation from hydro
    diagrams[len_diagrams-5].texts[-1].set_text('') #remove excitation from excitation
    diagrams[len_diagrams-3].texts[0].set_text('') #remove absorbed from absorbed
    diagrams[len_diagrams-2].texts[0].set_text('') #remove use from use-elec
    diagrams[len_diagrams-2].texts[-1].set_text('') #remove electrical from use-elec
    diagrams[len_diagrams-1].texts[0].set_text('')  #remove electrical in from elec

    if len_diagrams > 5:
        axes.legend()   #add legend for the reference arrows
    if len_diagrams >6:
      diagrams[1].texts[0].set_text('') 

    axes.set_aspect('equal')

    axes.set_title(axes_title)
    axes.axis("off")

    if return_fig_and_axes:
        return fig, axes

