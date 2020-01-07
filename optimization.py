from typing import Callable as func

# Impliments a single iteration of the Bracketing algorithm found in Section 4.1 of Numerical Methods in Economics by Kenneth L. Judd.
def one_iteration_bracketing(f: func, a: float, b: float, c: float) -> (float, float, float):
    """
    Runs one iteration of the Bracketing method. 
    
    Parameters:
        f: The function whose minimum value we are trying to find
    
        a: Minimum value of the three values passed. The Method requires that f_(a) > f_(b).
        
        b: The center value of the three floats passed (a < b < c). The Method requires that f_(a), f_(c) > f_(b).
        
        c: Maximum value of our three points. The Method requires that f_(c) > f_(b).
        
    Returns:
        (a, b, c): Returns the newly computed set of points found after a single iteration.
    """
    
    # First select our new point as either the mid point between (a, b) or (b, c)
    if b-a < c-b :
        d = (b + c) / 2
    else: 
        d = (a + b) / 2
    
    # Compute the value at the new point d. 
    f_d = f(d)
    
    # Finally select the new triple to return.
    if d < b:
        if f(b) < f(d):
            return (d, b, c)
        else:
            return (a, d, b)
    else:
        if f(b) < f(d):
            return (b, d, c)
        else:
            return (a, b, d)
        
        
# Impliments a full Bracketing algorithm found in Section 4.1 of Numerical Methods in Economics by Kenneth L. Judd. 
def bracketing(f: func, triple: (float, float, float), n_iter: int = 100, eps: float = 0) -> (float, float, float): 
    """
    Runs the Bracketing method untill either a specified number of iterations is reached, or when a stopping criterion is reached. 
    
    Parameters:
        f: The function whose minimum value we are trying to find
    
        triple: An initial set of three points such that (triple[0] < triple[1] < triple[2]) and f_(triple[0]), f_(triple[2]) > f_(triple[1])
        
        n_iter: Desired Number of iterations to run. Initialized to 1000.
        
        eps: Stopping criterion for the function. 
        
    Returns:
        (a, b, c): Returns the newly computed set of points found after running the Bracketing Method.
    """
    # Simply call iterations of the function untill one of the conditions are reached.
    # Initialize the number of iterations to 0, and the error to Inifite.
    n = 0; 
    
    while n < n_iter:
        triple = one_iteration_bracketing(f, *triple)
        if triple[2] - triple[0] < eps:
            break
        n += 1
        
    return triple
    