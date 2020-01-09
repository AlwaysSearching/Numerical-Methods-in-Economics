from typing import Callable as func

# Impliments a single iteration of the Bracketing algorithm found in Section 4.1 of Numerical Methods in Economics by Kenneth L. Judd.
def one_iteration_bracketing(
    f: func, a: float, b: float, c: float
) -> (float, float, float):
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
    if b - a < c - b:
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
def bracketing(
    f: func, triple: [float, float, float], maxiter: int = 100, eps: float = 1e-10
) -> (float, float, float):
    """
    Runs the Bracketing method untill either a specified number of iterations is reached, or when a stopping criterion is reached.

    Parameters:
        f: The function whose minimum value we are trying to find

        triple: An initial set of three points such that (triple[0] < triple[1] < triple[2]) and f_(triple[0]), f_(triple[2]) > f_(triple[1])

        maxiter: Maximum Number of iterations to run. Initialized to 100.

        eps: Stopping criterion for the function.

    Returns:
        (a, b, c): Returns the newly computed set of points found after running the Bracketing Method.
    """
    # Simply call iterations of the function untill one of the conditions are reached.
    # Initialize the number of iterations to 0, and the error to Inifite.
    n = 0

    while n < n_iter:
        triple = one_iteration_bracketing(f, *triple)
        if triple[2] - triple[0] < eps:
            break
        n += 1

    return triple


def newton(
    x_0: float,
    f: func,
    fprime: func,
    tol: float = 1e-5,
    maxiter: int = 20,
    epsilon: float = 1e-14,
) -> (float, str):
    """
    Runs Newtons Method for root finding. Differs slightly from Newtons method applied to minimization.

    Parameters:
        x_0: An intial value that is hopefully close to the root of our function.

        f: The function whose roots we are trying to find

        fprime: the derivative of the function f.

        tol: Stopping criterion for the function.

        maxiter: Maximum Number of iterations to run. Initialized to 20.

        epsilon: Smalledt allowable value for our derivative. Stops method when fprime < epsilon

    Returns:
        (float, str): Returns the approximation of the functions root as well as a string indicating whether the method
                      converged, or stopped due to having reached either the maximum number of iterations, or if fprime < epsilon.
    """
    x_0 = float(x_0)  # If fprime is an autograd function, it requires float.
    n = 0

    while n < maxiter:
        y = f(x_0)
        yprime = fprime(x_0)

        if abs(yprime) < epsilon:
            return x_0, "Root not Found. Derivative value near 0."

        x_1 = x_0 - y / yprime

        if abs(x_1 - x_0) <= tol:
            return (
                x_1,
                "Root Found. Difference between iterations meets given tolerance.",
            )

        x_0 = x_1

    return x_0, "Max iterations reached before given tolerance was met."
