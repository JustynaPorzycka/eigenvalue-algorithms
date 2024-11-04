import numpy as np
import sympy as sym
import helper_functions as hf

# --------------------------- Newton's method ---------------------------

def prepare_data_for_NM(coefficients):
    # Just to be able to use variables as mathematical symbols
    # when defining functions and calculating Cauchy's bound for roots
    
    # Defining function from given list of degree and coefficients
    x = sym.Symbol('x')
    f = 0
    pow = 0
    coefficients = list(reversed([float(coefficient) for coefficient in coefficients]))
    
    for coefficient in coefficients:
        f+=coefficient*x**pow
        pow += 1
    
    # Calculate derivative, second derivative
    df = sym.diff(f,x)
    ddf = sym.diff(df,x)
    
    # Changing function and its derivatives such that we can use them in NM
    f   = sym.lambdify(x,f,'numpy')
    df  = sym.lambdify(x, df, 'numpy')
    ddf = sym.lambdify(x, ddf, 'numpy')

    # Calculating bound for roots (Cauchy's bound)
    # 1 + max{|a_n-1/a_n|,...,|a_0/a_n|}
    # https://en.wikipedia.org/wiki/Geometrical_properties_of_polynomial_roots
    divided_by_an = list(map(lambda x: abs(x/coefficients[-1]), coefficients[:-1]))
    bound = 1 + max(divided_by_an)
    
    return f, df, ddf, bound

def calculate_delta_and_L(x0, f, df, ddf):
    delta = abs(f(x0)/df(x0)) #||N(x0)-x0||<=delta
    L = abs(1 - (df(x0)*df(x0)-f(x0)*ddf(x0))/df(x0)**2) #||DN(x0)||
    return delta, L

def contraction(L, delta):
    if L >= 1.0:
        return False
    if delta/(1-L) < 1e-10:
        return True
    return False

def newtonMethod(x0, iterationNumber, f, df):
    x=x0 # Start point
    # We iterate a certain number of times. 
    # If we do not find a satisfactory result in the process, 
    # we return "Can't estimate"
    for _ in range(iterationNumber):
        # Step in Newton's method
        x=x-f(x)/df(x)
        residual=np.abs(f(x))
        # If the value of the function at a given point is close 
        # enough to zero, we return this point as the root
        if residual < 1e-10:
            return x
    else:
        return "Can't estimate"

def startNewtonMethod(A):
    M = sym.Matrix(A)
    char_poly = M.charpoly()
    deg = sym.degree(char_poly)
    coeffs = char_poly.coeffs()
    coeffs = coeffs + list(np.zeros(deg + 1 - len(coeffs)))

    # Calculating the necessary data
    f, df, ddf, bound = prepare_data_for_NM(coeffs)

    # Create a table of start points in the calculated bounded area
    real_values = np.linspace(-bound, bound, 30)
    imaginary_values = np.linspace(-bound, bound, 30)

    # Finding (hopefully) all roots of given function
    roots = []
    for r in real_values:
        for im in imaginary_values:
            x0 = r + im*1j
            next_root = newtonMethod(x0, 500, f, df)

            # Check if the calculated root is unique or was already calculated
            duplicate = False
            if type(next_root) != str:
                for root in roots:
                    if abs(root - next_root) < 1e-8:
                        duplicate = True
                        break
                if not duplicate:
                    delta, L = calculate_delta_and_L(next_root, f, df, ddf)
                    if contraction(L, delta):
                        roots.append(next_root)
                        # Finish if all roots are already found
                        if len(roots) == deg:
                            eigenvalues = [np.round(root, 6) for root in roots]
                            if any([val.imag != 0 for val in eigenvalues]):
                                return eigenvalues, []
                            else:
                                eigenvectors = hf.solve_eigenvalue_equation(A, eigenvalues)
                            return eigenvalues, eigenvectors
                        break
                    
# -----------------------------------------------------------------------