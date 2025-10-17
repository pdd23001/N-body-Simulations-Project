import numpy as np
import scipy 

def find_thetas(m, P_m, intervals=50):
    # Define the equation based on the "Tight Bounds on Quantum Searching" paper
    def equation(theta, m, P_m):
        # this equals 0 when theta is the correct value, and theta != 0 but is in the range [0, pi/2]
        # refer to "Tight Bounds on Quantum Searching" by Boyer et al., 1999
        return 1/2-P_m-(np.sin(4*m*theta)/(4*m*np.sin(2*theta)))
    
    # Initialize an empty list to store unique roots
    roots = []
    
    # Calculate the size of each interval
    interval_size = (np.pi / 2) / intervals
    
    # Loop over each interval and find roots
    for i in range(intervals):
        start = i * interval_size
        if i == 0:
            start = 1e-6
        end = (i + 1) * interval_size
        if i == intervals - 1:
            end = np.pi / 2 - 1e-6
        try:
            root = scipy.optimize.brentq(equation, start, end, args=(m, P_m))

            # Add root if it's not already in the list (within tolerance)
            if not any(np.isclose(root, r, atol=1e-6) for r in roots):

                # Make sure the root evaluated is close to 0
                if np.isclose(equation(root, m, P_m), 0, atol=1e-6):
                    roots.append(root)

            
        except ValueError as e:
            # No root found in this interval, skip it
            pass

    return roots