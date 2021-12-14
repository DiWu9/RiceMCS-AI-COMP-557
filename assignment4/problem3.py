import util, submission, sys

def solve_zebra_problem():

    csp = util.CSP()

    # varnames
    colors = ["red", "green", "blue", "yellow", "ivory"]
    nationalities = ["E", "S", "N", "U", "J"]
    candies = ["hershey", "smarties", "snickers", "milky", "kitkats"]
    drinks = ["juice", "tea", "coffee", "milk", "water"]
    pets = ["dog", "fox", "horse", "snail", "zebra"]
    vars = [colors, nationalities, candies, drinks, pets]

    # domain = [1,2,3,4,5] house number from left to right
    domain = list(range(1, 6))

    # add variables
    for varNames in vars:
        for varName in varNames:
            csp.add_variable(varName, domain)
        # same type of vars cannot have the same value
        for i in range(4):
            for j in range(i+1,5):
                csp.add_binary_potential(varNames[i], varNames[j], lambda x, y: x != y) 

    # add extra constraints
    csp.add_binary_potential("E", "red", lambda x, y: x == y) # English and Red in same house number
    csp.add_binary_potential("S", "dog", lambda x, y: x == y) # Spain and Dog same house number
    csp.add_unary_potential("N", lambda x: x == 1) # Norwegian in first house from left
    csp.add_binary_potential("ivory", "green", lambda x, y: y == x+1) # Ivory left to Green
    csp.add_binary_potential("hershey", "fox", lambda x, y: abs(x-y)==1) # hershey next to fox
    csp.add_binary_potential("kitkats", "yellow", lambda x, y: x == y) # kitkat and yellow in same house
    csp.add_binary_potential("N", "blue", lambda x, y: abs(x-y)==1) # Norwegian next to blue house
    csp.add_binary_potential("smarties", "snail", lambda x, y: x == y) # smarties eater owns snails
    csp.add_binary_potential("snickers", "juice", lambda x, y: x == y) # snickers eater drinks juice
    csp.add_binary_potential("U", "tea", lambda x, y: x == y) # Ukrainian drinks tea
    csp.add_binary_potential("J", "milky", lambda x, y: x == y) # Japanese eats milky
    csp.add_binary_potential("kitkats", "horse", lambda x, y: abs(x-y) == 1) # kitkat next to horse
    csp.add_binary_potential("coffee", "green", lambda x, y: x == y) # coffee in green
    csp.add_unary_potential("milk", lambda x: x == 3) # milk in middle

    zebraSolver = submission.BacktrackingSearch()
    zebraSolver.solve(csp, mcv = True, mac = True)
    if zebraSolver.optimalAssignment:
        for key, value in zebraSolver.optimalAssignment.items():
            print ("%s = %s" %(key, value))

solve_zebra_problem()