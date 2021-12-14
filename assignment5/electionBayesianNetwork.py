from pgmpy.models import BayesiaNetwork
from pgmpy.factors.discrete import TabularCPD

'''
The Bayesian Network Tool: https://github.com/pgmpy/pgmpy

Bayesian Network Design:



Variables: 

Root:
Gender (0 female; 1 male)
Vaccinated (0 no; 1 yes)
LiveInVirginia (0: no; 1 yes)
Jobless (0: jobless; 1 have a job)
IsParent (0: no; 1: yes)
Democartic (0: no; 1: yes)
Republican (0: no; 1: yes)


Intermediate:
ShouldEnforceCOVID19Vaccine (0: no; 1: yes) // P.S. more voters (46 percent) thought McAuliffe would handle vaccines better than Youngkin (41 percent)
ThinkAbortionShouldBeLegal (0 no; 1 yes)
ThinkParentShouldHaveMoreInfluenceOnSchoolCurricula (0: no; 1 yes)
EmphasizeVASafety (0 no; 1 yes)
ThinkVANeedsMoreJobs (0 no; 1 yes)


Output: 
Vote McAuliffe (0: vote YoungKin; 1: vote MCAuliffe)

'''
class BN:

    def __init__(self):
        self.model = None
    
    def defineModelStructure(self):
        self.model = BayesiaNetwork([('Gender', 'AbortionShouldBeLegal'), ('Gender', 'VASafety'), ('Vaccinated', 'ShouldEnforceCOVIDVaccine'), 
        ('LiveInVA', 'EmphasizeVASafety'), ('LiveInVA', 'ShouldEnforceCOVIDVaccine'), ('Jobless', 'ThinkVANeedMoreJobs'), ('Parent', 'ThinkParentShouldEnforceCurricula'),
        ('AbortionShouldBeLegal', 'VoteMc'), ('ShouldEnforceCOVIDVaccine', 'VoteMc'), ('ThinkParentShouldEnforceCurricula', 'VoteMc'), ('VaSafety', 'VoteMc'), ('ThinkVANeedMoreJobs', 'VoteMc'),
        ('Parent', 'VASafety'), ('Democrat', 'VoteMc'), ('Republican', 'VoteMc')])

    def defineCPD(self):
        gender = TabularCPD(variable = 'Gender', variable_card = 2, values = [[0.5], [0.5]])
        vaccinated = TabularCPD(variable = 'Vaccinated', variable_card = 2, values = [[0.2], [0.8]])
        inVA = TabularCPD(variable = 'LiveInVA', variable_card = 2, values = [[0.8], [0.2]])
        job = TabularCPD(variable = 'Jobless', variable_card = 2, values = [[0.3], [0.7]])
        parent = TabularCPD(variable = 'Parent', variable_card = 2, values = [[0.3], [0.7]])
        abortion = TabularCPD(variable = 'AbortionShouldBeLegal', variable_card = 2, values = [[], []], evidence = ['Gender'], evidence_card = [2])


    def estimate(self, varName):
