from borg import *
from lake import *

maxevals = 100

# need to start up MPI first
Configuration.startMPI()

# create an instance of the serial Borg MOEA
borg = Borg(nvars, nobjs, nconstrs, LakeProblemDPS)

# set the decision variable bounds and objective epsilons
borg.setBounds(*[[-2, 2], [0, 2], [0, 1]] * int((nvars / 3)))
borg.setEpsilons(0.01, 0.01, 0.0001, 0.0001)

# perform the optimization
# pass in a dictionary of arguments, as defined in borg.py
result = borg.solve({"maxEvaluations": maxevals})


# shut down MPI
Configuration.stopMPI()

# only the master node returns a result
# print the objectives to output
if result:
    print('success!')
    for solution in result:
        #print(solution.getObjectives())
        objectives = solution.getObjectives()
        objectives = np.column_stack(objectives)
        objectives_total = np.append(objectives_total,objectives,axis=0)
        strategies = solution.getVariables()
        
print(objectives_total)
