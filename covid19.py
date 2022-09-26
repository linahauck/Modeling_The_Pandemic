###   Problem 4   ###

from seir_interaction import *

def read_regions(filename):
    with open(filename) as infile:
        instances = []
        for line in infile:
            word = line.replace(" ", "")
            if "\t" in word:
                word = word.replace("\t","")
            word = word.replace("\n","")
            w = word.split(";")
            name = w[1]
            S_0 = float(w[2])
            E2_0 = float(w[3])
            lat = float(w[4])
            long = float(w[5])
            instances.append(RegionInteraction(name,S_0,E2_0,lat,long))
        return instances

def covid19_Norway(beta, filename, num_days, dt):
    instances = read_regions(filename)
    problem = ProblemInteraction(instances,"Norge",beta)
    solver = SolverSEIR(problem, num_days, dt)
    solver.solve()

    plt.figure(figsize=(10, 8)) # set figsize
    index = 1

    for i in instances:
        plt.subplot(4,3,index)
        i.plot()
        index += 1
    plt.subplot(4,3, index)
    plt.subplots_adjust(hspace = 0.75, wspace=0.75)
    problem.plot()
    plt.legend(loc='right', prop={"size": 6})
    plt.show()

covid19_Norway(beta=0.4, filename="fylker.txt", num_days =150, dt =1.0)



def beta2(t):
    r_e2=1.25; lmbda_2=0.5; r_ia=0.1 ; mu=0.2

    if 0 <= t < 29:
        R = 3.2
    elif 29 <= t < 65:
        R = 0.5
    elif 65 <= t < 137:
        R = 0.7
    elif 137 <= t < 168:
        R = 1.2
    elif 168 <= t < 229:
        R = 1.0
    elif 229 <= t < 264:
        R = 1.3
    else:
        R = 1.1

    return R/(r_e2/lmbda_2 + r_ia/mu + 1/mu)


days = []
b = []
for t in range(0,564+1):
    days.append(t)
    b.append(beta2(t))
np.array(days)
np.array(b)
plt.plot(days,b)
plt.xlabel("Time in days")
plt.ylabel("Beta")
plt.show()

covid19_Norway(beta=beta2, filename="fylker.txt", num_days =150, dt =1.0)


"""
b)
 approximate peak for the I category: 1.4e6
 In hospital - 20% of I at ap. peak: 0.28e6
 Need m.vent. - 5% of the 20%: 14000
 Det er 20 ganger fler enn de 700 vent. som er tilgjengelig.


Terminal> python3 covid19.py

"""
