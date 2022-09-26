###   Problem 3   ###
from SEIR import *

class RegionInteraction(Region):
    def __init__(self, name,S_0,E2_0,lat,long):
        self.lat = lat * np.pi/180  #latitude
        self.long = long * np.pi/180 #longitude
        super().__init__(name,S_0,E2_0)

    def distance(self, other):
        arg = np.sin(self.lat)*np.sin(other.lat) + np.cos(self.lat)*np.cos(other.lat)*np.cos(abs(self.long - other.long))
        if arg > 1:
            arg = 1
        return 6400*np.arccos(arg)

class ProblemInteraction(ProblemSEIR):
    def __init__(self, region, area_name, beta, r_ia = 0.1, \
    r_e2=1.25, lmbda_1=0.33, lmbda_2=0.5, p_a=0.4,mu=0.2, k=0.01):
        super().__init__(region, beta, r_ia, r_e2, lmbda_1, lmbda_2, p_a, mu)
        self.area_name = area_name
        self.k = k

    def get_population(self):
        totpop = 0
        for r in self.region:
            totpop += r.population
        return totpop

    def set_initial_condition(self):
        ic_list  = []
        for r in self.region:
            ic_list += [r.S_0, r.E1_0, r.E2_0, r.I_0, r.Ia_0, r.R_0]
        self.initial_condition = ic_list

    def __call__(self, u, t):
        n = len(self.region)
        beta = self.beta(t)
        r_ia, r_e2 = self.r_ia, self.r_e2
        lmbda_1, lmbda_2 = self.lmbda_1, self.lmbda_2
        p_a, mu = self.p_a, self.mu
        # create a nested list:
        # SEIR_list[i] = [S_i, E1_i, E2_i, I_i, Ia_i, R_i]:
        SEIR_list = [u[i:i+6] for i in range(0, len(u), 6)]
        # Create separate lists containing E2 and Ia values:
        E2_list = [u[i] for i in range(2, len(u), 6)]
        Ia_list = [u[i] for i in range(4, len(u), 6)]
        derivative = []
        for i in range(n):
            S, E1, E2, I, Ia, R = SEIR_list[i]
            Ni = self.region[i].population
            dS = -beta*S*I/Ni
            for j in range(n):
                Nj = self.region[j].population
                d = self.region[i].distance(self.region[j])    #distance
                E2_other = E2_list[j]
                Ia_other = Ia_list[j]
                dS += - r_ia*beta*S*(Ia_other/Nj)*np.exp(-self.k*d) - r_e2*beta*S*(E2_other/Nj)*np.exp(-self.k*d)
            dE1 = -dS - lmbda_1*E1
            dE2 = lmbda_1*(1-p_a)*E1 - lmbda_2*E2
            dI  = lmbda_2*E2 - mu*I
            dIa = lmbda_1*p_a*E1 - mu*Ia
            dR  = mu*(I + Ia)
            derivative += [dS, dE1, dE2, dI, dIa, dR]
        return derivative

    def split_solution(self, u, t):
        n = len(t)
        n_reg = len(self.region)
        self.t = t
        self.S = np.zeros(n)
        self.E1 = np.zeros(n)
        self.E2 = np.zeros(n)
        self.I = np.zeros(n)
        self.Ia = np.zeros(n)
        self.R = np.zeros(n)
        SEIR_list = [u[:, i:i+6] for i in range(0, n_reg*6, 6)]
        for part, SEIR in zip(self.region, SEIR_list):
            part.set_SEIR_values(SEIR, t)
            self.S += part.S
            self.E1 += part.E1
            self.E2 += part.E2
            self.I += part.I
            self.Ia += part.Ia
            self.R += part.R

    def plot(self):
        plt.plot(self.t,self.S,label="S",color="b")
        plt.plot(self.t,self.I,label="I",color="m")
        plt.plot(self.t,self.Ia,label="Ia",color="g")
        plt.plot(self.t,self.R,label="R",color="r")
        plt.xlabel("Time(days)")
        plt.ylabel("Total Population")
        plt.title(self.area_name)




if __name__ == '__main__':
    innlandet = RegionInteraction('Innlandet',S_0=371385, E2_0=0, \
                         lat=60.7945,long=11.0680)
    oslo = RegionInteraction('Oslo',S_0=693494,E2_0=100, \
                         lat=59.9,long=10.8)

    print(oslo.distance(innlandet))

    problem = ProblemInteraction([oslo,innlandet],'Norway_east', beta=0.4)
    print(problem.get_population())
    problem.set_initial_condition()
    print(problem.initial_condition) #non-nested list of length 12
    u = problem.initial_condition
    print(problem(u,0)) #list of length 12. Check that values make sense

    #when lines above work, add this code to solve a test problem:
    solver = SolverSEIR(problem,T=150,dt=1.0)
    solver.solve()
    problem.plot()
    plt.legend()
    plt.show()

    """
    Terminal> python3 seir_interaction.py
101.00809386285283
1064979
[693494, 0, 100, 0, 0, 0, 371385, 0, 0, 0, 0, 0]
[-49.992723746591004, 49.992723746591004, -50.0, 50.0, 0.0, 0.0, -9.75026585941784, 9.75026585941784, 0.0, 0.0, 0.0, 0.0]
"""
