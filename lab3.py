import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([
    ('S', 'O'),
    ('S', 'L'),
    ('L', 'M'),
    ('S', 'M')
])

#distributions
cpd_s = [[0.4], [0.6]]
cpd_o = [[0.1, 0.9],
          [0.7, 0.3]]
cpd_l = [[0.3, 0.7],
          [0.8, 0.2]]
cpd_m = [[0.2, 0.8, 0.5, 0.5],
          [0.8, 0.2, 0.5, 0.1]]

cpd_S = TabularCPD(variable='S', variable_card=2, values=cpd_s)
cpd_O = TabularCPD(variable='O', variable_card=2, values=cpd_o, evidence=['S'], evidence_card=[2])
cpd_L = TabularCPD(variable='L', variable_card=2, values=cpd_l, evidence=['S'], evidence_card=[2])
cpd_M = TabularCPD(variable='M', variable_card=2, values=cpd_m, evidence=['S', 'L'], evidence_card=[2, 2])

model.add_cpds(cpd_S, cpd_O, cpd_L, cpd_M)

inference = VariableElimination(model)

prob_O_given_S1 = inference.query(variables=['O'], evidence={'S': 1})
print("P(O | S=1):")
print(prob_O_given_S1)

prob_M_given_S1_L1 = inference.query(variables=['M'], evidence={'S': 1, 'L': 1})
print("\nP(M | S=1, L=1):")
print(prob_M_given_S1_L1)

prob_S_given_O1 = inference.query(variables=['S'], evidence={'O': 1})
print("\nP(S | O=1):")
print(prob_S_given_O1)