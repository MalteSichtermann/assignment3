import pickle

# Engineering 1.1
def bayes_function(poss_h, poss_d_given_h, poss_d_given_not_h):
    poss_d = poss_d_given_h * poss_h + poss_d_given_not_h * (1 - poss_h)
    return (poss_h * poss_d_given_h) / poss_d


print('\nEngineering 1.1')
print(bayes_function(0.1, 0.9, 0.3))
print(bayes_function(0.9, 0.9, 0.3))
print(bayes_function(0.9, 0.3, 0.9))
print(bayes_function(0.001, 0.99, 0.02))
print(bayes_function(0.3, 0.5, 0.5))


# Engineering 1.2
def bayes_function_multiple_hypotheses_gpt(priors, likelihoods):
    if len(priors) != len(likelihoods):
        raise ValueError("The length of priors and likelihoods must be the same")

    # Calculate the marginal likelihood of the data
    marginal_likelihood = sum(prior * likelihood for prior, likelihood in zip(priors, likelihoods))

    # Calculate posterior probabilities for each hypothesis
    posteriors = [(prior * likelihood) / marginal_likelihood for prior, likelihood in zip(priors, likelihoods)]

    return posteriors


print('\nEngineering 1.2')
print(bayes_function_multiple_hypotheses_gpt([0.4, 0.3, 0.3], [0.99, 0.9, 0.2]))
print(bayes_function_multiple_hypotheses_gpt([0.4, 0.3, 0.3], [0.9, 0.9, 0.2]))
print(bayes_function_multiple_hypotheses_gpt([0.3, 0.3, 0.4], [0.9, 0.9, 0.2]))
print(bayes_function_multiple_hypotheses_gpt([0.3, 0.3, 0.4], [0.9, 0.2, 0.2]))
print(bayes_function_multiple_hypotheses_gpt([0.4, 0.2, 0.2, 0.2], [0.9, 0.3, 0.3, 0.3]))
print(bayes_function_multiple_hypotheses_gpt([0.4, 0.2, 0.2, 0.2], [0.9, 0.6, 0.3, 0.3]))
print(bayes_function_multiple_hypotheses_gpt([0.01, 0.2, 0.2, 0.2], [0.99, 0.01, 0.01, 0.01]))


# Engineering 1.3
def bayesFactor(posteriors, priors):
    # Ensure input vectors are of the same length
    assert len(posteriors) == len(priors), "Length of posteriors and priors must be the same"

    # Calculate Bayes Factors for each hypothesis against all others
    BFs = []
    for i in range(len(posteriors)):
        for j in range(len(posteriors)):
            if i == j:
                BF = (posteriors[i] / (1 - posteriors[i])) / (priors[i] / (1 - priors[i]))
                BFs.append((i, f'-{j}', BF))
            else:
                BF = (posteriors[i] / (posteriors[j])) / (priors[i] / (priors[j]))
                BFs.append((i, j, BF))

    return BFs


print('\nEngineering 1.3')
print(bayesFactor([0.9, 0.05, 0.05], [0.2, 0.6, 0.2]))
print(bayesFactor([0.85, 0.05, 0.1], [0.2, 0.6, 0.2]))
print(bayesFactor([0.15, 0.35, 0.5], [0.4, 0.3, 0.3]))
print(bayesFactor([0.35, 0.15, 0.5], [0.3, 0.4, 0.3]))

print('\nBlackboard Task 1')
# Blackboard 1A
print('\na)')
posterior_initial = bayes_function(0.5,0.531, 0.52)
print(posterior_initial)

# Blackboard 1B
print('\nb)')
bayes_factor = bayesFactor([posterior_initial], [0.5])
print(bayes_factor)

# Blackboard 1C
print('\nc)')
posterior_second = bayes_function(0.001,0.531, 0.52)
print(posterior_second)

# Blackboard 1D
print('\nd)')
posterior1 = bayes_function(posterior_initial,0.471, 0.52)
print(posterior1)
print(bayesFactor([posterior1], [posterior_initial]))
posterior2 = bayes_function(posterior1,0.491, 0.65)
print(posterior2)
print(bayesFactor([posterior2], [posterior1]))
posterior3 = bayes_function(posterior2,0.505, 0.70)
print(posterior3)
print(bayesFactor([posterior3], [posterior2]))

with open('../fit_results/part_0_model_3.pkl', 'rb') as f:
    data = pickle.load(f)
    print('\nBlackboard Task 2')
    print(data)
