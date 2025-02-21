def C(n, k):
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    numerator = 1
    denominator = 1
    for i in range(k):
        numerator *= n - i
        denominator *= i+1
    return numerator//denominator

def binomial(N, m):
    """
    N: numbre of results
    m: numbre of H
    C(N,m)*(m/N)^m*(1-m/N)^(N-m)
    """
    return C(N, m) * ((m / N) ** m) * ((1 - (m / N)) ** (N - m))

if __name__ == '__main__':
    cases = 1
    prior_a = int(input('Initial beta prior a = '))
    prior_b = int(input('Initial beta prior b = '))
    with open('ML_HW/HW2/testfile.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            print(f'case {cases}: {line}')

            # Compute Binomial likelihood & Beta posterior
            # ========================================================
            outcomes = []
            for trial in line:
                outcomes.append(int(trial))

            num_trial = len(outcomes)
            H = sum(outcomes)
            T = num_trial - H

            likelihood = binomial(num_trial, H)
            posterior_a = prior_a + H
            posterior_b = prior_b + T
            
            # Print resule
            # ========================================================
            print(f'Likelihood: {likelihood}')
            print(f'Beta prior:     a = {prior_a} b = {prior_b}')
            print(f'Beta posterior: a = {posterior_a} b = {posterior_b}\n')
            
            prior_a = posterior_a
            prior_b = posterior_b
            cases += 1

# python ML_HW/HW2/online_learning.py