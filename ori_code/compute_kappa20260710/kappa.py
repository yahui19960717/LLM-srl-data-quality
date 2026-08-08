#!/usr/bin/env python3
"""Compute Cohen's kappa from a 2x2 LLM-human agreement matrix.

Matrix layout:
                 human=1   human=0
    LLM=1           a         b
    LLM=0           c         d

Cohen's kappa:
    Po = observed agreement = (a + d) / N
    Pe = expected agreement by chance
       = P(LLM=1) * P(human=1) + P(LLM=0) * P(human=0)
       = ((a+b)/N) * ((a+c)/N) + ((c+d)/N) * ((b+d)/N)
    kappa = (Po - Pe) / (1 - Pe)
"""


def cohen_kappa(a: int, b: int, c: int, d: int) -> None:
    n = a + b + c + d

    llm_1 = a + b
    llm_0 = c + d
    human_1 = a + c
    human_0 = b + d

    po = (a + d) / n
    pe = (llm_1 / n) * (human_1 / n) + (llm_0 / n) * (human_0 / n)
    kappa = (po - pe) / (1 - pe)

    print("Matrix:")
    print("                 human=1   human=0")
    print(f"LLM=1            {a:>7}   {b:>7}")
    print(f"LLM=0            {c:>7}   {d:>7}")
    print()
    print(f"N = {a} + {b} + {c} + {d} = {n}")
    print()
    print(f"Observed agreement Po = (a + d) / N = ({a} + {d}) / {n} = {po:.6f}")
    print()
    print("Expected agreement Pe = P(LLM=1)*P(human=1) + P(LLM=0)*P(human=0)")
    print(f"P(LLM=1) = (a + b) / N = ({a} + {b}) / {n} = {llm_1 / n:.6f}")
    print(f"P(LLM=0) = (c + d) / N = ({c} + {d}) / {n} = {llm_0 / n:.6f}")
    print(f"P(human=1) = (a + c) / N = ({a} + {c}) / {n} = {human_1 / n:.6f}")
    print(f"P(human=0) = (b + d) / N = ({b} + {d}) / {n} = {human_0 / n:.6f}")
    print(f"Pe = ({llm_1}/{n})*({human_1}/{n}) + ({llm_0}/{n})*({human_0}/{n}) = {pe:.6f}")
    print()
    print(f"Kappa = (Po - Pe) / (1 - Pe) = ({po:.6f} - {pe:.6f}) / (1 - {pe:.6f}) = {kappa:.6f}")
    print()
    print(f"Observed agreement = {po * 100:.2f}%")
    print(f"Expected agreement = {pe * 100:.2f}%")
    print(f"Cohen's kappa = {kappa:.3f}")


if __name__ == "__main__":
    # User-provided matrix:
    #                  human=1  human=0
    # LLM=1              58       2
    # LLM=0               4      53
    cohen_kappa(a=58, b=2, c=4, d=53)
