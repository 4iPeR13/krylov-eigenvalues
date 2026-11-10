import numpy as np
from sympy import symbols, Poly, lambdify



A = np.array([
    [6.00, 1.10, 0.97, 1.24],
    [1.10, 4.00, 1.30, 0.16],
    [0.97, 1.30, 5.00, 2.10],
    [1.24, 0.16, 2.10, 7.00]
])

b = np.array([1.0, 0.0, 0.0, 0.0])
n = A.shape[0]

# Побудова матриці Крилова
B = np.zeros((n, n))
B[:, 0] = b
for i in range(1, n):
    B[:, i] = A @ B[:, i - 1]

# Система B·p = -A^n·b
rhs = -A @ B[:, -1]
p = np.linalg.solve(B, rhs) # Тут міг би бути шмат з коду лаби 2 але навіщо якщо є прекрасний solve від пітона) 
poly_coeffs = np.append(p, 1)[::-1]  


x = symbols('x')
poly = Poly(poly_coeffs, x)
f = lambdify(x, poly.as_expr(), 'numpy')
f_prime = lambdify(x, poly.diff().as_expr(), 'numpy')

def newton_root_search(f, f_prime, x0, eps=1e-8, max_iter=100):
    for _ in range(max_iter):
        fx = f(x0)
        fpx = f_prime(x0)
        if abs(fpx) < 1e-12:
            break
        x1 = x0 - fx / fpx
        if abs(x1 - x0) < eps:
            return x1
        x0 = x1
    return x0


manual_roots = []
for guess in np.linspace(1, 10, 100):
    root = newton_root_search(f, f_prime, guess)
    root_rounded = round(float(np.real_if_close(root)), 6)
    if all(abs(root_rounded - r) > 1e-4 for r in manual_roots):
        manual_roots.append(root_rounded)


numpy_roots = np.roots(poly_coeffs)

print("\nВектори Крилова (y1, y2, y3):")
for i in range(1, n):
    print(f"y{i} =")
    print(B[:, i])

print("\nМатриця системи Y (базис Крилова):")
print(B)

print("\nПрава частина системи: -A^n y0")
print(rhs)

print("\nХарактеристичне рівняння:")
print(poly.as_expr())

print("\nКорені:")
print(manual_roots)
print("\nПеревірка:")
print([round(r, 6) for r in numpy_roots])
