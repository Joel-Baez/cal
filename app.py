import re
import random
from functools import reduce
import operator
from string import ascii_uppercase
from typing import Dict, List

from flask import Flask, jsonify, request, send_from_directory
from sympy import (
    E,
    Integral,
    Symbol,
    Pow,
    Wild,
    acos,
    apart,
    asin,
    atan,
    cos,
    cosh,
    cot,
    Eq,
    Rational,
    csc,
    diff,
    exp,
    expand,
    factor,
    fraction,
    integrate,
    latex,
    log,
    pi,
    sec,
    simplify,
    sin,
    sinh,
    Poly,
    solve,
    sqrt,
    symbols,
    tan,
    tanh,
    Integer,
)
from sympy.core.function import AppliedUndef
from sympy.core.sympify import SympifyError
from sympy.parsing.sympy_parser import (
    TokenError,
    function_exponentiation,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

app = Flask(__name__, static_url_path='', static_folder='.')

TRANSFORMATIONS = (
    standard_transformations
    + (implicit_multiplication_application, function_exponentiation)
)

ALLOWED_FUNCTIONS = {
    'sin': sin,
    'cos': cos,
    'tan': tan,
    'asin': asin,
    'acos': acos,
    'atan': atan,
    'sinh': sinh,
    'cosh': cosh,
    'tanh': tanh,
    'cot': cot,
    'sec': sec,
    'csc': csc,
    'exp': exp,
    'log': log,
    'ln': log,
    'sqrt': sqrt,
    'E': E,   # Euler (mayúscula)
    'e': E,   # Soporte de 'e' minúscula en entrada
    'pi': pi,
}

METHOD_DETAILS: Dict[str, Dict[str, object]] = {
    'substitution': {
        'title': 'Sustitución simple',
        'badge': 'u-substitución',
        'summary': (
            r'Una función compuesta $f(g(x))$ cuya derivada $g\'(x)$ aparece multiplicando '
            r'permite introducir $u = g(x)$ para integrar en una sola variable auxiliar.'
        ),
    },
    'parts': {
        'title': 'Integración por partes',
        'badge': '$u$ · $dv$',
        'summary': (
            r'Cuando el integrando es un producto, conviene derivar la parte que se simplifica '
            r'y antiderivar la que mantiene una forma manejable.'
        ),
    },
    'trig': {
        'title': 'Sustitución trigonométrica',
        'badge': r'$\theta$-sustitución',
        'summary': (
            r'Las raíces de la forma $\sqrt{a^2 - x^2}$, $\sqrt{a^2 + x^2}$ o '
            r'$\sqrt{x^2 - a^2}$ sugieren introducir un ángulo $\theta$ para aprovechar identidades trigonométricas.'
        ),
    },
    'partial_fractions': {
        'title': 'Fracciones parciales',
        'badge': 'descomposición',
        'summary': (
            r'Un cociente de polinomios factorizable se puede expresar como suma de fracciones '
            r'más simples cuya integración es directa.'
        ),
    },
    'repeated_factors': {
        'title': 'Fracciones parciales con factores repetidos',
        'badge': 'potencias lineales',
        'summary': (
            r'Cuando el denominador tiene factores lineales elevados a una potencia, cada potencia requiere '
            r'un término separado en la descomposición para integrar sin complicaciones.'
        ),
    },
    'default': {
        'title': 'Exploración general',
        'badge': 'observación',
        'summary': (
            r'No se detectó un patrón dominante. Simplifica el integrando, separa en sumas '
            r'o intenta sustituciones básicas para avanzar.'
        ),
    },
}

# -----------------------------
# Utilidades de “ejemplo parecido”
# -----------------------------
def _perturb_int(n: int) -> int:
    delta = random.choice([-2, -1, 1, 2])
    m = n + delta
    return m if m != 0 else (m + 1)

def perturb_constants(expr):
    """
    Devuelve una expresión 'parecida' con constantes enteras perturbadas ±1/±2.
    No toca la variable de integración ni símbolos arbitrarios.
    """
    return expr.replace(lambda e: isinstance(e, Integer), lambda e: Integer(_perturb_int(int(e))))
# -----------------------------


def clean_latex(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    cleaned = text
    replacements = [
        (r'\\left', ''),
        (r'\\right', ''),
    ]
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r'\\operatorname\{([A-Za-z]+)\}', lambda match: f"\\{match.group(1)}", cleaned)
    cleaned = re.sub(r'\\log(?!_)', r'\\ln', cleaned)
    cleaned = cleaned.replace(r'\mathrm{d}', 'd')
    return cleaned


def latexize(obj) -> str:
    if isinstance(obj, str):
        return clean_latex(obj)
    try:
        rendered = latex(obj)
    except Exception:
        rendered = str(obj)
    return clean_latex(rendered)


def make_integral_latex(expr, variable: Symbol) -> str:
    return rf"\int {latexize(expr)}\\,d{latexize(variable)}"


def format_antiderivative(expr, variable: Symbol) -> str:
    constant_suffix = ' + c'
    if isinstance(expr, Integral):
        return latexize(expr) + constant_suffix
    return f"{latexize(expr)}{constant_suffix}"


def linearize_argument(arg, var: Symbol):
    poly = arg.as_poly(var)
    if poly is None:
        return var
    degree = poly.degree()
    if degree <= 1:
        return simplify(poly.as_expr())
    leading = poly.LC()
    constant = poly.eval(0)
    return simplify(leading * var + constant)


def simplify_parts_component(expr, var: Symbol):
    if expr.is_Function:
        func = expr.func
        if func in (sin, cos, tan, cot, sec, csc, sinh, cosh, tanh):
            arg = expr.args[0] if expr.args else var
            return func(linearize_argument(arg, var))
        if func is exp:
            arg = expr.args[0] if expr.args else var
            return exp(linearize_argument(arg, var))
        if func is log:
            arg = expr.args[0] if expr.args else var
            candidate = linearize_argument(arg, var)
            if simplify(candidate) == 0:
                candidate = var + 1
            return log(candidate)
    if expr.is_Pow:
        base = simplify_parts_component(expr.base, var)
        exponent = expr.exp
        if exponent.has(var):
            evaluated = simplify(exponent.subs(var, 1))
            exponent = 2 if evaluated.free_symbols else evaluated
        return base ** exponent
    if expr.is_Mul:
        result = 1
        for factor in expr.args:
            result *= simplify_parts_component(factor, var)
        return simplify(result)
    if expr.is_Add:
        return simplify(sum(simplify_parts_component(term, var) for term in expr.args))
    return expr


def format_differential(name: str, expr, variable: Symbol) -> str:
    simplified = simplify(expr)
    return rf"{name} = {latexize(simplified)}\\,d{latexize(variable)}"


def format_dx_from_du(variable: Symbol, derivative) -> str:
    simplified = simplify(derivative)
    if simplify(simplified - 1) == 0:
        return rf"d{latexize(variable)} = du"
    return rf"d{latexize(variable)} = \\frac{{du}}{{{latexize(simplified)}}}"


def build_step(title: str, description: str, equations: List[str] | None = None) -> Dict[str, object]:
    payload: Dict[str, object] = {'title': title, 'description': description}
    if equations:
        payload['equations'] = equations
    return payload


def describe_trig_substitution(inner_expr, var: Symbol):
    inner = simplify(inner_expr)
    a = Wild('a', exclude=[var])
    k = Wild('k', exclude=[var])

    patterns = [
        (a**2 - (k * var) ** 2, 'sqrt(a^2 - (bx)^2)'),
        ((k * var) ** 2 - a**2, 'sqrt((bx)^2 - a^2)'),
        (a**2 + (k * var) ** 2, 'sqrt(a^2 + (bx)^2)'),
        ((k * var) ** 2 + a**2, 'sqrt(a^2 + (bx)^2)'),
    ]

    for pattern_expr, pattern_label in patterns:
        match = inner.match(pattern_expr)
        if match and match.get(a) not in (None, 0) and match.get(k) not in (None, 0):
            a_val = simplify(abs(match[a]))
            b_val = simplify(abs(match[k]))
            return {'pattern': pattern_label, 'a': a_val, 'b': b_val}
    return None


def generate_substitution_example(expr, var: Symbol):
    # Caso típico racional: (g'(x))/g(x) → log
    if expr.is_rational_function(var) and not expr.is_polynomial(var):
        _, denominator = fraction(expr)
        inner = simplify(denominator)
        derived = simplify(diff(inner, var))
        if derived != 0:
            shifted_inner = simplify(inner + 1)  # parecido, no igual
            example_integrand = simplify(derived / shifted_inner)
            antiderivative = integrate(example_integrand, var)
            u_symbol = Symbol('u')
            reduced_integrand = 1 / u_symbol
            reduced_antiderivative = integrate(reduced_integrand, u_symbol)
            du_value = format_differential('du', derived, var)
            dx_value = format_dx_from_du(var, derived)
            inner_tex = latexize(shifted_inner)
            derived_tex = latexize(derived)
            reduced_integrand_tex = latexize(reduced_integrand)
            setup = [
                {'label': 'u(x)', 'value': rf"u(x) = {inner_tex}"},
                {'label': 'du', 'value': du_value},
                {'label': 'dx', 'value': dx_value},
                {'label': 'Integral en u', 'value': rf"\\int {reduced_integrand_tex}\\,du"},
            ]
            steps = [
                build_step(
                    '1) Observamos el denominador',
                    (
                        'El denominador compuesto '
                        f"${inner_tex}$ aparece junto con su derivada ${derived_tex}$, "
                        'lo que sugiere una sustitución directa.'
                    ),
                    [make_integral_latex(example_integrand, var)],
                ),
                build_step(
                    '2) Declaramos $u(x)$ y el diferencial',
                    'Expresamos el cambio de variable y cómo se transforma $dx$.',
                    [rf"u(x) = {inner_tex}", du_value, dx_value],
                ),
                build_step(
                    '3) Integramos en términos de $u$',
                    'La integral resultante es elemental y conduce a un logaritmo.',
                    [rf"\\int {reduced_integrand_tex}\\,du = {format_antiderivative(reduced_antiderivative, u_symbol)}"],
                ),
                build_step(
                    '4) Sustituimos de vuelta',
                    'Reemplazamos $u$ por la expresión original para obtener la antiderivada.',
                    [format_antiderivative(antiderivative, var)],
                ),
            ]
            return {
                'example_integral': make_integral_latex(example_integrand, var),
                'example_solution': format_antiderivative(antiderivative, var),
                'setup': setup,
                'steps': steps,
            }

    # Fallback genérico con ligera perturbación del interior
    inner = var**3 + var
    outer = sin
    derived = diff(inner, var)
    shifted_inner = simplify(perturb_constants(inner) + 1)
    example_integrand = simplify(derived * outer(shifted_inner))
    antiderivative = integrate(example_integrand, var)
    u_symbol = Symbol('u')
    du_value = format_differential('du', derived, var)
    dx_value = format_dx_from_du(var, derived)
    reduced_integrand = outer(u_symbol)
    reduced_integral = integrate(reduced_integrand, u_symbol)
    inner_tex = latexize(shifted_inner)
    derived_tex = latexize(derived)
    reduced_integrand_tex = latexize(reduced_integrand)
    reduced_integral_tex = format_antiderivative(reduced_integral, u_symbol)
    setup = [
        {'label': 'u(x)', 'value': rf"u(x) = {inner_tex}"},
        {'label': 'du', 'value': du_value},
        {'label': 'dx', 'value': dx_value},
        {'label': 'Integral en u', 'value': rf"\\int {reduced_integrand_tex}\\,du"},
    ]
    steps = [
        build_step(
            '1) Identificamos la función compuesta',
            (
                'Notamos que la parte interior '
                f"${inner_tex}$ aparece junto con su derivada ${derived_tex}$ "
                'multiplicando a la función exterior.'
            ),
            [make_integral_latex(example_integrand, var)],
        ),
        build_step(
            '2) Declaramos $u(x)$ y el diferencial',
            'Elegimos una variable auxiliar que simplifique la composición.',
            [rf"u(x) = {inner_tex}", du_value, dx_value],
        ),
        build_step(
            '3) Integramos en la variable auxiliar',
            'Al sustituir, obtenemos una integral elemental en $u$.',
            [rf"\\int {reduced_integrand_tex}\\,du = {reduced_integral_tex}"],
        ),
        build_step(
            '4) Regresamos a la variable original',
            'Reemplazamos $u$ por la expresión inicial para concluir la antiderivada.',
            [format_antiderivative(antiderivative, var)],
        ),
    ]
    return {
        'example_integral': make_integral_latex(example_integrand, var),
        'example_solution': format_antiderivative(antiderivative, var),
        'setup': setup,
        'steps': steps,
    }


def split_product(expr, var: Symbol):
    factors = list(expr.as_ordered_factors()) if expr.is_Mul else [expr]
    polynomial = None
    other = None
    for factor_candidate in factors:
        poly = factor_candidate.as_poly(var)
        if poly is not None:
            polynomial = factor_candidate
            break
    if polynomial is None:
        polynomial = var
        other = expr / var
    else:
        remaining = simplify(expr / polynomial)
        other = remaining
    return polynomial, other


def generate_parts_example(expr, var: Symbol):
    """
    Genera un ejemplo por partes muy parecido:
    - Si el integrando es P(x)*exp(ax) o P(x)*{sin,cos}(bx), conserva la función especial y
      perturba levemente coeficientes de P(x).
    """
    poly, other = split_product(expr, var)

    # Mantener la parte trascendental "igual" (linealizada) y perturbar el polinomio
    u_expr = simplify(perturb_constants(poly) + 1)  # similar a P(x)
    dv_expr = simplify(other)
    dv_simple = simplify_parts_component(dv_expr, var)

    example_integrand = simplify(u_expr * dv_simple)
    du_expr = simplify(diff(u_expr, var))
    try:
        v_expr = integrate(dv_simple, var)
    except Exception:
        v_expr = Integral(dv_simple, var)
    try:
        reduction = integrate(simplify(v_expr * du_expr), var)
    except Exception:
        reduction = Integral(simplify(v_expr * du_expr), var)
    antiderivative = simplify(u_expr * v_expr - reduction)

    u_tex = latexize(u_expr)
    du_tex = latexize(du_expr)
    v_tex = latexize(v_expr)
    var_tex = latexize(var)
    product_integral_tex = latexize(example_integrand)

    setup = [
        {'label': 'u(x)', 'value': rf"u(x) = {u_tex}"},
        {'label': 'du', 'value': format_differential('du', du_expr, var)},
        {'label': 'dv', 'value': format_differential('dv', dv_simple, var)},
        {'label': 'v(x)', 'value': rf"v(x) = {v_tex}"},
    ]
    steps = [
        build_step(
            '1) Elegimos integración por partes',
            'Reorganizamos el producto para derivar la parte algebraica y antiderivar la parte especial.',
            [make_integral_latex(example_integrand, var)],
        ),
        build_step(
            '2) Fijamos las asignaciones',
            'Asignamos $u$ y $dv$ para que la derivada de $u$ reduzca el grado del polinomio.',
            [rf"u(x) = {u_tex}", format_differential('du', du_expr, var),
             format_differential('dv', dv_simple, var), rf"v(x) = {v_tex}"],
        ),
        build_step(
            '3) Aplicamos la fórmula',
            r'Utilizamos $\int u\,dv = uv - \int v\,du$ y simplificamos la integral restante.',
            [r"\int u\,dv = uv - \int v\,du",
             rf"\int {product_integral_tex}\\,d{var_tex} = {u_tex}{v_tex} - \int {v_tex}\\,{du_tex}\\,d{var_tex}"],
        ),
        build_step(
            '4) Presentamos la primitiva final',
            'Sumamos el resultado y añadimos la constante de integración.',
            [format_antiderivative(antiderivative, var)],
        ),
    ]
    return {
        'example_integral': make_integral_latex(example_integrand, var),
        'example_solution': format_antiderivative(antiderivative, var),
        'setup': setup,
        'steps': steps,
    }


def generate_trig_example(expr, var: Symbol):
    radicands = [term.args[0] for term in expr.atoms(sqrt) if term.has(var)]
    inner = radicands[0] if radicands else var**2 + 1
    description = describe_trig_substitution(inner, var)
    a = description['a'] if description else 1
    b = description['b'] if description else 1
    pattern = description['pattern'] if description else 'sqrt(a^2 + (bx)^2)'
    theta = Symbol('theta')

    # Perturbar levemente a y/o b para no copiar el caso exacto del usuario
    try:
        if hasattr(a, 'is_integer') and a.is_integer:
            a = Integer(_perturb_int(int(a)))
        if hasattr(b, 'is_integer') and b.is_integer:
            b = Integer(_perturb_int(int(b)))
    except Exception:
        pass

    if pattern == 'sqrt(a^2 - (bx)^2)':
        example_integrand = 1 / sqrt(a**2 - (b * var) ** 2)
        substitution_expr = (a / b) * sin(theta)
        inverse = latexize(asin(var * b / a))
    elif pattern == 'sqrt((bx)^2 - a^2)':
        example_integrand = sqrt((b * var) ** 2 - a**2) / var
        substitution_expr = (a / b) * sec(theta)
        inverse = latexize(acos(a / (b * var)))
    else:
        example_integrand = 1 / sqrt(a**2 + (b * var) ** 2)
        substitution_expr = (a / b) * tan(theta)
        inverse = latexize(atan(var * b / a))

    dx_theta = diff(substitution_expr, theta)
    var_tex = latexize(var)
    theta_tex = latexize(theta)
    substitution = rf"{var_tex} = {latexize(substitution_expr)}"
    differential = rf"d{var_tex} = {latexize(dx_theta)}\\,d{theta_tex}"
    integrand_theta = simplify(example_integrand.subs(var, substitution_expr) * dx_theta)
    theta_integral = latexize(integrand_theta)
    theta_antiderivative = latexize(integrate(integrand_theta, theta))
    antiderivative = integrate(example_integrand, var)

    setup = [
        {'label': 'Sustitución', 'value': substitution},
        {'label': 'Diferencial', 'value': differential},
        {'label': 'Inversa', 'value': inverse},
    ]
    steps = [
        build_step(
            '1) Reconocemos el patrón cuadrático',
            'La raíz identifica el uso de una identidad trigonométrica para eliminar la raíz.',
            [make_integral_latex(example_integrand, var)],
        ),
        build_step(
            '2) Realizamos la sustitución angular',
            r'Expresamos $x$ y $dx$ con $\\theta$ para simplificar la raíz.',
            [substitution, differential],
        ),
        build_step(
            r'3) Integramos en $\\theta$',
            'Resolvemos la integral elemental resultante y simplificamos.',
            [rf"\int {theta_integral}\\,d\\theta = {theta_antiderivative}"],
        ),
        build_step(
            '4) Retornamos a la variable $x$',
            'Aplicamos la sustitución inversa para expresar el resultado final en términos de la variable original.',
            [format_antiderivative(antiderivative, var)],
        ),
    ]
    return {
        'example_integral': make_integral_latex(example_integrand, var),
        'example_solution': format_antiderivative(antiderivative, var),
        'setup': setup,
        'steps': steps,
    }


def generate_partial_fractions_example(expr, var: Symbol):
    _, denominator = fraction(expr)
    denominator = factor(simplify(denominator))

    poly = denominator.as_poly(var)
    factors_data: List[tuple] = []

    if poly is not None:
        for factor_expr, multiplicity in poly.factor_list()[1]:
            factor_poly = factor_expr.as_poly(var)
            if factor_poly is None:
                continue
            deg = factor_poly.degree()
            factors_data.append((factor_expr, multiplicity, deg))

    if not factors_data:
        factors_data = [
            (var - 2, 1, 1),
            (var + 1, 1, 1),
            (var**2 + 4, 1, 2),
        ]

    # Denominador muy parecido (coeficientes perturbados)
    perturbed_factors = []
    for f_expr, mult, deg in factors_data:
        if deg == 1:
            # (x - a) -> (x - (a±))
            coeffs = f_expr.as_poly(var).all_coeffs()  # [1, -a]
            a = -coeffs[1]
            try:
                a = Integer(_perturb_int(int(a)))
            except Exception:
                pass
            perturbed = (var - a)
            perturbed_factors.extend([perturbed] * mult)
        elif deg == 2:
            # x^2 + p x + q -> variar p o q
            coeffs = f_expr.as_poly(var).all_coeffs()  # [1, p, q]
            p, q = coeffs[1], coeffs[2]
            try:
                p = Integer(_perturb_int(int(p)))
            except Exception:
                pass
            try:
                q = Integer(_perturb_int(int(q)))
            except Exception:
                pass
            perturbed = var**2 + p*var + q
            perturbed_factors.extend([perturbed] * mult)
        else:
            perturbed_factors.extend([f_expr] * mult)

    denominator_example = factor(reduce(operator.mul, perturbed_factors, 1))

    # Forma general de descomposición
    coeff_iter = (Symbol(name) for name in ascii_uppercase if name.lower() != str(var).lower())
    def next_symbol(index: int) -> Symbol:
        try:
            return next(coeff_iter)
        except StopIteration:
            return Symbol(f'c_{index}')

    decomposition_terms = []
    used_symbols: List[Symbol] = []

    for factor_expr in perturbed_factors:
        deg = factor_expr.as_poly(var).degree()
        if deg == 1:
            A = next_symbol(len(used_symbols))
            decomposition_terms.append(A / factor_expr)
            used_symbols.append(A)
        else:
            B = next_symbol(len(used_symbols))
            C = next_symbol(len(used_symbols) + 1)
            decomposition_terms.append((B*var + C) / factor_expr)
            used_symbols.extend([B, C])

    if not decomposition_terms:
        A = Symbol('A')
        decomposition_terms.append(A / (var - 1))
        used_symbols.append(A)

    decomposition_general = sum(decomposition_terms)

    # Elegimos valores concretos "bonitos" para el ejemplo
    value_pool = [Rational(3, 2), -1, Rational(5, 3), 2, Rational(-4, 3), 1]
    chosen_values = {sym: value_pool[i % len(value_pool)] for i, sym in enumerate(used_symbols)}

    example_integrand = simplify(decomposition_general.subs(chosen_values))
    decomposition_specific = apart(example_integrand, var, full=True)
    antiderivative = integrate(example_integrand, var)

    # Igualación de numeradores
    lhs = expand(decomposition_general * denominator_example)
    rhs = expand(example_integrand * denominator_example)
    lhs_poly = Poly(lhs, var)
    rhs_poly = Poly(rhs, var)
    lhs_coeffs = lhs_poly.all_coeffs()
    rhs_coeffs = rhs_poly.all_coeffs()
    pad_length = max(len(lhs_coeffs), len(rhs_coeffs))

    def pad(coeffs, length):
        return [0] * (length - len(coeffs)) + coeffs

    lhs_coeffs = pad(lhs_coeffs, pad_length)
    rhs_coeffs = pad(rhs_coeffs, pad_length)

    equations = [
        latexize(Eq(simplify(lhs_coeffs[i]), simplify(rhs_coeffs[i]))) for i in range(pad_length)
    ]
    system_latex = r'\begin{cases}' + r'\\'.join(equations) + r'\end{cases}'

    solutions = solve(
        [Eq(simplify(lhs_coeffs[i]), simplify(rhs_coeffs[i])) for i in range(pad_length)],
        tuple(used_symbols),
        dict=True,
    )
    solution = solutions[0] if solutions else chosen_values
    assignments = [
        latexize(Eq(sym, simplify(solution.get(sym, chosen_values[sym])))) for sym in used_symbols
    ]

    coeff_summary = r',\; '.join(
        f"{latexize(sym)} = {latexize(simplify(solution.get(sym, chosen_values[sym])))}"
        for sym in used_symbols
    )

    setup = [
        {'label': 'Denominador factorizado (ejemplo)', 'value': latexize(denominator_example)},
        {'label': 'Forma general', 'value': latexize(decomposition_general)},
        {'label': 'Coeficientes hallados', 'value': coeff_summary},
    ]
    steps = [
        build_step(
            '1) Factorizamos el denominador',
            'Expresamos el denominador como producto de factores lineales y cuadráticos irreductibles.',
            [latexize(denominator_example)],
        ),
        build_step(
            '2) Proponemos la descomposición',
            'Asignamos constantes a cada fracción elemental según su tipo.',
            [latexize(decomposition_general)],
        ),
        build_step(
            '3) Igualamos numeradores y resolvemos',
            'Primero escribimos la identidad de numeradores y luego igualamos coeficientes para hallar A, B, C, D, Ax+B, Cx+D.',
            [
                latexize(Eq(lhs, rhs)),   # ← identidad completa de numeradores (nuevo)
                system_latex,             # ← sistema por coeficientes
                *assignments,             # ← valores obtenidos
            ],
        ),
        build_step(
            '4) Integramos término a término',
            'Sustituimos los coeficientes hallados y sumamos las primitivas de cada término.',
            [latexize(decomposition_specific), format_antiderivative(antiderivative, var)],
        ),
    ]
    return {
        'example_integral': make_integral_latex(example_integrand, var),
        'example_solution': format_antiderivative(antiderivative, var),
        'setup': setup,
        'steps': steps,
    }


def generate_repeated_factors_example(expr, var: Symbol):
    _, denominator = fraction(expr)
    factors = factor(denominator)
    factor_terms = factors.as_ordered_factors() if factors != 0 else []
    dominant = factor_terms[0] if factor_terms else (var - 1) ** 2
    # (x-a)^m -> (x-(a±))^m
    if dominant.is_Pow and dominant.base.as_poly(var) is not None:
        base = dominant.base.as_poly(var)
        a = -base.all_coeffs()[1]
        try:
            a = Integer(_perturb_int(int(a)))
        except Exception:
            pass
        dominant = (var - a) ** int(dominant.exp)

    example_integrand = 1 / dominant
    antiderivative = integrate(example_integrand, var)
    decomposition = apart(example_integrand, var, full=True)
    setup = [
        {'label': 'Factor repetido', 'value': latexize(dominant)},
        {'label': 'Descomposición', 'value': latexize(decomposition)},
    ]
    steps = [
        build_step(
            '1) Aislamos el factor repetido',
            'Identificamos el factor dominante y preparamos términos para cada potencia.',
            [latexize(dominant)],
        ),
        build_step(
            '2) Asignamos fracciones parciales escalonadas',
            'Cada potencia genera una fracción con numeradores constantes a determinar.',
            [latexize(decomposition)],
        ),
        build_step(
            '3) Integramos sumando cada contribución',
            'Aparecen potencias y logaritmos según la potencia del factor.',
            [format_antiderivative(antiderivative, var)],
        ),
    ]
    return {
        'example_integral': make_integral_latex(example_integrand, var),
        'example_solution': format_antiderivative(antiderivative, var),
        'setup': setup,
        'steps': steps,
    }


def generate_default_example(var: Symbol):
    example_integrand = var**2 + 2 * var + 3
    antiderivative = integrate(example_integrand, var)
    steps = [
        build_step(
            '1) Separar en sumas manejables',
            'Dividimos la integral en términos independientes.',
            [latexize(example_integrand)],
        ),
        build_step(
            '2) Aplicar reglas básicas',
            r'Integramos cada potencia usando la regla $\int x^{n}\,dx = x^{n+1}/(n+1)$.',
            [format_antiderivative(antiderivative, var)],
        ),
    ]
    return {
        'example_integral': make_integral_latex(example_integrand, var),
        'example_solution': format_antiderivative(antiderivative, var),
        'setup': [],
        'steps': steps,
    }


EXAMPLE_GENERATORS = {
    'substitution': generate_substitution_example,
    'parts': generate_parts_example,
    'trig': generate_trig_example,
    'partial_fractions': generate_partial_fractions_example,
    'repeated_factors': generate_repeated_factors_example,
}

def sanitize_expression(expression: str, variable: str) -> str:
    expr = expression or ''
    # Quitar símbolos de integral y diferenciales
    expr = expr.replace('∫', '')
    expr = re.sub(rf'd{re.escape(variable)}\b', '', expr, flags=re.IGNORECASE)
    expr = re.sub(r'd[a-zA-Z]\b', '', expr)

    # Normalizaciones básicas
    replacements = [
        ('^', '**', False),
        ('√', 'sqrt', False),
        ('π', 'pi', False),
        ('\\pi', 'pi', False),
        ('\\sqrt', 'sqrt', False),
        ('\\sin', 'sin', False),
        ('\\cos', 'cos', False),
        ('\\tan', 'tan', False),
        ('\\sec', 'sec', False),
        ('\\csc', 'csc', False),
        ('\\cot', 'cot', False),
        ('\\sinh', 'sinh', False),
        ('\\cosh', 'cosh', False),
        ('\\tanh', 'tanh', False),
        ('\\arcsin', 'asin', False),
        ('\\arccos', 'acos', False),
        ('\\arctan', 'atan', False),
        ('\\exp', 'exp', False),
        ('\\ln', 'log', False),
        ('\\log', 'log', False),
        ('sen', 'sin', True),
        ('tg', 'tan', True),
        ('ctg', 'cot', True),
        ('ln', 'log', True),
    ]
    for pattern, replacement, is_alpha in replacements:
        if is_alpha:
            expr = re.sub(pattern, replacement, expr, flags=re.IGNORECASE)
        else:
            expr = expr.replace(pattern, replacement)

    # Quitar restos de LaTeX y espacios
    expr = re.sub(r'\\int', '', expr, flags=re.IGNORECASE)
    expr = expr.replace('\\cdot', '*').replace('\\times', '*')
    expr = expr.replace('\\,', '').replace('\\!', '')
    expr = re.sub(r'\\left|\\right', '', expr)

    # \frac{a}{b} -> ((a))/((b))
    def _replace_frac(match):
        numerator, denominator = match.group(1), match.group(2)
        return f'(({numerator}))/(({denominator}))'
    expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', _replace_frac, expr)

    # Normalizar potencias de Euler: e**(algo) o E**(algo) -> exp(algo)
    expr = re.sub(r'\b[eE]\s*\*\*\s*\(([^()]+)\)', r'exp(\1)', expr)
    # Caso sin paréntesis: e**x -> exp(x)
    expr = re.sub(r'\b[eE]\s*\*\*\s*([A-Za-z0-9_]+)', r'exp(\1)', expr)

    expr = expr.replace('{', '(').replace('}', ')')
    expr = re.sub(r'\s+', '', expr)
    return expr


def parse_expression(expression: str, variable: str):
    local_dict = dict(ALLOWED_FUNCTIONS)
    local_dict[variable] = Symbol(variable)
    try:
        parsed = parse_expr(
            expression,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except (SympifyError, TokenError) as exc:
        raise ValueError(f'No se pudo interpretar el integrando: {exc}') from exc
    if parsed.has(AppliedUndef):
        raise ValueError('Se detectaron funciones no soportadas.')
    return parsed


def describe_features(expr, var: Symbol) -> List[str]:
    features: List[str] = []
    if expr.is_polynomial(var):
        features.append('Polinomio en la variable principal.')
    if expr.is_rational_function(var) and not expr.is_polynomial(var):
        features.append('Cociente de polinomios: candidato a fracciones parciales.')
        _, denominator = fraction(expr)
        poly = denominator.as_poly(var)
        if poly is not None:
            factors = poly.factor_list()[1]
            if any(multiplicity > 1 for _, multiplicity in factors):
                features.append('Se detectaron factores repetidos en el denominador.')
    if expr.has(log):
        features.append('Aparecen logaritmos naturales en el integrando.')
    if expr.has(exp):
        features.append(r'Incluye exponenciales $e^{x}$ u $\exp(x)$.')
    if expr.has(sin) or expr.has(cos) or expr.has(tan) or expr.has(cot) or expr.has(sec) or expr.has(csc):
        features.append('Contiene funciones trigonométricas.')
    if expr.has(sqrt):
        features.append('Incluye raíces cuadradas que podrían simplificarse con sustitución trigonométrica.')
    if expr.has(var) and expr.is_Mul:
        features.append('Producto de factores con la variable principal.')
    return features


def is_transcendental_factor(expr, var: Symbol) -> bool:
    transcendental_funcs = (sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, exp, log)
    return any(expr.has(func) for func in transcendental_funcs) or expr.has(sqrt)


def indicates_parts(expr, var: Symbol) -> bool:
    # Casos clásicos: x*exp(ax), x*sin(bx), x*cos(bx), x*log(x), etc.
    single_argument_funcs = (log, asin, acos, atan)
    if expr.func in single_argument_funcs and len(expr.args) == 1 and expr.args[0] == var:
        return True
    if expr.has(log) and not expr.is_Mul and expr.as_poly(var) is None:
        return True

    terms = expr.as_ordered_terms()
    for term in terms:
        factors = term.as_ordered_factors()
        if len(factors) < 2:
            continue
        poly_like = any((factor.as_poly(var) is not None and factor.as_poly(var).degree() >= 1)
                        for factor in factors if factor.has(var))
        transc_like = any(is_transcendental_factor(factor, var) for factor in factors)
        if poly_like and transc_like:
            return True
    return False


def indicates_substitution(expr, var: Symbol) -> bool:
    composite_candidates = expr.atoms(exp, log, sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, sqrt)
    for func_expr in composite_candidates:
        if not func_expr.args:
            continue
        inner = func_expr.args[0]
        derivative = diff(inner, var)
        if derivative == 0:
            continue
        try:
            ratio = simplify(expr / derivative)
        except Exception:
            continue
        if ratio.has(func_expr):
            return True
        if expr.has(derivative) and expr.has(func_expr):
            return True
    power_candidates = [term for term in expr.atoms(Pow) if term.has(var)]
    for pow_expr in power_candidates:
        base = pow_expr.base
        if base.has(var):
            derivative = diff(base, var)
            if derivative != 0 and expr.has(derivative):
                return True
    if expr.is_rational_function(var) and not expr.is_polynomial(var):
        numerator, denominator = fraction(expr)
        denominator = simplify(denominator)
        derivative = diff(denominator, var)
        if derivative != 0:
            if simplify(expr - derivative / denominator) == 0:
                return True
            if simplify(numerator - derivative) == 0:
                return True
            try:
                ratio = simplify(numerator / derivative)
            except Exception:
                ratio = None
            if ratio is not None:
                if not ratio.free_symbols:
                    return True
                try:
                    if ratio.is_constant(var):
                        return True
                except Exception:
                    if not ratio.free_symbols:
                        return True
                if simplify(ratio - 1 / denominator) == 0:
                    return True
    return False


def detect_rational_method(expr, var: Symbol) -> str | None:
    if not expr.is_rational_function(var) or expr.is_polynomial(var):
        return None

    numerator, denominator = fraction(expr)
    denominator = simplify(denominator)
    derivative = simplify(diff(denominator, var))

    if derivative != 0:
        if simplify(numerator - derivative) == 0:
            return 'substitution'
        try:
            ratio = simplify(numerator / derivative)
        except Exception:
            ratio = None
        if ratio is not None:
            try:
                if ratio.is_constant(var):
                    return 'substitution'
            except Exception:
                if not ratio.free_symbols:
                    return 'substitution'
            if not ratio.free_symbols:
                return 'substitution'
        if simplify(expr - derivative / denominator) == 0:
            return 'substitution'

    poly = denominator.as_poly(var)
    if poly is not None:
        if poly.degree() <= 1:
            return 'substitution'
        factors = poly.factor_list()[1]
        if len(factors) == 1 and factors[0][1] == 1:
            factor_poly = factors[0][0].as_poly(var)
            if factor_poly is not None and factor_poly.degree() <= 2:
                return 'substitution'
        if any(multiplicity > 1 for _, multiplicity in factors):
            return 'repeated_factors'

    return 'partial_fractions'


def detect_method(expr, var: Symbol) -> str:
    if expr.has(sqrt):
        for radicand in expr.atoms(sqrt):
            inner = simplify(radicand.args[0])
            description = describe_trig_substitution(inner, var)
            if description:
                return 'trig'

    if indicates_parts(expr, var):
        return 'parts'

    rational_method = detect_rational_method(expr, var)
    if rational_method:
        return rational_method

    if indicates_substitution(expr, var):
        return 'substitution'

    return 'substitution' if expr.has(exp, log, sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, sqrt) else 'default'


@app.route('/')
def root():
    return send_from_directory('.', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    payload = request.get_json(silent=True) or {}
    expression = payload.get('expression', '')
    variable_name = payload.get('variable', 'x')
    variable_name = re.sub(r'[^a-zA-Z]', '', variable_name) or 'x'

    sanitized = sanitize_expression(expression, variable_name)
    if not sanitized:
        return jsonify({'status': 'error', 'error': 'No se recibió ningún integrando.'}), 400

    try:
        expr = parse_expression(sanitized, variable_name)
    except ValueError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 400

    var_symbol = symbols(variable_name)
    features = describe_features(expr, var_symbol)
    method_key = detect_method(expr, var_symbol)
    method = METHOD_DETAILS.get(method_key, METHOD_DETAILS['default'])

    warnings: List[str] = []
    extra_symbols = [latexize(sym) for sym in expr.free_symbols if sym != var_symbol]
    if extra_symbols:
        warnings.append('Se detectaron otras variables en el integrando: ' + ', '.join(extra_symbols))

    analysis = {
        'latex_integral': make_integral_latex(expr, var_symbol),
        'variable': variable_name,
        'detected_features': features,
        'warnings': warnings,
        'method_key': method_key,
    }

    generator = EXAMPLE_GENERATORS.get(method_key)
    try:
        example_payload = generator(expr, var_symbol) if generator else generate_default_example(var_symbol)
    except Exception:
        example_payload = generate_default_example(var_symbol)

    method_payload = dict(method)
    method_payload['key'] = method_key
    method_payload.update(example_payload)

    response = {'status': 'ok', 'analysis': analysis, 'method': method_payload}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
