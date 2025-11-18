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
    Abs,
    together,  # 👈 IMPORTANTE para fracciones parciales
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

    # Quitar \left y \right que ensucian un poco
    replacements = [
        (r'\\left', ''),
        (r'\\right', ''),
    ]
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)

    # SymPy suele sacar \operatorname{asin}, \operatorname{acos}, etc.
    # Los convertimos a comandos "normales"
    cleaned = re.sub(
        r'\\operatorname\{([A-Za-z]+)\}',
        lambda m: "\\" + m.group(1),
        cleaned,
    )

    # Mapear funciones que MathJax NO conoce (\asin, \acos, \atan)
    # a las que sí conoce (\arcsin, \arccos, \arctan).
    func_map = {
        r'\asin': r'\arcsin',
        r'\acos': r'\arccos',
        r'\atan': r'\arctan',
        # por si acaso algún día salen hiperbólicas inversas:
        r'\asinh': r'\operatorname{arsinh}',
        r'\acosh': r'\operatorname{arcosh}',
        r'\atanh': r'\operatorname{artanh}',
    }
    for bad, good in func_map.items():
        cleaned = cleaned.replace(bad, good)

    # log → ln
    cleaned = re.sub(r'\\log(?!_)', r'\\ln', cleaned)

    # d en modo texto
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
    return rf"\int {latexize(expr)}\,d{latexize(variable)}"


def format_antiderivative(expr, variable: Symbol) -> str:
    """
    Formatea la antiderivada en LaTeX y añade + c.
    Además, corrige algunas formas raras que devuelve SymPy,
    como log(tan(x)**2 + 1)/2 en integrales de tan(x).
    """
    constant_suffix = ' + c'

    # Si SymPy no pudo integrar y dejó un Integral, lo devolvemos tal cual
    if isinstance(expr, Integral):
        return latexize(expr) + constant_suffix

    from sympy import log, tan, sec, Wild, simplify, expand

    # --- Corrección específica: ∫ tan(u) dx ---
    u = Wild('u')
    match = expr.match(log(tan(u)**2 + 1) / 2)
    if match:
        # log(tan(u)^2 + 1)/2  ->  log(sec(u))
        expr = log(sec(match[u]))

    # 🔧 NUEVO: para polinomios en la variable principal, usamos expand
    # para evitar que SymPy los devuelva factorizados.
    poly = expr.as_poly(variable)
    if poly is not None:
        expr_to_show = expand(expr)
    else:
        expr_to_show = simplify(expr)

    return f"{latexize(expr_to_show)}{constant_suffix}"


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
    return rf"{name} = {latexize(simplified)}\,d{latexize(variable)}"


def format_dx_from_du(variable: Symbol, derivative) -> str:
    simplified = simplify(derivative)
    if simplify(simplified - 1) == 0:
        return rf"d{latexize(variable)} = du"
    return rf"d{latexize(variable)} = \frac{{du}}{{{latexize(simplified)}}}"


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
            return {'pattern': pattern_label, 'a': a_val, 'b': b_val, 'inner': inner}
    return None


def find_trig_pattern(expr, var: Symbol):
    """
    Busca patrones de sustitución trigonométrica tanto en sqrt(...)
    como en potencias con exponente 1/2 o -1/2.
    """
    candidates = []

    # sqrt(...)
    for term in expr.atoms(sqrt):
        if term.has(var):
            candidates.append(term.args[0])

    # (algo)^(1/2) o (algo)^(-1/2)
    for term in expr.atoms(Pow):
        if not term.has(var):
            continue
        if term.exp == Rational(1, 2) or term.exp == Rational(-1, 2):
            candidates.append(term.base)

    for inner in candidates:
        info = describe_trig_substitution(inner, var)
        if info:
            return info
    return None


def generate_substitution_example(expr, var: Symbol):
    """
    Genera un ejemplo por sustitución parecido al integrando del usuario.

    - Si el integrando es racional (cociente de polinomios), usa el patrón
      (g'(x))/g(x) → log.
    - Si el integrando contiene funciones compuestas (exp(g(x)), sin(g(x)), ...),
      genera un ejemplo con LA MISMA función exterior (exp, sin, cos, etc.)
      y un polinomio interior parecido.
    - Si nada de eso aplica, usa un ejemplo genérico.
    """
    """
    Genera un ejemplo por sustitución parecido al integrando del usuario.
    """

    x = var

    # --- 0) Caso especial: k/(a + x^(1/n)) con n = 2 o 3 (raíces) ---
    radical_pows = [
        p for p in expr.atoms(Pow)
        if p.base == x and p.exp in (Rational(1, 2), Rational(1, 3))
    ]
    if radical_pows:
        root = radical_pows[0]
        K = Wild('K', exclude=[x])
        a = Wild('a', exclude=[x])

        simplified = simplify(expr)
        m = simplified.match(K / (a + root)) or simplified.match(K / (root + a))
        if m and m.get(K) is not None and m.get(a) is not None:
            K_val = m[K]
            a_val = m[a]

            # Constantes "parecidas" pero no idénticas (si son enteras)
            K_like = K_val
            a_like = a_val
            if isinstance(K_val, Integer):
                K_like = Integer(_perturb_int(int(K_val)))
            if isinstance(a_val, Integer):
                a_like = Integer(_perturb_int(int(a_val)))

            example_integrand = simplify(K_like / (a_like + root))
            antiderivative = integrate(example_integrand, x)

            # x = t^n  con n = 2 (raíz cuadrada) o 3 (raíz cúbica)
            n = int(1 / root.exp)   # 2 para 1/2, 3 para 1/3
            t = Symbol('t')
            x_sub = t**n
            dx_sub = diff(x_sub, t)
            integrand_t = simplify(example_integrand.subs(x, x_sub) * dx_sub)

            substitution_table = (
                r"\begin{aligned}"
                rf"x &= t^{n}\\"
                rf"dx &= {latexize(dx_sub)}\,dt\\"
                rf"x^{{1/{n}}} &= t"
                r"\end{aligned}"
            )

            setup = [
                {"label": "Sustitución de raíz", "value": substitution_table},
                {"label": "Integral en la nueva variable", "value": make_integral_latex(integrand_t, t)},
            ]

            steps = [
                build_step(
                    "1) Detectamos la raíz en el denominador",
                    "Al tener una raíz de la variable en el denominador, conviene eliminarla con una sustitución del tipo $x = t^n$.",
                    [make_integral_latex(example_integrand, x)],
                ),
                build_step(
                    "2) Planteamos el cambio de variable",
                    f"Tomamos $x = t^{n}$ para que $x^{{1/{n}}} = t$ y el denominador quede lineal en la nueva variable.",
                    [substitution_table],
                ),
                build_step(
                    "3) Reescribimos la integral en $t$",
                    "Sustituimos $x$ y $dx$ en la integral original y simplificamos el integrando.",
                    [make_integral_latex(integrand_t, t)],
                ),
                build_step(
                    "4) Integramos y volvemos a $x$",
                    "Integramos en $t$ y volvemos a expresar el resultado en función de $x$.",
                    [format_antiderivative(antiderivative, x)],
                ),
            ]

            return {
                "example_integral": make_integral_latex(example_integrand, x),
                "example_solution": format_antiderivative(antiderivative, x),
                "setup": setup,
                "steps": steps,
            }

    # 1) Caso típico racional: (g'(x))/g(x) → log
    if expr.is_rational_function(var) and not expr.is_polynomial(var):
        _, denominator = fraction(expr)
        inner = simplify(denominator)
        derived = simplify(diff(inner, var))
        if derived != 0:
            # parecido pero no idéntico al denominador original
            shifted_inner = simplify(inner + 1)
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
                {'label': 'Integral en u', 'value': rf"\int {reduced_integrand_tex}\,du"},
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
                    [rf"\int {reduced_integrand_tex}\,du = {format_antiderivative(reduced_antiderivative, u_symbol)}"],
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

    # 2) Caso no racional: imitar la MISMA función compuesta del usuario
    composite_candidates = list(
        expr.atoms(exp, log, sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, sqrt)
    )
    for func_expr in composite_candidates:
        if not func_expr.args:
            continue
        inner = func_expr.args[0]
        if not inner.has(var):
            continue

        # derivada del interior
        derived = simplify(diff(inner, var))
        if derived == 0:
            continue

        # interior "parecido" pero perturbado
        inner_like = simplify(perturb_constants(inner) + 1)
        derived_like = simplify(diff(inner_like, var))

        if derived_like == 0:
            continue

        outer = func_expr.func  # exp, sin, cos, ...

        example_integrand = simplify(derived_like * outer(inner_like))
        antiderivative = integrate(example_integrand, var)

        u_symbol = Symbol('u')
        du_value = format_differential('du', derived_like, var)
        dx_value = format_dx_from_du(var, derived_like)
        reduced_integrand = outer(u_symbol)
        reduced_integral = integrate(reduced_integrand, u_symbol)

        inner_tex = latexize(inner_like)
        derived_tex = latexize(derived_like)
        reduced_integrand_tex = latexize(reduced_integrand)
        reduced_integral_tex = format_antiderivative(reduced_integral, u_symbol)

        setup = [
            {'label': 'u(x)', 'value': rf"u(x) = {inner_tex}"},
            {'label': 'du', 'value': du_value},
            {'label': 'dx', 'value': dx_value},
            {'label': 'Integral en u', 'value': rf"\int {reduced_integrand_tex}\,du"},
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
                [rf"\int {reduced_integrand_tex}\,du = {reduced_integral_tex}"],
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
            # 3) Caso potencia algebraica: g'(x)*(g(x))^n
    power_candidates = [term for term in expr.atoms(Pow) if term.has(var)]
    for pow_expr in power_candidates:
        base = pow_expr.base
        if not base.has(var):
            continue

        der = simplify(diff(base, var))
        if der == 0:
            continue

        try:
            ratio = simplify(expr / der)
        except Exception:
            continue

        C = Wild('C', exclude=[base])
        p = Wild('p')

        match = ratio.match(C * base**p)
        if not match:
            continue

        C_val = match[C]
        if C_val is None:
            continue

        # C_val constante respecto a la variable
        try:
            if hasattr(C_val, "is_constant") and not C_val.is_constant(var):
                continue
        except Exception:
            if var in C_val.free_symbols:
                continue

        exponent = match[p]
        inner = base

        # Hacemos una versión "parecida" de g(x) perturbando constantes
        inner_like = simplify(perturb_constants(inner) + 1)
        der_like = simplify(diff(inner_like, var))

        # Integrando de ejemplo: g'(x)*(g(x))^n (mismo patrón que el usuario)
        example_integrand = simplify(der_like * inner_like**exponent)
        antiderivative = integrate(example_integrand, var)

        u_symbol = Symbol('u')
        du_value = format_differential('du', der_like, var)
        dx_value = format_dx_from_du(var, der_like)
        reduced_integrand = u_symbol**exponent
        reduced_integral = integrate(reduced_integrand, u_symbol)

        inner_tex = latexize(inner_like)
        reduced_integrand_tex = latexize(reduced_integrand)
        reduced_integral_tex = format_antiderivative(reduced_integral, u_symbol)

        setup = [
            {'label': 'u(x)', 'value': rf"u(x) = {inner_tex}"},
            {'label': 'du', 'value': du_value},
            {'label': 'dx', 'value': dx_value},
            {'label': 'Integral en u', 'value': rf"\int {reduced_integrand_tex}\,du"},
        ]

        steps = [
            build_step(
                '1) Identificamos la potencia compuesta',
                (
                    'Reconocemos una expresión del tipo $g(x)^n$ acompañada de '
                    'su derivada $g\'(x)$, lo que sugiere una sustitución directa.'
                ),
                [make_integral_latex(example_integrand, var)],
            ),
            build_step(
                '2) Declaramos $u(x)$ y el diferencial',
                'Tomamos como variable auxiliar la base de la potencia compuesta.',
                [rf"u(x) = {inner_tex}", du_value, dx_value],
            ),
            build_step(
                '3) Integramos en términos de $u$',
                'La integral se convierte en una potencia de $u$ fácil de integrar.',
                [rf"\int {reduced_integrand_tex}\,du = {reduced_integral_tex}"],
            ),
            build_step(
                '4) Volvemos a la variable original',
                'Reemplazamos $u$ por la expresión inicial para obtener la antiderivada.',
                [format_antiderivative(antiderivative, var)],
            ),
        ]

        return {
            'example_integral': make_integral_latex(example_integrand, var),
            'example_solution': format_antiderivative(antiderivative, var),
            'setup': setup,
            'steps': steps,
        }


    # 3) Fallback genérico si todo lo anterior falla
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
        {'label': 'Integral en u', 'value': rf"\int {reduced_integrand_tex}\,du"},
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
            [rf"\int {reduced_integrand_tex}\,du = {reduced_integral_tex}"],
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
    """
    Separa el producto en (polinomio en x) * (resto).

    Evita elegir constantes como polinomio; busca grado >= 1.
    """
    factors = list(expr.as_ordered_factors()) if expr.is_Mul else [expr]
    polynomial = None
    other = None

    # Buscar factor polinómico de grado >= 1
    for factor_candidate in factors:
        poly = factor_candidate.as_poly(var)
        if poly is not None and poly.degree() >= 1:
            polynomial = factor_candidate
            break

    if polynomial is None:
        # Si no encontramos, tomamos el primer factor que tenga la variable
        for factor_candidate in factors:
            if factor_candidate.has(var):
                polynomial = factor_candidate
                break

    if polynomial is None:
        # Fallback: usamos var como polinomio artificial
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
             rf"\int {product_integral_tex}\,d{var_tex} = {u_tex}{v_tex} - \int {v_tex}\,{du_tex}\,d{var_tex}"],
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
    """
    Genera un ejemplo de sustitución trigonométrica CLÁSICO según el tipo de raíz
    que aparezca en el integrando del usuario.

    Detecta el signo del polinomio cuadrático bajo la raíz:

      √(a² - x²)    -> caso seno      -> ejemplo: ∫ 1/√(9 - x²) dx      -> arcsin(x/3)
      √(a² + x²)    -> caso tangente  -> ejemplo: ∫ 1/√(4 + 9x²) dx     -> (1/3) ln|3x + √(4+9x²)|
      √(x² - a²)    -> caso secante   -> ejemplo: ∫ 1/√(9x² - 4) dx     -> (1/3) ln|3x + √(9x²-4)|
    """

    from sympy import asin, log, sqrt, sec, tan

    x = var
    theta = Symbol("theta")
    var_tex = latexize(x)
    theta_tex = latexize(theta)

    # ===================== 1) Buscamos "la" raíz cuadrada del integrando =====================
    inner = None

    # Raíces como sqrt(...)
    for term in expr.atoms(sqrt):
        if term.has(x):
            inner = term.args[0]
            break

    # O potencias 1/2, -1/2: (algo)^(1/2)
    if inner is None:
        for term in expr.atoms(Pow):
            if term.has(x) and term.exp in (Rational(1, 2), Rational(-1, 2)):
                inner = term.base
                break

    # Si no encontramos nada, usamos el caso tangente por defecto
    case = "tangent"

    if inner is not None:
        inner = simplify(inner)
        poly = inner.as_poly(x)
        if poly is not None and poly.degree() == 2:
            A = poly.LC()        # coeficiente de x^2
            C = poly.eval(0)     # término independiente

            # Solo intentamos clasificar si A y C son numéricos
            if A.is_number and C.is_number:
                # Forma ~ a^2 - x^2   -> A < 0, C > 0
                if A.evalf() < 0 and C.evalf() > 0:
                    case = "sine"
                # Forma ~ a^2 + x^2   -> A > 0, C > 0
                elif A.evalf() > 0 and C.evalf() > 0:
                    case = "tangent"
                # Forma ~ x^2 - a^2   -> A > 0, C < 0
                elif A.evalf() > 0 and C.evalf() < 0:
                    case = "secant"
                else:
                    case = "tangent"

    # ===================== 2) Definimos el EJEMPLO MODELO según el caso =====================

    if case == "sine":
        # ∫ 1 / √(9 - x²) dx  ---> arcsin(x/3)
        example_integrand = 1 / sqrt(9 - x**2)

        # Sustitución clásica:  x = 3 sinθ
        substitution_expr = 3 * sin(theta)
        dx_theta_expr = 3 * cos(theta)
        root_theta_expr = 3 * cos(theta)

        classical_table = (
            r"\begin{aligned}"
            r"x &= 3\sin\theta\\"
            r"dx &= 3\cos\theta\,d\theta\\"
            r"\sqrt{9-x^2} &= 3\cos\theta"
            r"\end{aligned}"
        )

        # Integral en θ, escrita a mano:
        # ∫ 1/√(9-x²) dx = ∫ 3cosθ / √(9-9sin²θ) dθ = ∫ 1 dθ = θ
        theta_integral_latex = (
            r"\int \frac{1}{\sqrt{9-x^2}}\,dx"
            r" = \int \frac{3\cos\theta}{\sqrt{9-9\sin^2\theta}}\,d\theta"
            r" = \int 1\,d\theta = \theta"
        )

        antiderivative_expr = asin(x / 3)

    elif case == "secant":
        # ∫ 1 / √(9x² - 4) dx  ---> (1/3) ln| 3x + √(9x²-4) |
        example_integrand = 1 / sqrt(9 * x**2 - 4)

        # Modelo: 9x²-4 = (3x)² - 2²  -> x = (2/3) secθ
        substitution_expr = (2 / 3) * sec(theta)
        dx_theta_expr = (2 / 3) * sec(theta) * tan(theta)
        root_theta_expr = 2 * tan(theta)

        classical_table = (
            r"\begin{aligned}"
            r"x &= \frac{2}{3}\sec\theta\\"
            r"dx &= \frac{2}{3}\sec\theta\tan\theta\,d\theta\\"
            r"\sqrt{9x^2-4} &= 2\tan\theta"
            r"\end{aligned}"
        )

        # ∫ 1/√(9x²-4) dx = (1/3) ∫ secθ dθ
        theta_integral_latex = (
            r"\int \frac{1}{\sqrt{9x^2-4}}\,dx"
            r" = \int \frac{\tfrac{2}{3}\sec\theta\tan\theta}{2\tan\theta}\,d\theta"
            r" = \int \frac{1}{3}\sec\theta\,d\theta"
            r" = \frac{1}{3}\ln\left|\sec\theta+\tan\theta\right|"
        )

        antiderivative_expr = log(3 * x + sqrt(9 * x**2 - 4)) / 3

    else:  # case == "tangent"
        # ∫ 1 / √(4 + 9x²) dx  ---> (1/3) ln| 3x + √(4+9x²) |
        example_integrand = 1 / sqrt(4 + 9 * x**2)

        # 4+9x² = 2² + (3x)²  -> x = (2/3) tanθ
        substitution_expr = (2 / 3) * tan(theta)
        dx_theta_expr = (2 / 3) * sec(theta)**2
        root_theta_expr = 2 * sec(theta)

        classical_table = (
            r"\begin{aligned}"
            r"x &= \frac{2}{3}\tan\theta\\"
            r"dx &= \frac{2}{3}\sec^2\theta\,d\theta\\"
            r"\sqrt{4+9x^2} &= 2\sec\theta"
            r"\end{aligned}"
        )

        # ∫ 1/√(4+9x²) dx = (1/3) ∫ secθ dθ
        theta_integral_latex = (
            r"\int \frac{1}{\sqrt{4+9x^2}}\,dx"
            r" = \int \frac{\tfrac{2}{3}\sec^2\theta}{2\sec\theta}\,d\theta"
            r" = \int \frac{1}{3}\sec\theta\,d\theta"
            r" = \frac{1}{3}\ln\left|\sec\theta+\tan\theta\right|"
        )

        antiderivative_expr = log(3 * x + sqrt(4 + 9 * x**2)) / 3

    # ===================== 3) Construimos setup + pasos =====================

    substitution = rf"{var_tex} = {latexize(substitution_expr)}"
    differential = rf"d{var_tex} = {latexize(dx_theta_expr)}\,d{theta_tex}"
    root_theta_tex = latexize(root_theta_expr)
    final_antiderivative_latex = format_antiderivative(antiderivative_expr, x)

    setup = [
        {"label": "Tabla trigonométrica clásica", "value": classical_table},
        {"label": "Sustitución usada", "value": substitution},
        {"label": "Diferencial", "value": differential},
        {"label": "Raíz en función de θ", "value": root_theta_tex},
    ]

    steps = [
        build_step(
            "1) Reconocemos el patrón cuadrático",
            "La raíz encaja en uno de los patrones clásicos de sustitución trigonométrica.",
            [make_integral_latex(example_integrand, x)],
        ),
        build_step(
            "2) Planteamos la sustitución",
            "Usamos la fila adecuada de la tabla y escribimos x, dx y la raíz en términos de θ.",
            [classical_table, substitution, differential, root_theta_tex],
        ),
        build_step(
            r"3) Integramos en $\theta$",
            "Sustituimos en la integral y simplificamos hasta llegar a una integral conocida en θ.",
            [theta_integral_latex],
        ),
        build_step(
            "4) Volvemos a la variable original",
            "Usamos la relación trigonométrica para expresar el resultado final en función de x.",
            [final_antiderivative_latex],
        ),
    ]

    return {
        "example_integral": make_integral_latex(example_integrand, x),
        "example_solution": final_antiderivative_latex,
        "setup": setup,
        "steps": steps,
    }



def generate_partial_fractions_example(expr, var: Symbol):
    A, B, C = symbols('A B C')

    numerator = 2*var**2 + 4*var + 6
    denominator = (var - 1) * (var + 2) * (var + 3)
    example_integrand = numerator / denominator

    # Descomposición propuesta
    decomposition = A/(var - 1) + B/(var + 2) + C/(var + 3)

    # Identidad de numeradores
    rhs_raw = A*(var + 2)*(var + 3) + B*(var - 1)*(var + 3) + C*(var - 1)*(var + 2)
    rhs_expanded = expand(rhs_raw)

    # Coeficientes y sistema
    eq1 = Eq(A + B + C, 2)
    eq2 = Eq(5*A + 2*B + C, 4)
    eq3 = Eq(6*A - 3*B - 2*C, 6)

    solutions = solve((eq1, eq2, eq3), (A, B, C), dict=True)
    sol = solutions[0] if solutions else {A: Rational(3, 2), B: Rational(-1, 2), C: 1}

    A_val = sol[A]
    B_val = sol[B]
    C_val = sol[C]

    decomposition_specific = decomposition.subs({A: A_val, B: B_val, C: C_val})
    antiderivative = integrate(example_integrand, var)

    # TEXTO / ECUACIONES PARA LOS PASOS
    identidad_numeradores_1 = Eq(numerator, rhs_raw)
    identidad_numeradores_2 = Eq(numerator, rhs_expanded)

    sistema_latex = (
        r"\begin{cases}"
        + latexize(eq1) + r"\\"
        + latexize(eq2) + r"\\"
        + latexize(eq3) +
        r"\end{cases}"
    )

    asignaciones = [
        latexize(Eq(A, A_val)),
        latexize(Eq(B, B_val)),
        latexize(Eq(C, C_val)),
    ]

    setup = [
        {
            'label': 'Denominador factorizado',
            'value': latexize(denominator),
        },
        {
            'label': 'Descomposición propuesta',
            'value': latexize(decomposition),
        },
    ]

    steps = [
        build_step(
            '1) Planteamos la descomposición',
            (
                'Al tener tres factores lineales distintos en el denominador, '
                'suponemos una combinación de fracciones simples con constantes '
                '$A$, $B$ y $C$.'
            ),
            [
                latexize(
                    Eq(
                        example_integrand,
                        decomposition
                    )
                )
            ],
        ),
        build_step(
            '2) Multiplicamos por el denominador común',
            (
                'Multiplicamos ambos lados por $(x-1)(x+2)(x+3)$ para eliminar '
                'los denominadores y obtener una identidad entre polinomios.'
            ),
            [
                latexize(identidad_numeradores_1),
                latexize(identidad_numeradores_2),
            ],
        ),
        build_step(
            '3) Igualamos coeficientes',
            (
                'Al comparar los coeficientes de $x^2$, $x$ y el término constante, '
                'obtenemos un sistema lineal para $A$, $B$ y $C$.'
            ),
            [sistema_latex] + asignaciones,
        ),
        build_step(
            '4) Integramos término a término',
            (
                'Sustituimos los valores de $A$, $B$ y $C$ en la descomposición '
                'y calculamos la antiderivada sumando las integrales básicas.'
            ),
            [
                latexize(decomposition_specific),
                format_antiderivative(antiderivative, var),
            ],
        ),
    ]

    return {
        'example_integral': make_integral_latex(example_integrand, var),
        'example_solution': format_antiderivative(antiderivative, var),
        'setup': setup,
        'steps': steps,
    }

def generate_repeated_factors_example(expr, var: Symbol):
    A, B, C = symbols('A B C')

    numerator = var**2 + 2*var + 3
    denominator = (var - 1) * (var + 1)**2
    example_integrand = numerator / denominator

    # Descomposición estándar con factor repetido
    decomposition = A/(var - 1) + B/(var + 1) + C/(var + 1)**2

    # Identidad de numeradores
    rhs_raw = A*(var + 1)**2 + B*(var - 1)*(var + 1) + C*(var - 1)
    rhs_expanded = expand(rhs_raw)

    # Sistema a partir de coeficientes
    eq1 = Eq(A + B, 1)      # coef x^2
    eq2 = Eq(2*A + C, 2)    # coef x
    eq3 = Eq(A - B - C, 3)  # término independiente

    solutions = solve((eq1, eq2, eq3), (A, B, C), dict=True)
    sol = solutions[0] if solutions else {A: Rational(3, 2), B: Rational(-1, 2), C: -1}

    A_val = sol[A]
    B_val = sol[B]
    C_val = sol[C]

    decomposition_specific = decomposition.subs({A: A_val, B: B_val, C: C_val})
    antiderivative = integrate(example_integrand, var)

    identidad_numeradores_1 = Eq(numerator, rhs_raw)
    identidad_numeradores_2 = Eq(numerator, rhs_expanded)

    sistema_latex = (
        r"\begin{cases}"
        + latexize(eq1) + r"\\"
        + latexize(eq2) + r"\\"
        + latexize(eq3) +
        r"\end{cases}"
    )

    asignaciones = [
        latexize(Eq(A, A_val)),
        latexize(Eq(B, B_val)),
        latexize(Eq(C, C_val)),
    ]

    setup = [
        {
            'label': 'Denominador factorizado con factor repetido',
            'value': latexize(denominator),
        },
        {
            'label': 'Descomposición propuesta',
            'value': latexize(decomposition),
        },
    ]

    steps = [
        build_step(
            '1) Reconocemos el factor repetido',
            (
                'El denominador tiene $(x+1)^2$; por ello, en la descomposición '
                'aparece un término con $(x+1)$ y otro con $(x+1)^2$.'
            ),
            [
                latexize(
                    Eq(
                        example_integrand,
                        decomposition
                    )
                )
            ],
        ),
        build_step(
            '2) Multiplicamos por el denominador común',
            (
                'Multiplicamos ambos lados por $(x-1)(x+1)^2$ para eliminar '
                'los denominadores y obtener una igualdad entre polinomios.'
            ),
            [
                latexize(identidad_numeradores_1),
                latexize(identidad_numeradores_2),
            ],
        ),
        build_step(
            '3) Igualamos coeficientes',
            (
                'Agrupamos los términos en $x^2$, $x$ y constantes y formamos '
                'un sistema de tres ecuaciones para $A$, $B$ y $C$.'
            ),
            [sistema_latex] + asignaciones,
        ),
        build_step(
            '4) Integramos cada término',
            (
                'Sustituimos los valores hallados de $A$, $B$ y $C$ y calculamos '
                'las integrales tipo logaritmo y potencia que aparecen.'
            ),
            [
                latexize(decomposition_specific),
                format_antiderivative(antiderivative, var),
            ],
        ),
    ]

    return {
        'example_integral': make_integral_latex(example_integrand, var),
        'example_solution': format_antiderivative(antiderivative, var),
        'setup': setup,
        'steps': steps,
    }
def generate_default_example(var: Symbol):
    # Ejemplo sencillo: polinomio
    example_integrand = var**2 + 2 * var + 3

    # Antiderivada y la dejamos en forma expandida
    antiderivative = expand(integrate(example_integrand, var))

    # Integral separada en términos independientes (para que coincida con el texto)
    separated_integral = (
        r"\int x^2\,dx \;+\; \int 2x\,dx \;+\; \int 3\,dx"
    )

    steps = [
        build_step(
            '1) Separar en sumas manejables',
            'Dividimos la integral en términos independientes.',
            [
                latexize(example_integrand),   # x^2 + 2x + 3
                separated_integral,            # ∫x^2 dx + ∫2x dx + ∫3 dx
            ],
        ),
        build_step(
            '2) Aplicar reglas básicas',
            r'Integramos cada potencia usando la regla '
            r'$\int x^{n}\,dx = \dfrac{x^{n+1}}{n+1}$.',
            [
                format_antiderivative(antiderivative, var),  # x^3/3 + x^2 + 3x + c
            ],
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
    # raíces cuadradas como sqrt(...) o potencias 1/2 / -1/2
    has_root = bool(expr.has(sqrt))
    if not has_root:
        for term in expr.atoms(Pow):
            if term.exp == Rational(1, 2) or term.exp == Rational(-1, 2):
                has_root = True
                break
    if has_root:
        features.append('Incluye raíces cuadradas que podrían simplificarse con sustitución trigonométrica.')
    if expr.has(var) and expr.is_Mul:
        features.append('Producto de factores con la variable principal.')
    return features


def is_transcendental_factor(expr, var: Symbol) -> bool:
    """
    Devuelve True si el factor tiene funciones trascendentales 'clásicas'
    (trig, exp, log...). Las raíces sqrt NO las marcamos como trascendentales
    para evitar mandar cosas como x^3*sqrt(x^2-1) a integración por partes.
    """
    transcendental_funcs = (sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, exp, log)
    return any(expr.has(func) for func in transcendental_funcs)



def indicates_parts(expr, var: Symbol) -> bool:
    """
    Detecta integrales que realmente conviene hacer por partes.
    Evita marcar como 'partes' casos típicos de sustitución como
    2x*exp(x^2), x*cos(x^2), etc.
    """
    # Casos clásicos: ln(x), asin(x), acos(x), atan(x)
    single_argument_funcs = (log, asin, acos, atan)
    if expr.func in single_argument_funcs and len(expr.args) == 1 and expr.args[0] == var:
        return True

    # Casos como ln(g(x)) "suelto"
    if expr.has(log) and not expr.is_Mul and expr.as_poly(var) is None:
        return True

    # 🔥 NUEVO BLOQUE:
    # productos tipo ln(x) * x^n (incluye n negativo), por ejemplo ln(x)/x^2
    if expr.has(log(var)):
        for factor in expr.as_ordered_factors():
            # x, x^n o x^(-n)
            if factor == var:
                return True
            if factor.is_Pow and factor.base == var:
                return True

    # ---------------- resto igual que antes ----------------
    terms = expr.as_ordered_terms()
    for term in terms:
        factors = term.as_ordered_factors()
        if len(factors) < 2:
            continue

        # ¿Hay un polinomio de grado >= 1?
        poly_like = any(
            (factor.as_poly(var) is not None and factor.as_poly(var).degree() >= 1)
            for factor in factors
            if factor.has(var)
        )

        # ¿Hay una parte trascendental?
        transc_like = any(is_transcendental_factor(factor, var) for factor in factors)

        if not (poly_like and transc_like):
            continue

        # Filtro: si la parte trascendental es f(g(x)) con g(x) de grado >= 2
        # (ej: exp(x^2), sin(x^3), ...), mejor sustitución simple.
        high_degree_inner = False
        for factor in factors:
            if factor.is_Function and factor.args:
                inner = factor.args[0]
                poly_inner = inner.as_poly(var)
                if poly_inner is not None and poly_inner.degree() >= 2:
                    high_degree_inner = True
                    break
        if high_degree_inner:
            continue

        # Buen candidato a partes: P(x)*sin(bx), P(x)*exp(ax), etc.
        return True

    return False

def indicates_substitution(expr, var: Symbol) -> bool:
    from sympy import fraction, sqrt, Add

    # --- 0) Caso especial: constante / (a + sqrt(x)) ---
    #
    # Ejemplo típico: 3/(4 + sqrt(x))
    # Didácticamente se hace con u = 4 + sqrt(x).
    try:
        num, den = fraction(expr)
        num = simplify(num)
        den = simplify(den)
    except Exception:
        num, den = None, None

    if num is not None and den is not None:
        # numerador constante (respecto a la variable de integración)
        if not num.has(var):
            a = Wild('a', exclude=[var])
            c = Wild('c', exclude=[var])

            # patrón a + c*sqrt(x)
            pattern1 = a + c*sqrt(var)
            pattern2 = c*sqrt(var) + a  # por si el orden es distinto

            if den.match(pattern1) or den.match(pattern2):
                return True

    # --- 1) Funciones compuestas: exp(g), log(g), sin(g), cos(g), sqrt(g), etc. ---
    composite_candidates = expr.atoms(
        exp, log, sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, sqrt
    )
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
        # Mismo patrón f(g(x)) * g'(x)
        if ratio.has(func_expr):
            return True
        if expr.has(derivative) and expr.has(func_expr):
            return True

    # --- 2) Potencias algebraicas: g'(x) * (g(x))^n (ej: x^2 (2x^3-5)^4) ---
    power_candidates = [term for term in expr.atoms(Pow) if term.has(var)]
    for pow_expr in power_candidates:
        base = pow_expr.base
        if not base.has(var):
            continue

        derivative = simplify(diff(base, var))
        if derivative == 0:
            continue

        try:
            ratio = simplify(expr / derivative)
        except Exception:
            continue

        C = Wild('C', exclude=[base])
        p = Wild('p')

        match = ratio.match(C * base**p)
        if not match:
            continue

        C_val = match[C]
        if C_val is None:
            continue

        # C_val debe ser constante respecto a 'var'
        try:
            if hasattr(C_val, "is_constant") and not C_val.is_constant(var):
                continue
        except Exception:
            if var in C_val.free_symbols:
                continue

        # Si llegamos aquí, hay patrón g'(x)*(g(x))^p → sustitución simple
        return True

    # --- 3) Caso racional clásico: (g'(x))/g(x) ---
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
        # 4) Casos tipo k/(a + x^(1/n)) con n = 2 o 3  → sustitución por raíz
    radical_pows = [
        p for p in expr.atoms(Pow)
        if p.base == var and p.exp in (Rational(1, 2), Rational(1, 3))
    ]
    if radical_pows:
        root = radical_pows[0]
        K = Wild('K', exclude=[var])
        a = Wild('a', exclude=[var])

        simplified = simplify(expr)
        match = simplified.match(K / (a + root)) or simplified.match(K / (root + a))
        if match and match.get(K) is not None and match.get(a) is not None:
            K_val = match[K]
            a_val = match[a]
            # Aseguramos que K y a sean constantes (no dependan de x)
            if (not K_val.has(var)) and (not a_val.has(var)):
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
def prefers_simple_sub_over_trig(expr, var: Symbol, inner):
    """
    Devuelve True solo si el integrando se ve como:
        const * g'(x) * (g(x))**p
    con g(x) = inner y 'const' realmente constante (no depende de x).
    Eso detecta casos limpios de u–sustitución como:
        2x*sqrt(x**2+1)
        x/sqrt(4-x**2)
    pero NO cosas como:
        x^3*sqrt(x^2-1)
        x^2*sqrt(2-x^2)
    que son mejores para sustitución trigonométrica.
    """
    der = simplify(diff(inner, var))
    if der == 0:
        return False

    try:
        ratio = simplify(expr / der)
    except Exception:
        return False

    C = Wild('C', exclude=[inner])
    p = Wild('p')

    match = ratio.match(C * inner**p)
    if not match:
        return False

    C_val = match[C]
    if C_val is None:
        return False

    # Pedimos que C_val sea CONSTANTE respecto a 'var'
    try:
        if hasattr(C_val, "is_constant"):
            return bool(C_val.is_constant(var))
    except Exception:
        pass

    # Fallback: que no aparezca la variable en los símbolos libres
    return var not in C_val.free_symbols


def detect_method(expr, var: Symbol) -> str:
    """
    Decide el método predominante.

    Reglas importantes:
    - Si se detecta un patrón de raíz cuadrática tipo a^2 ± x^2 o x^2 ± a^2:
        * Si además es un caso limpio de u–sustitución (derivada * una sola
          potencia de ese interior) → usamos 'substitution'.
        * En caso contrario → usamos 'trig'.
    """

    # 1) Miramos si hay patrón de raíz trigonométrica
    trig_info = find_trig_pattern(expr, var)
    if trig_info:
        inner = trig_info['inner']

        # ¿Se ve como g'(x) * (g(x))**p ? → preferimos sustitución simple
        if prefers_simple_sub_over_trig(expr, var, inner):
            return 'substitution'

        # Si no es un caso "limpio" de u-sub, lo consideramos trigonométrica
        return 'trig'

    # 2) Por partes (casos clásicos tipo x*e^x, x*sin x, ln x, etc.)
    if indicates_parts(expr, var):
        return 'parts'

    # 3) Racionales: fracciones parciales / sustitución / factores repetidos
    rational_method = detect_rational_method(expr, var)
    if rational_method:
        return rational_method

    # 4) Sustitución simple (u–sustitución) en el resto de casos compuestos
    if indicates_substitution(expr, var):
        return 'substitution'

    # 5) Si hay funciones compuestas pero nada muy claro, probamos con sustitución simple
    if expr.has(exp, log, sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, sqrt):
        return 'substitution'

    # 6) Fallback completamente básico
    return 'default'


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
