"""Modelo de utilidad y optimizacion del bid/ask de un formador de mercado.

El formador de mercado no observa si el proximo trader es informado o de
liquidez. Cotiza un bid B y un ask A alrededor del valor esperado S0 para
balancear dos fuerzas opuestas:
  - Con un trader de liquidez, cotizar lejos de S0 es rentable, pero reduce
    la probabilidad de que el trader acepte la cotizacion (pi_LB / pi_LS
    decrecen con la distancia a S0).
  - Con un trader informado (que conoce P, el valor verdadero), cualquier
    cotizacion se ejecuta siempre en contra del formador: si P > A, comprara
    al ask y el formador pierde (P-A); si P < B, vendera al bid y el
    formador pierde (B-P).
Pi(A,B) pondera ambos efectos por sus probabilidades pi_L y pi_I. Optimizar
A y B es entonces un tradeoff clasico entre "spread" (cobro por proveer
liquidez) y "adverse selection" (perdida esperada frente al informado).
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import erlang

S0 = 19.90
ERLANG_K = 60
ERLANG_LAMBDA = 3
PI_I = 0.40
PI_L = 0.60
COEF_INTERCEPTO_LIQUIDEZ = 0.50
COEF_PENDIENTE_LIQUIDEZ = 0.08

_DISTRIBUCION_VALOR = erlang(a=ERLANG_K, scale=1 / ERLANG_LAMBDA)


def densidad_valor(P):
    """f(P): densidad Erlang(k=60, lambda=3) del valor verdadero del activo.

    Esta es la creencia del formador de mercado sobre P antes de que llegue
    el trader informado. Con media k/lambda = 20 y varianza k/lambda^2,
    la masa de probabilidad se concentra cerca de S0=19.90, lo cual es
    consistente con S0 siendo la mejor estimacion previa del valor.
    """
    return _DISTRIBUCION_VALOR.pdf(P)


def prob_ejecucion_liquidez(s):
    """pi_LB(s) = pi_LS(s): prob. de que un trader de liquidez acepte una
    cotizacion a distancia s de S0.

    Decrece linealmente porque un trader de liquidez, al no tener ventaja
    informativa, es cada vez mas renuente a operar mientras peor sea el
    precio ofrecido respecto de S0. Se trunca en 0 (np.maximum) porque una
    probabilidad no puede ser negativa: mas alla de s=6.25 nadie acepta.
    """
    return np.maximum(COEF_INTERCEPTO_LIQUIDEZ - COEF_PENDIENTE_LIQUIDEZ * s, 0.0)


def _perdida_informada_ask(A):
    """E[(P-A) * 1{P>A}]: perdida esperada por vender al ask a un informado.

    Un trader informado compra al ask solo si el valor verdadero P supera
    a A, y en ese caso el formador deja de capturar la diferencia (P-A).
    Se integra con quad hasta np.inf porque la Erlang tiene cola de
    decaimiento exponencial (ligera) y quad maneja intervalos infinitos de
    forma nativa (QUADPACK/QAGI) sin necesidad de truncar arbitrariamente
    en un multiplo de la media, lo que evitaria introducir un error de
    truncamiento no controlado.
    """
    perdida, _ = quad(lambda P: (P - A) * densidad_valor(P), A, np.inf)
    return perdida


def _perdida_informada_bid(B):
    """E[(B-P) * 1{P<B}]: perdida esperada por comprar al bid a un informado.

    Simetrico a _perdida_informada_ask: el informado vende al bid solo si
    P < B, y el formador paga de mas por (B-P).
    """
    perdida, _ = quad(lambda P: (B - P) * densidad_valor(P), 0, B)
    return perdida


def utilidad_esperada(A, B, pi_I, pi_L, S0):
    """Pi(A,B): utilidad esperada por trade de un formador de mercado que
    cotiza bid B y ask A.

    Combina la ganancia esperada de traders de liquidez (que pagan spread
    por certeza de ejecucion) con la perdida esperada frente a traders
    informados (seleccion adversa). pi_I y pi_L son parametros -- no
    constantes fijas dentro de esta funcion -- porque el analisis de
    sensibilidad del Paso 4 reevalua Pi(A,B) variando pi_I.
    """
    ganancia_liquidez = pi_L * (
        prob_ejecucion_liquidez(A - S0) * (A - S0)
        + prob_ejecucion_liquidez(S0 - B) * (S0 - B)
    )
    perdida_informada = pi_I * (_perdida_informada_ask(A) + _perdida_informada_bid(B))
    return ganancia_liquidez - perdida_informada


def _objetivo_negativo(x, pi_I, pi_L, S0):
    """-Pi(A,B) en funcion de un vector x=[A,B], para minimizar con scipy."""
    A, B = x
    return -utilidad_esperada(A, B, pi_I, pi_L, S0)


def optimizar_cotizaciones(pi_I, pi_L, S0):
    """Encuentra el bid y ask que maximizan Pi(A,B) sujeto a B en (0,S0] y
    A en [S0, inf).

    Se usa el metodo L-BFGS-B porque las restricciones son bounds simples
    e independientes por variable (no hay restricciones que acoplen A y B),
    exactamente el caso para el que L-BFGS-B esta disenado; SLSQP existe
    para restricciones generales de igualdad/desigualdad que aqui no
    aplican, por lo que solo anadiria overhead innecesario. pi_I es
    parametro (no constante hardcodeada) para que el Paso 4 pueda llamar
    esta funcion repetidamente variando pi_I en {0.1, 0.4, 0.7}.
    """
    x0 = [S0 + 0.5, S0 - 0.5]
    limite_inferior_bid = 1e-6
    bounds = [(S0, None), (limite_inferior_bid, S0)]

    resultado = minimize(
        _objetivo_negativo,
        x0,
        args=(pi_I, pi_L, S0),
        method="L-BFGS-B",
        bounds=bounds,
    )
    A_opt, B_opt = resultado.x
    spread = A_opt - B_opt
    utilidad = utilidad_esperada(A_opt, B_opt, pi_I, pi_L, S0)

    return {
        "bid": round(B_opt, 2),
        "ask": round(A_opt, 2),
        "spread": round(spread, 2),
        "utilidad": round(utilidad, 2),
    }

def analisis_sensibilidad(
    valores_pi_I=(0.1, 0.4, 0.7),
    S0=S0
):
    """Repite la optimizacion al variar la proporcion informada.

    Como cada trader es informado o de liquidez,
    se usa pi_L = 1 - pi_I.

    El objetivo es verificar numericamente si un mayor riesgo
    de seleccion adversa lleva a un spread optimo mayor.
    """

    resultados = []

    for pi_I in valores_pi_I:

        pi_L = 1 - pi_I

        optimo = optimizar_cotizaciones(
            pi_I=pi_I,
            pi_L=pi_L,
            S0=S0
        )

        resultados.append({
            "pi_I": pi_I,
            "pi_L": pi_L,
            **optimo
        })

    return resultados