"""Simulador de trades y analisis de Monte Carlo para el formador de mercado.

src/model.py resuelve Pi(A,B) en forma cerrada (utilidad ESPERADA por trade).
Este modulo simula trades individuales para estimar cantidades que la forma
cerrada no entrega directamente: la distribucion completa del P&L, su
desviacion estandar, y la probabilidad de que una corrida termine en perdida.

Advertencia de interpretacion (ver Lab01/CLAUDE.md): cada llegada simulada
fuerza la ejecucion de un trade, incluso en dos casos donde en la realidad
podria no haberlo:
  - Trader de liquidez: pi_LB/pi_LS son probabilidades de aceptacion, pero
    aqui solo se usan para decidir el LADO (compra o venta), nunca para
    decidir si el trade ocurre -- siempre ocurre.
  - Trader informado sin ventaja real (B<=P<=A): en la realidad no operaria,
    pero aqui se le fuerza un lado via desempate contra S0 (ver columna
    `ventaja_genuina`). Como un spread mas ancho deja mas masa de la Erlang
    dentro de [B,A], estos trades forzados (que son SIEMPRE una ganancia
    para el formador, nunca una perdida, por construccion) se vuelven mas
    frecuentes mientras mas ancho es el spread -- esto infla el P&L medido
    del regimen Amplio por una razon que no es seleccion adversa real, y
    debe filtrarse con `ventaja_genuina` antes de comparar contra la
    integral de perdida de Paso 1.
Los resultados son entonces rentabilidad POR TRADE FORZADO, no por unidad de
tiempo real de mercado.
"""

import numpy as np
import pandas as pd
from scipy.stats import erlang

from src.model import ERLANG_K, ERLANG_LAMBDA, PI_I, PI_L, S0, optimizar_cotizaciones, prob_ejecucion_liquidez

REGIMEN_ESTRECHO = (19.75, 20.05)
REGIMEN_AMPLIO = (18.40, 21.40)

_DISTRIBUCION_VALOR = erlang(a=ERLANG_K, scale=1 / ERLANG_LAMBDA)


def _sortear_compra_liquidez(bid, ask, rng, n):
    """Decide, para n traders de liquidez, si compran (True) o venden (False).

    pi_LB(A-S0) y pi_LS(S0-B) no tienen por que ser iguales -- el optimo de
    Paso 1 no cae exactamente simetrico alrededor de S0 (la media de la
    Erlang es k/lambda=20, no S0=19.90) -- por eso se normalizan a una
    probabilidad de compra real en vez de asumir una moneda 50/50 a la
    fuerza; en un regimen donde ambos lados sean simetricos esto se reduce
    naturalmente a 50/50 porque asi lo da la normalizacion, no porque se
    haya asumido de antemano. Si ambos pesos cayeran en 0 (spread mas alla
    de s=6.25 en ambos lados), se usa un fallback 50/50 para seguir
    garantizando que siempre se ejecuta un trade.
    """
    peso_compra = prob_ejecucion_liquidez(ask - S0)
    peso_venta = prob_ejecucion_liquidez(S0 - bid)
    total = peso_compra + peso_venta
    prob_compra = peso_compra / total if total > 0 else 0.5
    return rng.random(n) < prob_compra


def simular_trades(bid, ask, n_trades=10000, seed=42):
    """Simula n_trades llegadas de traders bajo un regimen fijo (bid, ask).

    Cada llegada es informada con prob. PI_I o de liquidez con prob. PI_L
    (constantes de src/model.py, fuente unica de verdad). El P&L se calcula
    desde la perspectiva del formador de mercado:
      - Informado: conoce el valor verdadero P ~ Erlang(k,lambda) -- la misma
        distribucion que el formador cree, dibujada de nuevo en cada trade.
        Si P>A compra (formador pierde P-A); si P<B vende (formador pierde
        B-P). Si B<=P<=A no tiene ventaja real (`ventaja_genuina=False`),
        pero de todas formas se fuerza un lado comparando P contra S0 --
        esta regla coincide exactamente con la decision racional cuando
        P>A o P<B (porque A>=S0 y B<=S0), y es su unica extension natural
        dentro de la zona sin ventaja.
      - Liquidez: no conoce P, por eso su P&L no depende de P sino del
        spread capturado frente a S0 (A-S0 si compra, S0-B si vende) --
        exactamente el monto de la formula Pi(A,B) de Paso 1, no un
        promedio. `ventaja_genuina` es False para estas filas: un trader
        sin informacion nunca tiene ventaja informativa, por definicion.

    cambio_inventario usa la convencion estandar de microestructura: +1
    cuando el formador COMPRA (el trader vende, "hits the bid"), -1 cuando
    el formador VENDE (el trader compra, "lifts the ask") -- es el inventario
    propio del formador, no el del trader.
    """
    rng = np.random.default_rng(seed)

    es_informado = rng.random(n_trades) < PI_I
    valores_verdaderos = _DISTRIBUCION_VALOR.rvs(size=n_trades, random_state=rng)
    compra_liquidez = _sortear_compra_liquidez(bid, ask, rng, n_trades)

    compra_informado = valores_verdaderos >= S0
    ventaja_genuina_informado = (valores_verdaderos > ask) | (valores_verdaderos < bid)

    compra = np.where(es_informado, compra_informado, compra_liquidez)
    pnl_informado = np.where(
        compra_informado, ask - valores_verdaderos, valores_verdaderos - bid
    )
    pnl_liquidez = np.where(compra_liquidez, ask - S0, S0 - bid)

    return pd.DataFrame(
        {
            "pnl": np.where(es_informado, pnl_informado, pnl_liquidez),
            "cambio_inventario": np.where(compra, -1, 1),
            "tipo_trader": np.where(es_informado, "informado", "liquidez"),
            "lado": np.where(compra, "compra", "venta"),
            "ventaja_genuina": np.where(es_informado, ventaja_genuina_informado, False),
        }
    )


def construir_regimenes(pi_I=PI_I, pi_L=PI_L, S0=S0):
    """Arma los 3 regimenes de cotizacion del Paso 2.

    El regimen Optimo se obtiene llamando a optimizar_cotizaciones (Paso 1)
    en cada llamada -- nunca un numero fijo -- para que el analisis de
    sensibilidad de Paso 4 (que varia pi_I) pueda reusar esta funcion.
    Estrecho y Amplio si son valores fijos del enunciado (no se optimizan).
    """
    optimo = optimizar_cotizaciones(pi_I, pi_L, S0)
    return {
        "Optimo": (optimo["bid"], optimo["ask"]),
        "Estrecho": REGIMEN_ESTRECHO,
        "Amplio": REGIMEN_AMPLIO,
    }


def simular_monte_carlo(
    regimenes,
    n_corridas=1000,
    n_trades=1000,
    seed=42,
    devolver_pnl_finales=False
):
    """Corre simulaciones Monte Carlo para cada regimen.

    Si devolver_pnl_finales=True, tambien devuelve los P&L finales
    de cada corrida para poder construir el histograma obligatorio.
    """

    resultados = {}

    pnl_finales_por_regimen = {}

    for nombre_regimen, (bid, ask) in regimenes.items():

        pnl_por_corrida = np.empty(n_corridas)

        for i in range(n_corridas):

            trades = simular_trades(
                bid,
                ask,
                n_trades=n_trades,
                seed=seed + i
            )

            pnl_por_corrida[i] = trades["pnl"].sum()

        pnl_finales_por_regimen[
            nombre_regimen
        ] = pnl_por_corrida.copy()

        resultados[nombre_regimen] = {

            "pnl_promedio": round(
                float(pnl_por_corrida.mean()),
                2
            ),

            "pnl_std": round(
                float(
                    pnl_por_corrida.std(ddof=1)
                ),
                2
            ),

            "prob_perdida": round(
                float(
                    (pnl_por_corrida < 0).mean()
                ),
                4
            ),
        }

    if devolver_pnl_finales:

        return (
            resultados,
            pnl_finales_por_regimen
        )

    return resultados
