"""Figuras del Laboratorio 01 de Microestructura de Mercado.

Este modulo recibe resultados calculados por model.py y simulation.py.
No vuelve a implementar el modelo ni el simulador.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.model import (
    COEF_INTERCEPTO_LIQUIDEZ,
    COEF_PENDIENTE_LIQUIDEZ,
    prob_ejecucion_liquidez,
)


def figura_probabilidad_ejecucion():
    """Grafica la probabilidad de ejecucion y marca donde llega a cero."""

    s_cero = (
        COEF_INTERCEPTO_LIQUIDEZ
        / COEF_PENDIENTE_LIQUIDEZ
    )

    s = np.linspace(0, 8, 200)

    probabilidad = prob_ejecucion_liquidez(s)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        s,
        probabilidad,
        label="Probabilidad de ejecucion"
    )

    ax.scatter(
        [s_cero],
        [0],
        s=70,
        label=f"Probabilidad cero: s={s_cero:.2f}"
    )

    ax.set_title(
        "Probabilidad de ejecucion vs. distancia de cotizacion"
    )

    ax.set_xlabel(
        "s (distancia desde S0 por lado)"
    )

    ax.set_ylabel(
        "Probabilidad de ejecucion"
    )

    ax.set_ylim(-0.02, 0.52)

    ax.grid(alpha=0.3)

    ax.legend()

    fig.tight_layout()

    return fig


def figura_pnl_acumulado(trades_por_regimen):
    """Compara el P&L acumulado de los tres regimenes."""

    fig, ax = plt.subplots(figsize=(9, 5))

    for nombre, trades in trades_por_regimen.items():

        pnl_acumulado = trades["pnl"].cumsum()

        ax.plot(
            range(1, len(trades) + 1),
            pnl_acumulado,
            label=nombre
        )

    ax.set_title(
        "P&L acumulado por regimen"
    )

    ax.set_xlabel(
        "Numero de trade"
    )

    ax.set_ylabel(
        "P&L acumulado"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    fig.tight_layout()

    return fig


def figura_inventario_acumulado(trades_por_regimen):
    """Compara el inventario acumulado de los tres regimenes."""

    fig, ax = plt.subplots(figsize=(9, 5))

    for nombre, trades in trades_por_regimen.items():

        inventario = trades[
            "cambio_inventario"
        ].cumsum()

        ax.plot(
            range(1, len(trades) + 1),
            inventario,
            label=nombre
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        "Inventario acumulado por regimen"
    )

    ax.set_xlabel(
        "Numero de trade"
    )

    ax.set_ylabel(
        "Inventario acumulado"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    fig.tight_layout()

    return fig


def figura_histograma_monte_carlo(
    pnl_finales_por_regimen
):
    """Grafica la distribucion del P&L final del Monte Carlo."""

    fig, ax = plt.subplots(figsize=(9, 5))

    for nombre, pnl_finales in pnl_finales_por_regimen.items():

        ax.hist(
            pnl_finales,
            bins=30,
            alpha=0.45,
            label=nombre
        )

    ax.set_title(
        "Distribucion del P&L final - Monte Carlo"
    )

    ax.set_xlabel(
        "P&L final de una corrida"
    )

    ax.set_ylabel(
        "Frecuencia"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    fig.tight_layout()

    return fig


def figura_sensibilidad(sensibilidad):
    """Grafica el spread optimo para distintos valores de pi_I."""

    pi_I = [
        fila["pi_I"]
        for fila in sensibilidad
    ]

    spreads = [
        fila["spread"]
        for fila in sensibilidad
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        pi_I,
        spreads,
        marker="o",
        label="Spread optimo"
    )

    for x, y in zip(pi_I, spreads):

        ax.annotate(
            f"{y:.2f}",
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )

    ax.set_title(
        "Sensibilidad del spread optimo a la proporcion informada"
    )

    ax.set_xlabel(
        "Probabilidad de trader informado (pi_I)"
    )

    ax.set_ylabel(
        "Spread optimo"
    )

    ax.set_xticks(pi_I)

    ax.grid(alpha=0.3)

    ax.legend()

    fig.tight_layout()

    return fig