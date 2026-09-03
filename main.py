import pandas as pd
import matplotlib.pyplot as plt

from src.model import (
    PI_I,
    PI_L,
    S0,
    analisis_sensibilidad,
    optimizar_cotizaciones,
)

from src.simulation import (
    construir_regimenes,
    simular_monte_carlo,
    simular_trades,
)

from src.plots import (
    figura_histograma_monte_carlo,
    figura_inventario_acumulado,
    figura_pnl_acumulado,
    figura_probabilidad_ejecucion,
    figura_sensibilidad,
)

# Paso 1: Cotizaciones optimas

def main():

    print("\n--- Paso 1: Cotizaciones optimas ---")

    optimo = optimizar_cotizaciones(
        PI_I,
        PI_L,
        S0
    )

    print(optimo)

# Paso 2: Simulacion de 10,000 trades

    print("\n--- Paso 2: Simulacion de 10,000 trades ---")

    regimenes = construir_regimenes()

    trades_por_regimen = {}

    for nombre, (bid, ask) in regimenes.items():

        trades = simular_trades(
            bid,
            ask,
            n_trades=10000,
            seed=42
        )

        trades_por_regimen[nombre] = trades

        print(
            f"{nombre}: "
            f"P&L={trades['pnl'].sum():.2f}, "
            f"inventario final="
            f"{trades['cambio_inventario'].sum():.0f}"
        )

# Monte Carlo

    print(
        "\n--- Monte Carlo: "
        "1,000 corridas x 1,000 trades ---"
    )

    resumen_mc, pnl_finales = simular_monte_carlo(
        regimenes,
        n_corridas=1000,
        n_trades=1000,
        seed=42,
        devolver_pnl_finales=True
    )

    print(
        pd.DataFrame(resumen_mc).T
    )

# Sensibilidad

    print("\n--- Paso 4: Sensibilidad ---")

    sensibilidad = analisis_sensibilidad()

    print(
        pd.DataFrame(sensibilidad)
    )

# Figuras

    figura_probabilidad_ejecucion()

    figura_pnl_acumulado(
        trades_por_regimen
    )

    figura_inventario_acumulado(
        trades_por_regimen
    )

    figura_histograma_monte_carlo(
        pnl_finales
    )

    figura_sensibilidad(
        sensibilidad
    )

    plt.show()


if __name__ == "__main__":
    main()