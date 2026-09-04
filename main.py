import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

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

DIR_RAIZ = Path(__file__).resolve().parent
RUTA_NOTEBOOK = DIR_RAIZ / "notebooks" / "analysis.ipynb"

CODIGO_SETUP = """\
%matplotlib inline

from src.model import PI_I, PI_L, S0, analisis_sensibilidad, optimizar_cotizaciones
from src.simulation import construir_regimenes, simular_monte_carlo, simular_trades
from src.plots import (
    figura_histograma_monte_carlo,
    figura_inventario_acumulado,
    figura_pnl_acumulado,
    figura_probabilidad_ejecucion,
    figura_sensibilidad,
)

optimo = optimizar_cotizaciones(PI_I, PI_L, S0)
regimenes = construir_regimenes()
trades_por_regimen = {
    nombre: simular_trades(bid, ask, n_trades=10000, seed=42)
    for nombre, (bid, ask) in regimenes.items()
}
resumen_mc, pnl_finales = simular_monte_carlo(
    regimenes, n_corridas=1000, n_trades=1000, seed=42, devolver_pnl_finales=True
)
sensibilidad = analisis_sensibilidad()

print({clave: float(valor) for clave, valor in optimo.items()})
"""

# (interpretacion markdown, codigo de la celda de figura) por cada una de las
# 5 figuras del proyecto. Las interpretaciones condensan (1-3 lineas) las
# mismas 5 preguntas de analisis ya redactadas en README.md -- no son
# redaccion nueva, son un resumen de lo que ya existe ahi.
FIGURAS_NOTEBOOK = [
    (
        "**Probabilidad de ejecucion vs. spread.** Cae linealmente y llega a "
        "cero en s=6.25 (marcado en la grafica). Mas alla de ese punto, la "
        "simulacion solo sigue generando trades porque los fuerza via el "
        "desempate contra S0, no porque exista demanda real de un trader de "
        "liquidez -- ver la advertencia de interpretacion del proyecto.",
        "figura_probabilidad_ejecucion();",
    ),
    (
        "**P&L acumulado por regimen (Pregunta 1).** Con el regimen Estrecho "
        "(spread=0.30) el formador casi no cobra por proveer liquidez pero "
        "sigue expuesto por completo a los informados: el P&L acumulado de "
        "10,000 trades termina en -6,873.10, mostrando por que un spread "
        "demasiado angosto no compensa la seleccion adversa.",
        "figura_pnl_acumulado(trades_por_regimen);",
    ),
    (
        "**Inventario acumulado por regimen (Pregunta 3).** El inventario "
        "final tras 10,000 trades es +34 (Optimo), -38 (Estrecho) y -38 "
        "(Amplio). La componente de informados es identica en los tres "
        "regimenes (-28); la diferencia viene del sorteo de liquidez, 50/50 "
        "en Estrecho y Amplio (simetricos alrededor de S0) y sesgado en "
        "Optimo (ligeramente asimetrico).",
        "figura_inventario_acumulado(trades_por_regimen);",
    ),
    (
        "**Distribucion del P&L final -- Monte Carlo (Preguntas 1 y 2).** "
        "Estrecho pierde en el 100% de las 1,000 corridas (prob_perdida=1.0). "
        "Ademas, el PnL medido del lado informado se infla artificialmente "
        "en spreads anchos: en Amplio, el promedio \"con todo\" es -0.58 pero "
        "cae a -1.65 filtrando solo a trades con ventaja genuina -- 44.2% de "
        "los \"informados\" en ese regimen fueron forzados sin ventaja real.",
        "figura_histograma_monte_carlo(pnl_finales);",
    ),
    (
        "**Spread optimo vs. pi_I (Pregunta 4).** Crece monotonamente: 6.40 "
        "(pi_I=0.1) -> 6.98 (pi_I=0.4) -> 7.99 (pi_I=0.7), consistente con la "
        "teoria de seleccion adversa -- a mayor proporcion de informados, "
        "mayor el spread que compensa el riesgo.",
        "figura_sensibilidad(sensibilidad);",
    ),
]


def construir_notebook_analisis():
    """Genera y ejecuta notebooks/analysis.ipynb desde cero, sobrescribiendo
    el anterior.

    El notebook no se edita a mano (regla dura del Lab): se arma en memoria
    con nbformat cada vez que corre main.py, nunca se abre el .ipynb previo
    para agregarle celdas, por lo que dos corridas seguidas producen el mismo
    resultado limpio sin celdas duplicadas. Se ejecuta con nbclient (no
    nbconvert/subprocess) para que las figuras queden embebidas como imagenes
    en el output, no solo el codigo que las generaria.

    El kernel se registra con prefix=sys.prefix (dentro del propio venv, en
    vez de --user) para que no quede un kernelspec "python3" permanente en
    el sistema del usuario que tape el que ya existiera antes del proyecto --
    el registro vive y muere con el venv.
    """
    import ipykernel.kernelspec

    ipykernel.kernelspec.install(prefix=sys.prefix, kernel_name="python3")

    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }

    celdas = [
        new_markdown_cell(
            "# Lab01 -- Cotizaciones Optimas de un Formador de Mercado\n\n"
            "Notebook generado automaticamente por `main.py` (no editado a "
            "mano). Reproduce las 5 figuras del proyecto junto con una breve "
            "interpretacion de cada una, condensada de las 5 preguntas de "
            "analisis de README.md. Seed=42 en toda la simulacion."
        ),
        new_code_cell(CODIGO_SETUP),
    ]
    for interpretacion, codigo_figura in FIGURAS_NOTEBOOK:
        celdas.append(new_markdown_cell(interpretacion))
        celdas.append(new_code_cell(codigo_figura))
    nb.cells = celdas

    client = NotebookClient(
        nb,
        kernel_name="python3",
        timeout=600,
        resources={"metadata": {"path": str(DIR_RAIZ)}},
    )
    client.execute()

    RUTA_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, RUTA_NOTEBOOK)
    print(f"\nNotebook generado y ejecutado: {RUTA_NOTEBOOK}")

# Paso 1: Cotizaciones optimas

def main():

    print("\n--- Paso 1: Cotizaciones optimas ---")

    optimo = optimizar_cotizaciones(
        PI_I,
        PI_L,
        S0
    )

    print({clave: float(valor) for clave, valor in optimo.items()})

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

# Paso 5: Notebook generado automaticamente

    print("\n--- Paso 5: Generando notebooks/analysis.ipynb ---")

    construir_notebook_analisis()


if __name__ == "__main__":
    main()