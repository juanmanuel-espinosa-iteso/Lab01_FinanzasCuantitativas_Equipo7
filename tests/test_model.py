import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import optimizar_cotizaciones, prob_ejecucion_liquidez, utilidad_esperada


def test_optimizar_cotizaciones_pi_I_cero_spread_analitico():
    """Con pi_I=0, Pi(A,B) se reduce a maximizar h(s)=(0.50-0.08s)*s por lado.

    h'(s) = 0.50 - 0.16s = 0 => s* = 0.50/0.16 = 3.125, spread = 2*s* = 6.25.

    El enunciado del Paso 1 (Paso01_Modelo_CLAUDE.md) afirma que el spread
    analitico es 2*(0.50/0.08) = 12.50, pero eso corresponde al punto donde
    pi_LB(s) se trunca en 0 (borde del dominio), no al maximizador de h(s):
    le falta el factor 2 que aparece al derivar el termino cuadratico
    -0.08*s^2. Se usa aqui el valor correctamente derivado (6.25); pendiente
    de confirmar con el profesor segun lo acordado con el usuario.
    """
    resultado = optimizar_cotizaciones(pi_I=0, pi_L=1, S0=19.90)

    assert abs(resultado["spread"] - 6.25) < 1e-3


def test_prob_ejecucion_liquidez_nunca_negativa():
    assert prob_ejecucion_liquidez(10) == 0.0
    assert prob_ejecucion_liquidez(6.25) == 0.0
    assert prob_ejecucion_liquidez(0) == 0.50


def test_optimizar_cotizaciones_respeta_bounds():
    S0 = 19.90
    for pi_I in (0.1, 0.4, 0.7):
        resultado = optimizar_cotizaciones(pi_I=pi_I, pi_L=1 - pi_I, S0=S0)

        assert 0 < resultado["bid"] <= S0
        assert resultado["ask"] >= S0


def test_optimizar_cotizaciones_redondea_a_dos_decimales():
    resultado = optimizar_cotizaciones(pi_I=0.40, pi_L=0.60, S0=19.90)

    for clave in ("bid", "ask", "spread", "utilidad"):
        valor = resultado[clave]
        assert round(valor, 2) == valor


def test_utilidad_esperada_decrece_al_alejar_cotizaciones_del_optimo():
    """En el caso base (pi_I=0.40), ensanchar el spread mas alla del optimo
    debe reducir la utilidad esperada: confirma que el termino de perdida
    por seleccion adversa (integrales via quad) efectivamente penaliza
    cotizaciones demasiado agresivas.
    """
    S0 = 19.90
    optimo = optimizar_cotizaciones(pi_I=0.40, pi_L=0.60, S0=S0)

    utilidad_en_optimo = utilidad_esperada(
        optimo["ask"], optimo["bid"], pi_I=0.40, pi_L=0.60, S0=S0
    )
    utilidad_spread_amplio = utilidad_esperada(
        S0 + 5, S0 - 5, pi_I=0.40, pi_L=0.60, S0=S0
    )

    assert utilidad_en_optimo > utilidad_spread_amplio
