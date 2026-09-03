# Lab01_FinanzasCuantitativas_Equipo7

## Integrantes

- Juan Manuel Espinosa Cárdenas
- Jerónimo Rojas Alvarado

## Descripción del proyecto

Este proyecto modela y simula el problema de un formador de mercado (market
maker) que debe cotizar un bid y un ask alrededor de un valor de referencia
S0=19.90 para un activo cuyo valor verdadero sigue una distribución
Erlang(k=60, lambda=3). El formador enfrenta un tradeoff clásico de
microestructura: cotizar lejos de S0 aumenta el spread que cobra a los
traders de liquidez, pero también aumenta la pérdida esperada frente a
traders informados que conocen el valor verdadero del activo (selección
adversa). El proyecto resuelve las cotizaciones óptimas en forma cerrada
(`src/model.py`), simula trades individuales y corridas de Monte Carlo para
estimar la distribución completa del P&L (`src/simulation.py`), genera las
figuras obligatorias del análisis (`src/plots.py`), y corre un análisis de
sensibilidad de la proporción de traders informados (integrado en
`src/model.py` y orquestado desde `main.py`).

## Instalación

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
```

## Reproducción de resultados

```bash
python main.py
```

Este comando corre **todo** el pipeline en un solo paso: la optimización de
Paso 1, la simulación de 10,000 trades y el Monte Carlo de 1,000 corridas x
1,000 trades de Paso 2, las 4 figuras obligatorias de Paso 3 más la figura
de sensibilidad de Paso 4, y el análisis de sensibilidad de pi_I. Las 5
figuras se guardan como PNG en `docs/figuras/` (además de mostrarse en
pantalla con `plt.show()`) y todos los resultados numéricos se imprimen en
consola.

## Seed

Todas las simulaciones usan **seed=42** como semilla base
(`np.random.default_rng(42)`), fijada en `src/simulation.py`. El Monte Carlo
de 1,000 corridas no reusa esa seed globalmente: cada corrida `i` usa su
propia seed derivada `seed + i`, para que las 1,000 corridas sean
realizaciones independientes en vez de fragmentos de un mismo stream de
aleatoriedad (ver docstring de `simular_monte_carlo` en
`src/simulation.py`).

## Decisiones de diseño no evidentes desde el enunciado

- La función de análisis de sensibilidad (`analisis_sensibilidad`) vive en
  `src/model.py`, junto a `optimizar_cotizaciones`, en vez de un archivo
  separado — el enunciado no exige un archivo propio para esto y la
  estructura obligatoria del repo no lo contempla.
- Las 5 figuras (las 4 obligatorias de Paso 3 más la de sensibilidad de
  Paso 4) se guardan como PNG en `docs/figuras/` cada vez que se generan,
  además de mostrarse en pantalla.

## Advertencia de interpretación

**La simulación fuerza la ejecución de un trade en cada iteración**, tanto
para el trader de liquidez (cuyas probabilidades `pi_LB`/`pi_LS` en la
realidad podrían no resultar en una ejecución) como para el trader
informado sin ventaja real (`bid <= P <= ask`, forzado a tomar un lado por
desempate contra S0 — ver columna `ventaja_genuina` en
`simular_trades`). **Los resultados representan rentabilidad POR TRADE, no
por unidad de tiempo real de mercado.** Un spread muy amplio que en la
realidad casi nunca se ejecutaría queda artificialmente favorecido bajo
esta métrica — ver la pregunta de análisis 2 más abajo, donde esto se
cuantifica con cifras concretas en vez de solo mencionarse.

## Preguntas de análisis

### 1. ¿Por qué los traders informados generan la necesidad de un spread? (cifras del régimen Estrecho)

Con el régimen Estrecho (bid=19.75, ask=20.05, spread=0.30) el formador casi
no cobra nada por proveer liquidez, pero sigue expuesto por completo a los
traders informados. El PnL promedio por trade informado en este régimen es
**-1.93** (contando todos los trades informados), y la simulación completa
de 10,000 trades da un **P&L total de -6,873.10**. En el Monte Carlo de
1,000 corridas x 1,000 trades, el régimen Estrecho pierde en el **100% de
las corridas** (`prob_perdida=1.0`, `pnl_promedio=-671.35` por corrida). Sin
un spread que compense esa pérdida esperada frente al informado, el
formador pierde dinero de forma sistemática y no ocasional — de ahí la
necesidad de un spread positivo.

### 2. ¿Cómo cambia el costo de selección adversa conforme se amplía el spread? (mostrado, no afirmado)

Usando la columna `ventaja_genuina` de `simular_trades` (True si el
informado tenía ventaja real, P>ask o P<bid; False si fue forzado por
desempate porque bid<=P<=ask), y ordenando los 3 regímenes por ancho de
spread:

| Régimen | Spread total | % trades "informados" sin ventaja real (forzados) | PnL promedio informado (TODO) | PnL promedio informado (solo `ventaja_genuina=True`) |
|---|---|---|---|---|
| Estrecho | 0.30 | 4.7% | -1.93 | -2.03 |
| Amplio | 3.00 | 44.2% | -0.58 | -1.65 |
| Óptimo | 6.98 | 82.3% | +1.41 | -1.26 |

Dos cosas cambian con el spread, en direcciones opuestas:

- El costo de selección adversa **real** (columna de la derecha, condicionado
  a que el informado sí tuviera ventaja) se reduce monótonamente conforme el
  spread crece: -2.03 → -1.65 → -1.26. Esto es consistente con la teoría de
  Paso 1: un spread más ancho reduce la probabilidad de que el informado
  tenga ventaja, y por lo tanto la pérdida esperada por trade.
- Pero la métrica "ingenua" (PnL promedio contando TODOS los trades
  informados, sin filtrar) se aleja cada vez más de ese número real: la
  brecha entre ambas columnas crece de 0.10 (Estrecho) a 1.07 (Amplio) a
  2.67 (Óptimo). Esto ocurre porque el spread más ancho deja más masa de la
  Erlang dentro de la zona `[bid, ask]` sin ventaja real, y esos trades
  forzados son **siempre una ganancia** para el formador por construcción
  del desempate (no una pérdida) — inflando artificialmente el PnL medido
  del lado informado en los regímenes de spread ancho. Esta es exactamente
  la advertencia de interpretación de la sección anterior, cuantificada.

### 3. ¿Qué régimen acumula el mayor desbalance de inventario y por qué? ¿A qué riesgo real lo expone eso, que el modelo no captura?

En la simulación de 10,000 trades (seed=42), el inventario final acumulado
(`cambio_inventario.cumsum()`) es: **Óptimo +34, Estrecho -38, Amplio -38**.
Estrecho y Amplio empatan en el mayor desbalance absoluto (38 unidades).

La razón es rastreable en el código: la componente de inventario que viene
de los traders informados es **idéntica en los tres regímenes (-28)**,
porque la decisión de lado del informado (`P >= S0`) no depende de bid/ask,
solo de S0 y de la misma secuencia de valores `P` (misma seed). La
diferencia viene de la componente de traders de liquidez: Estrecho y Amplio
son ambos simétricos alrededor de S0 (por construcción del enunciado:
19.75/20.05 y 18.40/21.40), lo que hace que el sorteo ponderado de
compra/venta de liquidez sea exactamente 50/50 en los dos — y al compartir
la misma seed, terminan con la misma mezcla exacta de compras y ventas de
liquidez (-10 en ambos). El régimen Óptimo, en cambio, es ligeramente
asimétrico (ask-S0=3.53 vs S0-bid=3.45, porque la media de la Erlang es
20≠S0=19.90), lo que sesga el sorteo hacia compra y deja un inventario neto
positivo (+62 del lado liquidez, +34 total).

Riesgo real que el modelo **no captura**: el modelo optimiza la utilidad
esperada de un solo trade aislado, sin ningún costo por cargar inventario
en el tiempo. Un formador con -38 unidades de inventario neto está
direccionalmente expuesto: si el precio de mercado se mueve en su contra
mientras sostiene esa posición, pierde dinero por el movimiento del
subyacente (riesgo de mercado), no por selección adversa — y este modelo,
al evaluar cada trade de forma independiente sin actualizar S0 ni penalizar
el tamaño del inventario, no tiene forma de representar ese riesgo ni el
costo de capital o de cobertura que implicaría en la realidad.

### 4. ¿Cómo se comporta el spread óptimo al variar pi_I? ¿Coincide con la teoría?

| pi_I | pi_L | Spread óptimo |
|---|---|---|
| 0.1 | 0.9 | 6.40 |
| 0.4 | 0.6 | 6.98 |
| 0.7 | 0.3 | 7.99 |

El spread óptimo crece de forma monótona con pi_I (6.40 → 6.98 → 7.99). Esto
coincide con la predicción teórica de selección adversa vista en clase: a
mayor proporción de traders informados, mayor es la pérdida esperada por
trade frente a ellos, y el formador óptimo compensa ese riesgo ensanchando
el spread — nunca reduciéndolo, y la magnitud del ensanchamiento entre
pi_I=0.1 y pi_I=0.7 (+1.59) es comparable al que se pierde de utilidad
esperada por trade (1.38 → 0.34, ver `analisis_sensibilidad` en
`src/model.py`), consistente con que ambos efectos vienen de la misma
fuente (mayor pi_I).

### 5. Tres limitaciones de este modelo para un formador de mercado real

1. **Ejecución forzada:** cada iteración fuerza un trade (tanto para
   liquidez como, en la zona sin ventaja, para el informado), lo cual
   infla artificialmente regímenes de spread amplio que en la realidad casi
   nunca se ejecutarían — ver advertencia de interpretación y pregunta 2.
2. **Sin costo de inventario ni de tiempo:** el modelo optimiza la utilidad
   esperada de un trade aislado y no penaliza cargar inventario, ni
   considera el paso del tiempo entre trades, capital requerido, ni riesgo
   de que el precio se mueva mientras se sostiene una posición (ver
   pregunta 3).
3. **Sin aprendizaje ni actualización de creencias:** S0 permanece fijo en
   19.90 durante toda la simulación; un formador real actualiza su
   estimación del valor justo con cada trade que observa (especialmente
   tras ejecutar contra un informado), y este modelo no tiene ningún
   mecanismo para eso — cada trade se trata como independiente e idéntico.

## Uso de IA

Se usó Claude (Anthropic) como asistente de código a lo largo del proyecto:

- **Paso 1 (`src/model.py`):** diseño e implementación de
  `optimizar_cotizaciones`, discusión y justificación explícita de la
  elección de `L-BFGS-B` sobre `SLSQP` y de pasar `np.inf` directo a
  `scipy.integrate.quad` en vez de truncar manualmente; se identificó y
  corrigió un error de factor 2 en el valor analítico de referencia del
  checklist original del profesor (documentado en `tests/test_model.py`).
- **Paso 2 (`src/simulation.py`):** diseño de la convención de signo de
  `cambio_inventario`, del mecanismo de sorteo ponderado para el lado del
  trader de liquidez, y de la regla de desempate del trader informado sin
  ventaja real (columna `ventaja_genuina`), incluyendo una comparación
  explícita de alternativas (desempate vs. resampleo por rechazo) antes de
  decidir.
- **Paso 3 y 4:** auditoría del código ya implementado contra los
  checklists de ambas piezas, corrección de `requirements.txt` (faltaba
  `matplotlib`), adición de guardado a disco de las 5 figuras en
  `docs/figuras/`, y redacción de este README (la descripción del proyecto,
  instalación, y las 5 preguntas de análisis con las cifras obtenidas de la
  simulación real del proyecto).

Todo el código generado con asistencia de IA fue revisado y ejecutado por
el equipo antes de integrarse.
