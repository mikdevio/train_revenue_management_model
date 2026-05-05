# Definición y carga de la Red Neuronal
"""
MODELO DETERMINISTA DE REVENUE MANAGEMENT PARA TRENES UNITARIOS
===============================================================

Implementación completa del modelo de trenes unitarios con:
- Seguimiento de posición de trenes en la red
- Capacidad de acoplamiento de dos trenes del mismo tipo
- Asignación dinámica de precios por clase y tiempo de anticipación
- Demanda determinista basada en datos históricos

Soportes:
- Solvers comerciales: Gurobi, CPLEX, Gurobi Direct
- Solvers open-source: CBC (para problemas pequeños), GLPK

Autor: Basado en la adaptación del modelo de Kamandanipour et al. (2023)
"""

import pyomo.environ as pyo
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# CLASES AUXILIARES Y ESTRUCTURAS DE DATOS
# ============================================================================


class TipoTrenEnum(Enum):
    """Tipos de tren disponibles"""

    PREMIUM = 0
    LOWCOST = 1
    STANDARD = 2


@dataclass
class TipoTren:
    """Define un tipo de tren fabricado como unidad completa"""

    nombre: str
    capacidad: int  # Asientos por tren
    costo_fijo: float  # Coste por hacer salir el tren
    costo_variable_base: float  # Coste base por pasajero
    multiplicador_clase: List[float] = field(default_factory=lambda: [1.5, 1.0, 0.7])


@dataclass
class ServicioTren:
    """Define un servicio programado (número de tren en el horario)"""

    id: int
    origen: int  # Estación origen (1-indexado)
    destino: int  # Estación destino (1-indexado)
    duracion: int  # Días que dura el viaje (normalmente 1)
    horarios: List[int]  # Días en que opera este servicio


@dataclass
class DatosRedFerroviaria:
    """Contenedor completo de datos para el modelo determinista"""

    # ===== Dimensiones =====
    H: int  # Horizonte de planificación (días de salida)
    K: int  # Número de clases de servicio
    S: int  # Número de estaciones
    N: int  # Número de servicios programados
    R: int  # Número de trenes físicos disponibles
    O: int  # Número de opciones de precio
    SD: int  # Clases de estacionalidad
    TD: int  # Clases de tiempo-anticipación

    # ===== Trenes =====
    tipo_tren_por_r: List[int]  # Tipo de cada tren r (0,1,2...)
    tipos_tren: List[TipoTren]  # Definición de tipos disponibles

    # ===== Servicios =====
    servicios: List[ServicioTren]

    # ===== Parámetros operativos =====
    PL: np.ndarray  # (N, H): 1 si servicio n opera día D
    TR: np.ndarray  # (N, S, S): matriz de rutas (opcional, para compatibilidad)

    # ===== Posiciones iniciales =====
    pos_inicial: np.ndarray  # (R, S): 1 si tren r inicia en estación i

    # ===== Parámetros económicos =====
    precios: np.ndarray  # (K, N, O): precio opción o para clase k, servicio n
    CC: float  # Coste por acoplamiento
    OC: float  # Overhead
    AR: float  # Ingresos reales acumulados

    # ===== Estacionalidad y demanda =====
    SE: np.ndarray  # (H, N): clase de estacionalidad por día y servicio
    DE: np.ndarray  # (SD, TD, K, N, O): demanda determinista por segmento

    # ===== Ventas realizadas antes del planning =====
    RW: np.ndarray  # (H, K, N): trenes con reservas
    RS: np.ndarray  # (H, K, N): asientos reservados

    # ===== Parámetros de configuración =====
    MAX_COUPLE: int = 1  # Máximo acoplamientos (sí/no)
    EPSILON: float = 0.01  # Para desigualdad estricta
    BIG_M: float = 1e6  # Para restricciones big-M


# ============================================================================
# MODELO PRINCIPAL
# ============================================================================


class ModeloTrenesUnitariosDeterminista:
    """
    Modelo determinista de revenue management para trenes unitarios.

    Resuelve el problema de maximización de beneficio considerando:
    - Asignación de trenes a servicios
    - Posición geográfica de los trenes
    - Acoplamiento de trenes del mismo tipo
    - Selección dinámica de precios
    - Demanda determinista por segmento
    """

    def __init__(self, datos: DatosRedFerroviaria):
        """
        Inicializa el modelo con los datos de la red.

        Args:
            datos: Objeto DatosRedFerroviaria con todos los parámetros
        """
        self.datos = datos
        self.modelo = pyo.ConcreteModel()
        self._construir_modelo()

    def _construir_modelo(self):
        """Construye todas las variables, restricciones y función objetivo"""

        datos = self.datos
        m = self.modelo

        # ============================================================
        # 1. CONJUNTOS (Índices)
        # ============================================================

        m.DIAS = pyo.RangeSet(1, datos.H)  # Días de salida
        m.CLASES = pyo.RangeSet(1, datos.K)  # Clases de servicio
        m.ESTACIONES = pyo.RangeSet(1, datos.S)  # Estaciones
        m.SERVICIOS = pyo.RangeSet(1, datos.N)  # Servicios programados
        m.TRENES = pyo.RangeSet(1, datos.R)  # Trenes físicos
        m.OPCIONES = pyo.RangeSet(1, datos.O)  # Opciones de precio
        m.TIEMPO = pyo.RangeSet(1, datos.H)  # Días de anticipación (máx H)

        # ============================================================
        # 2. PARÁMETROS DEL MODELO
        # ============================================================

        # Capacidad del tren r (asientos)
        def _capacidad_tren_init(m, r):
            tipo_idx = datos.tipo_tren_por_r[r - 1]
            return datos.tipos_tren[tipo_idx].capacidad

        m.CAP_R = pyo.Param(m.TRENES, initialize=_capacidad_tren_init)

        # Coste fijo del tren r
        def _costo_fijo_init(m, r):
            tipo_idx = datos.tipo_tren_por_r[r - 1]
            return datos.tipos_tren[tipo_idx].costo_fijo

        m.FC_R = pyo.Param(m.TRENES, initialize=_costo_fijo_init)

        # Coste variable por clase k y tren r
        def _costo_variable_init(m, k, r):
            tipo_idx = datos.tipo_tren_por_r[r - 1]
            base = datos.tipos_tren[tipo_idx].costo_variable_base
            mult = datos.tipos_tren[tipo_idx].multiplicador_clase[k - 1]
            return base * mult

        m.VC_KR = pyo.Param(m.CLASES, m.TRENES, initialize=_costo_variable_init)

        # Precios por clase, servicio y opción
        def _precios_init(m, k, n, o):
            return datos.precios[k - 1, n - 1, o - 1]

        m.PO = pyo.Param(m.CLASES, m.SERVICIOS, m.OPCIONES, initialize=_precios_init)

        # Programación de servicios
        def _pl_init(m, n, D):
            return datos.PL[n - 1, D - 1]

        m.PL = pyo.Param(m.SERVICIOS, m.DIAS, initialize=_pl_init, mutable=True)

        # Estación origen por servicio
        def _origen_init(m, n):
            return datos.servicios[n - 1].origen

        m.ORIGEN_N = pyo.Param(m.SERVICIOS, initialize=_origen_init)

        # Estación destino por servicio
        def _destino_init(m, n):
            return datos.servicios[n - 1].destino

        m.DESTINO_N = pyo.Param(m.SERVICIOS, initialize=_destino_init)

        # Posición inicial de trenes
        def _pos_inicial_init(m, r, i):
            return datos.pos_inicial[r - 1, i - 1]

        m.POS_INICIAL = pyo.Param(m.TRENES, m.ESTACIONES, initialize=_pos_inicial_init)

        # Estacionalidad
        def _se_init(m, D, n):
            return datos.SE[D - 1, n - 1]

        m.SE = pyo.Param(m.DIAS, m.SERVICIOS, initialize=_se_init)

        # Demanda determinista
        def _demanda_init(m, D, t, k, n, o):
            sd = int(m.SE[D, n] - 1)  # convertir a 0-index
            td = min(t - 1, datos.TD - 1)
            if sd < 0 or sd >= datos.SD or td < 0:
                return 0.0
            return datos.DE[sd, td, k - 1, n - 1, o - 1]

        m.DE = pyo.Param(
            m.DIAS,
            m.TIEMPO,
            m.CLASES,
            m.SERVICIOS,
            m.OPCIONES,
            initialize=_demanda_init,
            default=0.0,
        )

        # Ventas reales acumuladas
        def _rw_init(m, D, k, n):
            return datos.RW[D - 1, k - 1, n - 1]

        m.RW = pyo.Param(m.DIAS, m.CLASES, m.SERVICIOS, initialize=_rw_init)

        def _rs_init(m, D, k, n):
            return datos.RS[D - 1, k - 1, n - 1]

        m.RS = pyo.Param(m.DIAS, m.CLASES, m.SERVICIOS, initialize=_rs_init)

        # Parámetros escalares
        m.CC = pyo.Param(initialize=datos.CC)
        m.OC = pyo.Param(initialize=datos.OC)
        m.AR = pyo.Param(initialize=datos.AR)
        m.EPSILON = pyo.Param(initialize=datos.EPSILON)
        m.BIG_M = pyo.Param(initialize=datos.BIG_M)

        # ============================================================
        # 3. VARIABLES DE DECISIÓN
        # ============================================================

        # (pos) Posición de trenes
        m.pos = pyo.Var(m.TRENES, m.ESTACIONES, m.DIAS, domain=pyo.Binary)

        # (u) Asignación de tren a servicio
        m.u = pyo.Var(m.TRENES, m.SERVICIOS, m.DIAS, domain=pyo.Binary)

        # (acoplado) Estado de acoplamiento
        m.acoplado = pyo.Var(m.TRENES, m.DIAS, domain=pyo.Binary)

        # (pareja) Relación de acoplamiento entre dos trenes
        m.pareja = pyo.Var(m.TRENES, m.TRENES, m.DIAS, domain=pyo.Binary)

        # (q) Selección de precio
        m.q = pyo.Var(
            m.CLASES, m.SERVICIOS, m.TIEMPO, m.OPCIONES, m.DIAS, domain=pyo.Binary
        )

        # (s) Ventas previstas
        m.s = pyo.Var(
            m.CLASES, m.SERVICIOS, m.TIEMPO, m.DIAS, domain=pyo.NonNegativeReals
        )

        # (c) Capacidad disponible por clase
        m.c = pyo.Var(m.CLASES, m.SERVICIOS, m.DIAS, domain=pyo.NonNegativeReals)

        # (cap_servicio) Capacidad total del servicio
        m.cap_servicio = pyo.Var(m.SERVICIOS, m.DIAS, domain=pyo.NonNegativeReals)

        # ============================================================
        # 4. RESTRICCIONES DE POSICIÓN Y MOVIMIENTO (P1-P11)
        # ============================================================

        # (P1) Posición inicial
        def _pos_inicial_rule(m, r, i):
            return m.pos[r, i, 1] == m.POS_INICIAL[r, i]

        m.PosInicial = pyo.Constraint(m.TRENES, m.ESTACIONES, rule=_pos_inicial_rule)

        # (P2) Unicidad de posición por día
        def _una_estacion_rule(m, r, D):
            return sum(m.pos[r, i, D] for i in m.ESTACIONES) == 1

        m.UnaEstacion = pyo.Constraint(m.TRENES, m.DIAS, rule=_una_estacion_rule)

        # (P3) Asignación requiere posición en origen
        def _asignacion_con_posicion_rule(m, r, n, D):
            return m.u[r, n, D] <= m.pos[r, m.ORIGEN_N[n], D]

        m.AsignacionConPosicion = pyo.Constraint(
            m.TRENES, m.SERVICIOS, m.DIAS, rule=_asignacion_con_posicion_rule
        )

        # (P4) Evolución de tren sin acoplamiento
        def _evolucion_solo_rule(m, r, j, D):
            if D >= datos.H:
                return pyo.Constraint.Skip
            viaje_solo = sum(
                m.u[r, n, D]
                * (1 - sum(m.pareja[r, r2, D] for r2 in m.TRENES if r2 != r))
                for n in m.SERVICIOS
                if m.DESTINO_N[n] == j
            )
            return m.pos[r, j, D + 1] >= viaje_solo

        m.EvolucionSolo = pyo.Constraint(
            m.TRENES, m.ESTACIONES, m.DIAS, rule=_evolucion_solo_rule
        )

        # (P5) Evolución de tren con acoplamiento
        def _evolucion_acoplado_rule(m, r, j, D):
            if D >= datos.H:
                return pyo.Constraint.Skip
            viaje_acoplado = sum(
                m.u[r, n, D] * sum(m.pareja[r, r2, D] for r2 in m.TRENES if r2 != r)
                for n in m.SERVICIOS
                if m.DESTINO_N[n] == j
            )
            return m.pos[r, j, D + 1] >= viaje_acoplado

        m.EvolucionAcoplado = pyo.Constraint(
            m.TRENES, m.ESTACIONES, m.DIAS, rule=_evolucion_acoplado_rule
        )

        # (P6) Tren que no viaja permanece
        def _permanece_rule(m, r, i, D):
            if D >= datos.H:
                return pyo.Constraint.Skip
            no_viaja = m.pos[r, i, D] - sum(m.u[r, n, D] for n in m.SERVICIOS)
            return m.pos[r, i, D + 1] >= no_viaja

        m.Permanece = pyo.Constraint(
            m.TRENES, m.ESTACIONES, m.DIAS, rule=_permanece_rule
        )

        # (P7) Acoplamiento requiere misma estación origen
        def _acoplamiento_misma_estacion_rule(m, r, r2, D):
            if r >= r2:
                return pyo.Constraint.Skip
            misma_estacion = sum(m.pos[r, i, D] * m.pos[r2, i, D] for i in m.ESTACIONES)
            return m.pareja[r, r2, D] <= misma_estacion

        m.AcoplamientoMismaEstacion = pyo.Constraint(
            m.TRENES, m.TRENES, m.DIAS, rule=_acoplamiento_misma_estacion_rule
        )

        # (P8) Relación pareja-acoplado
        def _acoplado_suma_rule(m, r, D):
            total_parejas = sum(m.pareja[r, r2, D] for r2 in m.TRENES if r2 != r)
            return m.acoplado[r, D] == total_parejas

        m.AcopladoSuma = pyo.Constraint(m.TRENES, m.DIAS, rule=_acoplado_suma_rule)

        # (P9) Simetría del acoplamiento
        def _pareja_simetrica_rule(m, r, r2, D):
            if r >= r2:
                return pyo.Constraint.Skip
            return m.pareja[r, r2, D] == m.pareja[r2, r, D]

        m.ParejaSimetrica = pyo.Constraint(
            m.TRENES, m.TRENES, m.DIAS, rule=_pareja_simetrica_rule
        )

        # (P10) No auto-acoplamiento
        def _no_auto_acople_rule(m, r, D):
            return m.pareja[r, r, D] == 0

        m.NoAutoAcople = pyo.Constraint(m.TRENES, m.DIAS, rule=_no_auto_acople_rule)

        # (P11) Límite de acoplamiento por tren
        def _limite_acoplado_rule(m, r, D):
            return m.acoplado[r, D] <= 1

        m.LimiteAcoplado = pyo.Constraint(m.TRENES, m.DIAS, rule=_limite_acoplado_rule)

        # ============================================================
        # 5. RESTRICCIONES DE CAPACIDAD Y VENTAS (C1-C5)
        # ============================================================

        # (C1) Capacidad total del servicio con acoplamiento
        def _capacidad_servicio_rule(m, n, D):
            capacidad_base = sum(
                m.CAP_R[r] * m.u[r, n, D] * (1 + m.acoplado[r, D]) for r in m.TRENES
            )
            return m.cap_servicio[n, D] == capacidad_base

        m.CapacidadServicio = pyo.Constraint(
            m.SERVICIOS, m.DIAS, rule=_capacidad_servicio_rule
        )

        # (C2) Capacidad disponible por clase
        def _capacidad_disponible_rule(m, k, n, D):
            asientos_reservados = sum(m.RS[D, kp, n] for kp in m.CLASES)
            return m.c[k, n, D] == m.cap_servicio[n, D] - asientos_reservados

        m.CapacidadDisponible = pyo.Constraint(
            m.CLASES, m.SERVICIOS, m.DIAS, rule=_capacidad_disponible_rule
        )

        # (C3) Ventas limitadas por demanda
        def _ventas_demanda_rule(m, k, n, t, D):
            if t > D:
                return pyo.Constraint.Skip
            demanda = m.PL[n, D] * sum(
                m.q[k, n, t, o, D] * m.DE[D, t, k, n, o] for o in m.OPCIONES
            )
            return m.s[k, n, t, D] <= demanda

        m.VentasDemanda = pyo.Constraint(
            m.CLASES, m.SERVICIOS, m.TIEMPO, m.DIAS, rule=_ventas_demanda_rule
        )

        # (C4) Ventas limitadas por capacidad remanente
        def _ventas_capacidad_rule(m, k, n, t, D):
            if t > D:
                return pyo.Constraint.Skip
            remanente = m.c[k, n, D] - sum(
                m.s[k, n, m_t, D] for m_t in m.TIEMPO if m_t > t
            )
            return m.s[k, n, t, D] <= remanente

        m.VentasCapacidad = pyo.Constraint(
            m.CLASES, m.SERVICIOS, m.TIEMPO, m.DIAS, rule=_ventas_capacidad_rule
        )

        # (C5) No ventas si el servicio no opera (Big-M)
        def _no_ventas_sin_servicio_rule(m, k, n, t, D):
            if t > D:
                return pyo.Constraint.Skip
            return m.s[k, n, t, D] <= m.BIG_M * m.PL[n, D]

        m.NoVentasSinServicio = pyo.Constraint(
            m.CLASES, m.SERVICIOS, m.TIEMPO, m.DIAS, rule=_no_ventas_sin_servicio_rule
        )

        # ============================================================
        # 6. RESTRICCIONES DE ASIGNACIÓN DE TRENES (A1-A4)
        # ============================================================

        # (A1) Un servicio por tren por día
        def _un_servicio_por_tren_rule(m, r, D):
            return sum(m.u[r, n, D] for n in m.SERVICIOS) <= 1

        m.UnServicioPorTren = pyo.Constraint(
            m.TRENES, m.DIAS, rule=_un_servicio_por_tren_rule
        )

        # (A2) Máximo dos trenes por servicio (acoplamiento)
        def _max_trenes_servicio_rule(m, n, D):
            return sum(m.u[r, n, D] for r in m.TRENES) <= 2

        m.MaxTrenesServicio = pyo.Constraint(
            m.SERVICIOS, m.DIAS, rule=_max_trenes_servicio_rule
        )

        # (A3) Consistencia de acoplamiento en el mismo servicio
        def _consistencia_acoplamiento_rule(m, r, r2, D):
            if r >= r2:
                return pyo.Constraint.Skip
            mismo_servicio = sum(m.u[r, n, D] * m.u[r2, n, D] for n in m.SERVICIOS)
            return m.pareja[r, r2, D] <= mismo_servicio

        m.ConsistenciaAcoplamiento = pyo.Constraint(
            m.TRENES, m.TRENES, m.DIAS, rule=_consistencia_acoplamiento_rule
        )

        # (A4) Restricción de tipo para acoplamiento (mismo tipo)
        def _mismo_tipo_acoplamiento_rule(m, r, r2, D):
            if r >= r2:
                return pyo.Constraint.Skip
            mismo_tipo = (
                1
                if datos.tipo_tren_por_r[r - 1] == datos.tipo_tren_por_r[r2 - 1]
                else 0
            )
            return m.pareja[r, r2, D] <= mismo_tipo

        m.MismoTipoAcoplamiento = pyo.Constraint(
            m.TRENES, m.TRENES, m.DIAS, rule=_mismo_tipo_acoplamiento_rule
        )

        # ============================================================
        # 7. RESTRICCIONES DE PRECIOS (PR1-PR2)
        # ============================================================

        # (PR1) Jerarquía de precios por clase
        def _precio_jerarquico_rule(m, k, n, t, D):
            if k >= datos.K or t > D:
                return pyo.Constraint.Skip
            precio_k = sum(m.q[k, n, t, o, D] * m.PO[k, n, o] for o in m.OPCIONES)
            precio_k1 = sum(
                m.q[k + 1, n, t, o, D] * m.PO[k + 1, n, o] for o in m.OPCIONES
            )
            return precio_k >= precio_k1 + m.EPSILON

        m.PrecioJerarquico = pyo.Constraint(
            m.CLASES, m.SERVICIOS, m.TIEMPO, m.DIAS, rule=_precio_jerarquico_rule
        )

        # (PR2) Una opción de precio por segmento
        def _una_opcion_precio_rule(m, k, n, t, D):
            if t > D:
                return pyo.Constraint.Skip
            return sum(m.q[k, n, t, o, D] for o in m.OPCIONES) == 1

        m.UnaOpcionPrecio = pyo.Constraint(
            m.CLASES, m.SERVICIOS, m.TIEMPO, m.DIAS, rule=_una_opcion_precio_rule
        )

        # ============================================================
        # 8. FUNCIÓN OBJETIVO (OBJ)
        # ============================================================

        def _objetivo_rule(m):
            # Ingresos totales
            ingresos = sum(
                m.s[k, n, t, D]
                * sum(m.q[k, n, t, o, D] * m.PO[k, n, o] for o in m.OPCIONES)
                for k in m.CLASES
                for n in m.SERVICIOS
                for t in m.TIEMPO
                for D in m.DIAS
                if t <= D
            )

            # Costes fijos de trenes
            costes_fijos = sum(
                m.FC_R[r] * m.u[r, n, D]
                for r in m.TRENES
                for n in m.SERVICIOS
                for D in m.DIAS
            )

            # Costes variables
            costes_variables = sum(
                m.VC_KR[k, r]
                * (m.RS[D, k, n] + sum(m.s[k, n, t, D] for t in m.TIEMPO if t <= D))
                for r in m.TRENES
                for n in m.SERVICIOS
                for k in m.CLASES
                for D in m.DIAS
            )

            # Costes de acoplamiento
            costes_acoplamiento = m.CC * sum(
                m.acoplado[r, D] for r in m.TRENES for D in m.DIAS
            )

            # Beneficio total
            beneficio = (
                ingresos
                + m.AR
                - costes_fijos
                - costes_variables
                - costes_acoplamiento
                - m.OC
            )

            return beneficio

        m.Objetivo = pyo.Objective(rule=_objetivo_rule, sense=pyo.maximize)

    # ============================================================
    # MÉTODOS DE SOLUCIÓN
    # ============================================================

    def resolver(
        self,
        solver_name: str = "gurobi",
        time_limit: int = 3600,
        gap: float = 0.01,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Resuelve el modelo con el solver especificado.

        Args:
            solver_name: Nombre del solver ('gurobi', 'cplex', 'cbc', 'glpk')
            time_limit: Límite de tiempo en segundos
            gap: Tolerancia de optimalidad MIP
            verbose: Mostrar información del solver

        Returns:
            Resultados de la optimización
        """

        # Configurar solver
        if solver_name.lower() == "gurobi":
            solver = pyo.SolverFactory("gurobi")
            solver.options["TimeLimit"] = time_limit
            solver.options["MIPGap"] = gap
            if not verbose:
                solver.options["OutputFlag"] = 0

        elif solver_name.lower() == "cplex":
            solver = pyo.SolverFactory("cplex")
            solver.options["timelimit"] = time_limit
            solver.options["mip_tolerances_mipgap"] = gap
            if not verbose:
                solver.options["output"] = "no"

        elif solver_name.lower() == "cbc":
            solver = pyo.SolverFactory("cbc")
            solver.options["sec"] = time_limit
            solver.options["ratio"] = gap
            if not verbose:
                solver.options["log"] = 0

        elif solver_name.lower() == "glpk":
            solver = pyo.SolverFactory("glpk")
            solver.options["tmlim"] = time_limit
            solver.options["mipgap"] = gap
            if not verbose:
                solver.options["msg_lev"] = 0
        else:
            raise ValueError(f"Solver no soportado: {solver_name}")

        # Resolver
        resultados = solver.solve(self.modelo, tee=verbose)

        return {
            "status": resultados.solver.termination_condition,
            "tiempo": (
                resultados.solver.wall_time
                if hasattr(resultados.solver, "wall_time")
                else None
            ),
            "gap": resultados.solver.gap if hasattr(resultados.solver, "gap") else None,
        }

    # ============================================================
    # MÉTODOS DE EXTRACCIÓN DE RESULTADOS
    # ============================================================

    def obtener_solucion(self) -> Dict[str, Any]:
        """Extrae la solución completa del modelo resuelto"""

        solucion = {
            "objetivo": pyo.value(self.modelo.Objetivo),
            "estado": "optimal" if self._es_optimal() else "no_optimal",
            "trenes": {"posiciones": {}, "asignaciones": {}, "acoplamientos": {}},
            "ventas": {},
            "precios": {},
        }

        # Posiciones de trenes
        for r in self.modelo.TRENES:
            for D in self.modelo.DIAS:
                for i in self.modelo.ESTACIONES:
                    val = pyo.value(self.modelo.pos[r, i, D])
                    if val and val > 0.5:
                        solucion["trenes"]["posiciones"][(r, D)] = i

        # Asignaciones
        for r in self.modelo.TRENES:
            for n in self.modelo.SERVICIOS:
                for D in self.modelo.DIAS:
                    val = pyo.value(self.modelo.u[r, n, D])
                    if val and val > 0.5:
                        solucion["trenes"]["asignaciones"][(r, D)] = n

        # Acoplamientos
        for r in self.modelo.TRENES:
            for D in self.modelo.DIAS:
                val = pyo.value(self.modelo.acoplado[r, D])
                if val and val > 0.5:
                    solucion["trenes"]["acoplamientos"][(r, D)] = True

        return solucion

    def _es_optimal(self) -> bool:
        """Verifica si la solución es óptima"""
        termination = self.modelo.solutions[-1].solver.termination_condition
        return termination == pyo.TerminationCondition.optimal

    def imprimir_resumen(self):
        """Imprime un resumen de la solución"""

        sol = self.obtener_solucion()

        print("\n" + "=" * 70)
        print("RESUMEN DE LA SOLUCIÓN DEL MODELO")
        print("=" * 70)

        print(f"\n✅ Estado: {sol['estado']}")
        print(f"💰 Beneficio total: {sol['objetivo']:,.2f} unidades monetarias")

        print("\n--- ASIGNACIONES DE TRENES ---")
        for (r, D), n in sol["trenes"]["asignaciones"].items():
            acoplado = " [ACOPLADO]" if (r, D) in sol["trenes"]["acoplamientos"] else ""
            print(f"  Día {D}: Tren {r}{acoplado} → Servicio {n}")

        print("\n--- POSICIONES ---")
        for (r, D), i in sol["trenes"]["posiciones"].items():
            print(f"  Día {D}: Tren {r} → Estación {i}")


# ============================================================================
# FUNCIONES AUXILIARES PARA GENERAR DATOS DE EJEMPLO
# ============================================================================


def generar_datos_ejemplo_pequeno() -> DatosRedFerroviaria:
    """
    Genera un conjunto de datos de ejemplo para probar el modelo.
    Este ejemplo simula una red pequeña con 2 estaciones, 2 trenes y 2 servicios.
    """

    # ===== Dimensiones =====
    H, K, S, N, R, O, SD, TD = 3, 2, 2, 2, 2, 2, 2, 2

    # ===== Tipos de tren =====
    tipos_tren = [
        TipoTren(
            "Premium",
            capacidad=40,
            costo_fijo=8500,
            costo_variable_base=75,
            multiplicador_clase=[1.5, 1.0, 0.7],
        ),
        TipoTren(
            "LowCost",
            capacidad=60,
            costo_fijo=5000,
            costo_variable_base=45,
            multiplicador_clase=[1.5, 1.0, 0.7],
        ),
    ]

    # Tren 1: Premium, Tren 2: LowCost
    tipo_tren_por_r = [0, 1]

    # ===== Servicios =====
    servicios = [
        ServicioTren(id=1, origen=1, destino=2, duracion=1, horarios=[1, 2, 3]),
        ServicioTren(id=2, origen=2, destino=1, duracion=1, horarios=[1, 2, 3]),
    ]

    # ===== Programación =====
    PL = np.ones((N, H))  # Todos los servicios operan todos los días

    # ===== Matriz de rutas (opcional) =====
    TR = np.zeros((N, S, S))
    for idx, srv in enumerate(servicios):
        TR[idx, srv.origen - 1, srv.destino - 1] = 1

    # ===== Posiciones iniciales =====
    pos_inicial = np.zeros((R, S))
    pos_inicial[0, 0] = 1  # Tren 1 en estación 1
    pos_inicial[1, 1] = 1  # Tren 2 en estación 2

    # ===== Precios =====
    precios = np.zeros((K, N, O))
    for k in range(K):
        for n in range(N):
            base = 100 if n == 0 else 80
            mult_clase = [1.5, 1.0][k]
            for o in range(O):
                mult_precio = [0.8, 1.0, 1.2][o]
                precios[k, n, o] = base * mult_clase * mult_precio

    # ===== Parámetros económicos =====
    CC, OC, AR = 200.0, 65000.0, 0.0

    # ===== Estacionalidad =====
    SE = np.ones((H, N), dtype=int)  # Siempre temporada media

    # ===== Demanda determinista =====
    DE = np.ones((SD, TD, K, N, O)) * 30.0
    DE[:, :, :, :, 0] = 45.0  # Precio bajo: más demanda
    DE[:, :, :, :, 1] = 30.0  # Precio medio: demanda media
    DE[:, :, :, :, 2] = 20.0 if O > 2 else 30.0  # Precio alto: menos demanda

    # ===== Ventas previas =====
    RW = np.zeros((H, K, N))
    RS = np.zeros((H, K, N))

    return DatosRedFerroviaria(
        H=H,
        K=K,
        S=S,
        N=N,
        R=R,
        O=O,
        SD=SD,
        TD=TD,
        tipo_tren_por_r=tipo_tren_por_r,
        tipos_tren=tipos_tren,
        servicios=servicios,
        PL=PL,
        TR=TR,
        pos_inicial=pos_inicial,
        precios=precios,
        CC=CC,
        OC=OC,
        AR=AR,
        SE=SE,
        DE=DE,
        RW=RW,
        RS=RS,
        MAX_COUPLE=1,
        EPSILON=0.01,
        BIG_M=1e6,
    )


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MODELO DETERMINISTA DE TRENES UNITARIOS")
    print("=" * 70)

    # Generar datos de ejemplo
    print("\n📊 Generando datos de ejemplo...")
    datos = generar_datos_ejemplo_pequeno()

    print(f"  - Horizonte: {datos.H} días")
    print(f"  - Servicios: {datos.N}")
    print(
        f"  - Trenes: {datos.R} (Premium: {datos.tipo_tren_por_r.count(0)}, LowCost: {datos.tipo_tren_por_r.count(1)})"
    )
    print(f"  - Estaciones: {datos.S}")
    print(f"  - Clases: {datos.K}")

    # Crear modelo
    print("\n🔧 Construyendo modelo Pyomo...")
    modelo = ModeloTrenesUnitariosDeterminista(datos)

    # Mostrar estadísticas
    num_vars = len(modelo.modelo.component_objects(pyo.Var, active=True))
    num_constr = len(modelo.modelo.component_objects(pyo.Constraint, active=True))
    print(f"  - Variables: {num_vars} grupos")
    print(f"  - Restricciones: {num_constr} grupos")

    # Resolver
    print("\n🚀 Resolviendo con Gurobi...")
    resultado = modelo.resolver(
        solver_name="gurobi", time_limit=60, gap=0.01, verbose=True
    )

    print(f"\n📈 Estado de la solución: {resultado['status']}")
    print(f"⏱️  Tiempo: {resultado['tiempo']:.2f} segundos")

    if resultado["gap"] is not None:
        print(f"📊 Gap de optimalidad: {resultado['gap']:.4%}")

    # Mostrar resultados
    modelo.imprimir_resumen()
