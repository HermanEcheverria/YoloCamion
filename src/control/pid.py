"""Controlador PID generico con anti-windup por clamping.

Convencion de signos:
    error = setpoint - medicion
    salida positiva  -> "empuja hacia arriba" del setpoint (acelera, gira a +x)
    salida negativa  -> "empuja hacia abajo"  (frena, gira a -x)

Anti-windup:
    El termino integral se acumula con `integral += error * dt`. Si el
    aporte integral (Ki*integral) ya excede el limite, lo recortamos al
    limite de saturacion para que el integrador no crezca sin freno
    durante saturacion sostenida (clamping clasico).

Uso tipico (PID de volante en gamepad_pid):
    pid = PIDController(kp=0.55, ki=0.015, kd=0.08, limite=1.0)
    salida = pid.calcular(setpoint=0.0, medicion=desviacion_carril, dt=0.033)
    gamepad.left_joystick_float(x_value_float=salida, y_value_float=0.0)
"""


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        limite: float = 1.0,
    ):
        self._kp = float(kp)
        self._ki = float(ki)
        self._kd = float(kd)
        self._limite = abs(float(limite))

        self._integral: float = 0.0
        self._error_anterior: float = 0.0
        self._primera_llamada: bool = True  # evita spike D en primer frame

    def calcular(self, setpoint: float, medicion: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        error = setpoint - medicion

        # P
        p = self._kp * error

        # I con anti-windup por clamping del termino integral
        self._integral += error * dt
        if self._ki != 0.0:
            i_raw = self._ki * self._integral
            if abs(i_raw) > self._limite:
                signo = 1.0 if i_raw > 0 else -1.0
                self._integral = signo * self._limite / self._ki
        i = self._ki * self._integral

        # D — se omite en la primera llamada (evita el spike por error_anterior=0)
        if self._primera_llamada:
            d = 0.0
            self._primera_llamada = False
        else:
            d = self._kd * (error - self._error_anterior) / dt
        self._error_anterior = error

        salida = p + i + d
        if salida > self._limite:
            return self._limite
        if salida < -self._limite:
            return -self._limite
        return salida

    def reset(self, error_inicial: float = 0.0) -> None:
        self._integral = 0.0
        self._error_anterior = error_inicial
        self._primera_llamada = True
