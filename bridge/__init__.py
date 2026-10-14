"""Bridge saliente de Synapta (Fase A del plan distribuido).

El worker se empareja con el back mediante un código corto y después
consulta trabajos por HTTP long-poll saliente: funciona detrás del
NAT/firewall del hogar, sin abrir puertos ni túneles.

Piezas:
    config    — archivo de configuración por-usuario (~/.config/…), cross-platform
    client    — cliente HTTP contra el back (register/heartbeat/claim/report/upload)
    journal   — journal de jobs en disco: sobrevive cortes de luz a mitad de job
    agent     — el loop: heartbeat + claim + ejecución vía gRPC local + upload
    pairing   — CLI de emparejamiento (el usuario pega el código de la app)
"""

# Versión del agente bridge; viaja en el heartbeat para que el back detecte
# workers desactualizados antes de un rollout (riesgo "versiones mezcladas"
# del plan).
BRIDGE_VERSION = "0.2.2"
