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
#
# 0.2.6: el worker arregla el CA heredado del instalador (las descargas de
# pesos volvían a fallar con "Permission denied" y no arrancaban nunca) y la
# cancelación de descargas (corta también el camino Xet y descarta los restos
# a medio bajar, que inflaban el tamaño/porcentaje que ve el panel).
#
# 0.2.7: el catálogo (ListModels) reporta la ruta de los pesos en disco
# (download_path), así el back y el panel ubican los archivos sin adivinar el
# slug de la caché. Un worker 0.2.6 sigue funcionando: sólo omite el campo.
#
# 0.2.8: el snapshot del heartbeat también lleva `downloaded` y
# `download_path`, que es lo único que ve el back cuando la GPU está detrás de
# NAT: sin esto el panel admin no podía mostrar la ruta en equipos emparejados.
#
# 0.2.9: el artifact server lista (GET /artifacts) y borra
# (DELETE /artifacts/<nombre>) lo generado, que es lo que proxya el back en
# /api/v1/worker/artifacts. Antes sólo se podía pedir un archivo por nombre.
BRIDGE_VERSION = "0.2.9"
