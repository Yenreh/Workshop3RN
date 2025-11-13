#!/usr/bin/env bash

###############################################################################
# Ejecutar Workshop3 dentro de un contenedor NVIDIA NGC con TensorFlow + GPU
#
# Modos:
#   ./run_tf_ngc.sh           -> shell interactiva dentro del contenedor
#   ./run_tf_ngc.sh test      -> ejecuta python test.py
#   ./run_tf_ngc.sh jupyter   -> levanta Jupyter Lab (http://127.0.0.1:8888)
#
# Auto-instala requirements.txt dentro del contenedor.
###############################################################################

IMAGE="nvcr.io/nvidia/tensorflow:25.02-tf2-py3"

# Carpeta local del proyecto = carpeta donde está este script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_PROJECT_DIR="$SCRIPT_DIR"
CONTAINER_PROJECT_DIR="/workspace/workshop3"

# Archivo de requirements
# REQ_FILE="${CONTAINER_PROJECT_DIR}/requirements.txt"

MODE="${1:-shell}"

# Comando base para instalar requirements
# PIP_INSTALL_CMD="pip install --no-cache-dir -r ${REQ_FILE}"

# Comando por defecto
CONTAINER_CMD="/bin/bash"

case "$MODE" in
    test)
        CONTAINER_CMD="cd ${CONTAINER_PROJECT_DIR} && ${PIP_INSTALL_CMD} && cd src && python test.py"
        ;;
    jupyter)
        CONTAINER_CMD="cd ${CONTAINER_PROJECT_DIR} && \
${PIP_INSTALL_CMD} && \
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"
        ;;
    *)
        CONTAINER_CMD="cd ${CONTAINER_PROJECT_DIR} && ${PIP_INSTALL_CMD} && /bin/bash"
        ;;
esac

echo "--------------------------------------------"
echo "  Imagen NGC:          ${IMAGE}"
echo "  Proyecto Host:       ${HOST_PROJECT_DIR}"
echo "  Proyecto Contenedor: ${CONTAINER_PROJECT_DIR}"
echo "  Requirements:        ${REQ_FILE}"
echo "  Modo:                ${MODE}"
echo "--------------------------------------------"
echo

docker run -it --rm \
    --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8888:8888 \
    -v "${HOST_PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
    -w "${CONTAINER_PROJECT_DIR}" \
    "${IMAGE}" \
    bash -lc "${CONTAINER_CMD}"
