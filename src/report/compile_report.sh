#!/bin/bash

# Script para compilar el informe del Taller 3
# Uso: ./compile_report.sh

echo "=================================================="
echo "Compilando Informe Taller 3 - Redes Neuronales"
echo "=================================================="
echo ""

# Verificar que existe el archivo .tex
if [ ! -f "informe_taller3.tex" ]; then
    echo "ERROR: No se encuentra el archivo informe_taller3.tex"
    exit 1
fi

# Verificar que existen las imágenes referenciadas
echo "Verificando existencia de imágenes..."
IMAGE_DIR="../output/images"
REQUIRED_IMAGES=(
    "lstm_comparison.png"
    "gru_comparison.png"
    "global_comparison.png"
    "seq_length_effect.png"
    "batch_size_effect.png"
    "simple_vs_stacked.png"
)

missing_images=0
for img in "${REQUIRED_IMAGES[@]}"; do
    if [ ! -f "$IMAGE_DIR/$img" ]; then
        echo "  ⚠️  ADVERTENCIA: Falta imagen $img"
        missing_images=$((missing_images + 1))
    else
        echo "  ✓ Encontrada: $img"
    fi
done

if [ $missing_images -gt 0 ]; then
    echo ""
    echo "⚠️  Faltan $missing_images imagen(es). La compilación continuará pero"
    echo "   pueden aparecer errores de imágenes no encontradas."
    echo ""
    read -p "¿Desea continuar? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Compilación cancelada."
        exit 1
    fi
fi

echo ""
echo "Primera compilación..."
pdflatex -interaction=nonstopmode informe_taller3.tex > compile.log 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR en la primera compilación. Revisar compile.log"
    tail -n 20 compile.log
    exit 1
fi

echo "Segunda compilación (para referencias cruzadas)..."
pdflatex -interaction=nonstopmode informe_taller3.tex >> compile.log 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR en la segunda compilación. Revisar compile.log"
    tail -n 20 compile.log
    exit 1
fi

# Limpiar archivos auxiliares
echo "Limpiando archivos auxiliares..."
rm -f *.aux *.log *.out *.toc *.lof *.lot

echo ""
echo "=================================================="
echo "✓ Compilación exitosa!"
echo "=================================================="
echo ""
echo "PDF generado: informe_taller3.pdf"
echo ""
