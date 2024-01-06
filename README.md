# Servicio de Predicción de Iris

Este proyecto es una aplicación basada en Flask que proporciona una API para predecir la especie de las flores de iris utilizando aprendizaje automático. Soporta varios modelos incluyendo regresión logística (lr), máquina de vectores de soporte (svm), árbol de decisión (dt) y vecinos más cercanos (knn).

## Características
- **API de Predicción**: Una ruta Flask para predecir especies de iris basada en características de entrada.
- **Soporte de Modelos**: Capacidad para elegir entre diferentes modelos o utilizar todos los modelos para la predicción.
- **Manejo de Errores**: Manejo elegante de errores y especificaciones incorrectas del modelo.

## Primeros Pasos

### Prerrequisitos
Asegúrate de tener instalado lo siguiente:
- Python 3
- Flask
- módulo pickle

### Instalación
1. Al usar Poetry todo será más fácil!!!

### Ejecución de la Aplicación

1. Navega al directorio del proyecto.
2. Ejecuta la aplicación Flask.

    ```bash
    poetry run python 5073_tasca03/predict_service.py
    ```

La aplicación se iniciará en `localhost` en el puerto `8000`.

## Uso

Para predecir la especie del iris, envía una solicitud POST al endpoint `/predict` con el siguiente payload JSON:

```json
{
    "model": "<nombre_del_modelo>", // 'lr', 'svm', 'dt', 'knn', o 'ALL'
    "X": <característica_x>,        // Valor numérico
    "Y": <característica_y>         // Valor numérico
}

Si no usas modelo o indicas el valor 'All0, se realizará predicción en los cuatro modelos.