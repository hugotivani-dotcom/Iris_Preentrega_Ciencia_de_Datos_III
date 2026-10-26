# Pre-entrega 1 — Pipeline base de Deep Learning con PyTorch

## Objetivo

Construir y documentar un pipeline reproducible de entrenamiento y validación con un clasificador base multiclase para **Iris** (150 flores, 4 características, 3 especies). Implementado en PyTorch mediante `nn.Sequential`, `CrossEntropyLoss`, `Adam` y `autograd`.

## Estructura

```text
preentrega_iris_pytorch/
├── notebooks/
│   └── 01_clasificador_iris_pytorch.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

No hace falta un directorio `data/`: el dataset se obtiene mediante `sklearn.datasets.load_iris()` y no se guarda en archivos externos.

## Entorno y ejecución

La ejecución de esta entrega se realizó con PyTorch **2.12.1+cpu**,
utilizando CPU.

Requiere Python 3.10 o posterior con dependencias de `requirements.txt`. Se puede ejecutar desde VS Code o Jupyter.

```bash
python -m pip install -r requirements.txt
```

Abrir `notebooks/01_clasificador_iris_pytorch.ipynb` y utilizar **Restart Kernel & Run All**. El notebook selecciona automáticamente CUDA, MPS o CPU. En la ejecución revisada se utilizó CPU.

> Las versiones exactas observadas al validar la entrega se muestran en una celda del notebook. Para instalar PyTorch con soporte GPU específico, consultar las instrucciones oficiales para el sistema operativo y hardware utilizados.

## Arquitectura e hiperparámetros

| Componente         | Configuración                                                      |
| ------------------ | ------------------------------------------------------------------ |
| Dataset            | Iris, integrado en scikit-learn                                    |
| División           | 80 % entrenamiento, 20 % validación, estratificada                 |
| Semilla            | `42` (NumPy, Python y PyTorch)                                     |
| Modelo             | `Linear(4,8) → ReLU → Linear(8,3)`                                 |
| Función de pérdida | `CrossEntropyLoss`                                                 |
| Optimizador        | `Adam`                                                             |
| Learning rate      | `0.1` (conservado del notebook original)                           |
| Máximo de épocas   | `100`                                                              |
| Early Stopping     | `patience = 10`, por mínimo Loss de validación                     |
| Checkpoint         | Se recuperan en memoria los mejores pesos (`deepcopy(state_dict)`) |
| Métricas           | Loss y Accuracy de entrenamiento y validación por época            |

## Resultados reproducidos en la revisión

Para Python/PyTorch en CPU, ejecutando el notebook completo:

| Métrica                             |          Resultado |
| ----------------------------------- | -----------------: |
| Épocas ejecutadas                   |                100 |
| Mejor época                         |    100 (índice 99) |
| Loss de entrenamiento, mejor modelo |           0.154392 |
| Loss de validación, mejor modelo    |           0.177231 |
| Accuracy de entrenamiento           |            97.50 % |
| Accuracy de validación              | 96.67 % (29 de 30) |
| Early Stopping                      |       No se activó |

Las métricas se calculan de nuevo al ejecutar el notebook; pueden variar con cambios de versión, plataforma o hiperparámetros. Las figuras de Loss y Accuracy están guardadas en la salida del notebook y son reproducibles.

### Interpretación breve

El Loss de entrenamiento y el de validación descienden durante las 100 épocas. El Accuracy de validación alcanza aproximadamente 96.67 % y se estabiliza mientras la función de pérdida continúa ajustándose. El mínimo Loss de validación se obtiene en la última época, por lo que no se cumple la condición de Early Stopping. Esta observación no demuestra por sí sola ausencia de sobreajuste.

## Alcance y limitaciones

- El conjunto de validación contiene **solo 30 flores**. Cada clasificación acertada adicional cambia el Accuracy en aproximadamente **3.33 puntos porcentuales**.
- La validación se utiliza para seguimiento y elección del mejor estado; no hay un conjunto de **test independiente** para una evaluación final imparcial.
- No se aplicó estandarización para preservar el experimento original. Como mejora futura, cualquier `StandardScaler` debe ajustarse únicamente en los datos de entrenamiento.
- Este proyecto evalúa la **calidad del pipeline**, no la complejidad del modelo.

## Guía mínima para los commits

Ejemplos de mensajes claros: `Inicializar notebook y datos Iris`, `Implementar entrenamiento y validación PyTorch`, `Agregar tracking y checkpoint Early Stopping`, `Documentar resultados reproducibles`.
