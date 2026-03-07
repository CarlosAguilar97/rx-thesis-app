# RX Tórax - Apoyo Diagnóstico con IA

Proyecto web para análisis de radiografías de tórax usando dos modelos de Deep Learning:

- **Modelo 1: Patrones radiomorfológicos**
- **Modelo 2: Enfermedades cardiopulmonares**

La aplicación permite:

- subir una radiografía desde la web
- ejecutar ambos modelos automáticamente
- mostrar resultados en interfaz web
- guardar resultados en **Turso**
- generar **Grad-CAM** para el modelo de patrones

---

## Tecnologías usadas

- **FastAPI**
- **Turso / libSQL**
- **PyTorch**
- **Torchvision**
- **OpenCV**
- **Pillow**
- **TailwindCSS + DaisyUI**

---

## Estructura del proyecto

```text
rx-thesis-app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── db.py
│   ├── config.py
│   ├── schemas.py
│   ├── models_logic.py
│   ├── sql/
│   │   └── schema.sql
│   └── models/
│       ├── patterns/
│       │   ├── __init__.py
│       │   ├── infer_patterns.py
│       │   ├── best_multilabelv2.pth
│       │   └── thresholds_best_f1.json
│       └── diseases/
│           ├── __init__.py
│           ├── infer_diseases.py
│           └── clinical_model.pth
├── static/
│   └── index.html
├── uploads/
├── outputs_infer/
├── requirements.txt
├── .env
└── README.md