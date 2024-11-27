
# Informe de Resultados del Modelo de Clasificación

## 1. Rendimiento del Modelo

### Métricas de Validación Cruzada
- Media: 0.4478 (+/- 0.0615)
- Rango: 0.4000 - 0.4944

### Comparación Train vs Test
- Score en Training: 0.9878
- Score en Test: 0.3700
- Diferencia: 0.6178

## 2. Análisis por Categoría (Test Set)

              precision    recall  f1-score  support
1              0.372881  0.415094  0.392857    53.00
2              0.340909  0.340909  0.340909    44.00
3              0.415094  0.392857  0.403670    56.00
4              0.340909  0.319149  0.329670    47.00
accuracy       0.370000  0.370000  0.370000     0.37
macro avg      0.367448  0.367002  0.366777   200.00
weighted avg   0.370154  0.370000  0.369607   200.00

## 3. Características Relevantes

tenure_age_ratio       0.092924
income_education       0.085462
potential_value        0.082939
tenure                 0.082337
stability_score        0.078646
income_per_age         0.073936
income_per_resident    0.071929
employ_tenure          0.062586
income                 0.061617
age                    0.059452

## 4. Control de Validación

### Estabilidad del Modelo
- Desviación estándar CV: 0.0308
- Diferencia Train-Test: 0.6178

### Consistencia por Categoría
              Train   Test
1             0.980  0.393
2             0.991  0.341
3             0.987  0.404
4             0.993  0.330
accuracy      0.988  0.370
macro avg     0.988  0.367
weighted avg  0.988  0.370

## 5. Recomendaciones

### Mejoras Técnicas
1. Implementar técnicas de balanceo de clases
2. Crear características adicionales más específicas
3. Explorar ensambles de modelos

### Recomendaciones de Negocio
1. Enfocar esfuerzos en mejorar predicción de categorías con menor rendimiento
2. Desarrollar estrategias específicas por segmento
3. Implementar sistema de monitoreo continuo
    