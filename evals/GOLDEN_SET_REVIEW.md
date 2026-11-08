# Revisión del golden set de recuperación (sesión 10)

> **PENDIENTE DE REVISIÓN HUMANA.** Este juicio de relevancia lo ha propuesto un LLM leyendo
> `data/budgets_sample.json`. Revísalo (≈5 min) antes de entregar: si cambias algún «Sí»/«No»,
> actualiza `relevant_budget_ids` en [golden_retrieval.json](golden_retrieval.json) y vuelve a
> ejecutar `scripts/eval_retrieval_s10.py` (los números dependen de esta anotación).

Criterio: **relevante = el presupuesto sirve de analogía real para estimar el proyecto de la
consulta** (alcance y componentes comparables), no «comparte palabras». Precision@5 cuenta chunks:
un presupuesto relevante con 4 componentes puede llenar como mucho 4 de las 5 posiciones.

## q1-ecommerce-checkout-trap (consulta trampa)

> Online shop for a clothing brand: product listing, shopping cart, checkout where customers pay by
> card through a payment provider, discount codes and an order confirmation email.

| Presupuesto | Relevante | Justificación |
| --- | --- | --- |
| BUD-2024-001 | No | API de banca móvil (OAuth, PSD2): fintech, nada de tienda. |
| BUD-2024-002 | No | Originación de préstamos. |
| BUD-2024-003 | No | **Trampa**: construye la pasarela de pagos (autorización, fraude, liquidación); la tienda solo la consume. |
| BUD-2024-004 | No | Robo-advisor. |
| BUD-2024-005 | **Sí** | Storefront headless: catálogo, carrito/checkout con promociones e integración con el proveedor de pago. |
| BUD-2024-006 | No | Marketplace multi-vendedor (payouts, disputas): otro modelo de negocio y alcance. |
| BUD-2024-007 | No | Reparto de supermercado (slots, rutas). |
| BUD-2024-008 | No | Devoluciones de moda: comparte «fashion/clothing», pero es un portal de devoluciones y reventa. |
| BUD-2024-009 | No | Telemedicina. |
| BUD-2024-010 | No | Monitorización remota de pacientes. |
| BUD-2024-011 | No | Farmacia. |
| BUD-2024-012 | No | Ensayos clínicos. |
| BUD-2024-013 | No | IoT industrial. |
| BUD-2024-014 | No | Gestión de almacén. |
| BUD-2024-015 | No | Red eléctrica. |
| BUD-2024-016 | No | Migración de core bancario. |
| BUD-2024-017 | **Sí** | MVP de e-commerce: lista de productos, carrito, checkout con tarjeta, email de pedido. |

## q2-finance-lending

> Digital lending platform for small businesses: online loan application with KYC and document
> upload, automated credit risk scoring from bank statements, legally binding e-signature of the
> contract and periodic reports for the regulator.

| Presupuesto | Relevante | Justificación |
| --- | --- | --- |
| BUD-2024-001 | No | Banca móvil/PSD2, sin préstamos ni scoring. |
| BUD-2024-002 | **Sí** | Scoring de crédito, flujo de solicitud con KYC, firma electrónica y reporting regulatorio: encaje casi exacto. |
| BUD-2024-003 | No | Pasarela de pagos. |
| BUD-2024-004 | No | «Risk profiling» de inversores, no riesgo de crédito. |
| BUD-2024-005 | No | E-commerce. |
| BUD-2024-006 | No | KYC de vendedores en un marketplace: vocabulario común, otro dominio. |
| BUD-2024-007 | No | Reparto. |
| BUD-2024-008 | No | Devoluciones. |
| BUD-2024-009 | No | Telemedicina. |
| BUD-2024-010 | No | Monitorización remota. |
| BUD-2024-011 | No | Farmacia. |
| BUD-2024-012 | No | «Regulatory dataset export» clínico: parecido en palabras, no en dominio. |
| BUD-2024-013 | No | IoT industrial. |
| BUD-2024-014 | No | Almacén. |
| BUD-2024-015 | No | Red eléctrica. |
| BUD-2024-016 | No | Migración de core bancario (finanzas, pero sin préstamos). |
| BUD-2024-017 | No | MVP e-commerce. |

## q3-healthcare-telemedicine

> Outpatient clinic app where patients book visits with doctors, receive reminders, join video
> consultations from home, with records synchronised to the hospital EHR through HL7 FHIR and GDPR
> consent tracking.

| Presupuesto | Relevante | Justificación |
| --- | --- | --- |
| BUD-2024-001 | No | Fintech. |
| BUD-2024-002 | No | Préstamos. |
| BUD-2024-003 | No | Pagos. |
| BUD-2024-004 | No | Robo-advisor. |
| BUD-2024-005 | No | E-commerce. |
| BUD-2024-006 | No | Marketplace. |
| BUD-2024-007 | No | «Slot booking» de reparto: vocabulario de reservas, otro dominio. |
| BUD-2024-008 | No | Devoluciones. |
| BUD-2024-009 | **Sí** | FHIR, agenda con recordatorios, videoconsulta WebRTC, consentimiento y auditoría: encaje completo. |
| BUD-2024-010 | No | Sanitario, pero monitorización con wearables; sin agenda ni vídeo. |
| BUD-2024-011 | No | Farmacia. |
| BUD-2024-012 | No | Ensayos clínicos (consentimiento/anonimización, alcance distinto). |
| BUD-2024-013 | No | IoT industrial. |
| BUD-2024-014 | No | «Dock appointment scheduling»: vocabulario de citas, otro dominio. |
| BUD-2024-015 | No | Red eléctrica. |
| BUD-2024-016 | No | Core bancario. |
| BUD-2024-017 | No | MVP e-commerce. |

## q4-industrial-iot-maintenance

> Monitoring of production line machines: collect sensor readings over MQTT into a time-series
> database, detect anomalies to predict breakdowns before they happen, and surface the alerts in the
> operators' existing control-room screens.

| Presupuesto | Relevante | Justificación |
| --- | --- | --- |
| BUD-2024-001 | No | Fintech. |
| BUD-2024-002 | No | Préstamos. |
| BUD-2024-003 | No | «Real-time fraud detection»: vocabulario de anomalías, otro dominio. |
| BUD-2024-004 | No | Robo-advisor. |
| BUD-2024-005 | No | E-commerce. |
| BUD-2024-006 | No | Marketplace. |
| BUD-2024-007 | No | Reparto. |
| BUD-2024-008 | No | Devoluciones. |
| BUD-2024-009 | No | Telemedicina. |
| BUD-2024-010 | No | **Dudoso**: ingesta de streams + alertas + dashboard (arquitectura parecida), pero dominio clínico. Revisar. |
| BUD-2024-011 | No | Farmacia. |
| BUD-2024-012 | No | Ensayos clínicos. |
| BUD-2024-013 | **Sí** | Telemetría MQTT a time-series, mantenimiento predictivo, integración con SCADA: encaje directo. |
| BUD-2024-014 | No | Almacén/AGV. |
| BUD-2024-015 | **Sí** | Pipeline de datos de contadores a escala + previsión: telemetría industrial comparable. |
| BUD-2024-016 | No | Core bancario. |
| BUD-2024-017 | No | MVP e-commerce. |

## q5-logistics-warehouse

> Software for a regional distribution centre: stock and picking with barcode scanners, booking time
> slots for trucks at the loading docks, and planning the delivery routes of the van fleet.

| Presupuesto | Relevante | Justificación |
| --- | --- | --- |
| BUD-2024-001 | No | Fintech. |
| BUD-2024-002 | No | Préstamos. |
| BUD-2024-003 | No | Pagos. |
| BUD-2024-004 | No | Robo-advisor. |
| BUD-2024-005 | No | E-commerce. |
| BUD-2024-006 | No | Fulfillment por vendedor: logística de pedidos, no de almacén. |
| BUD-2024-007 | **Sí** | Reserva de franjas, optimización de rutas e inventario en tiempo real. |
| BUD-2024-008 | No | Grading de devoluciones en almacén: vocabulario de almacén, otro alcance. |
| BUD-2024-009 | No | Telemedicina. |
| BUD-2024-010 | No | Monitorización remota. |
| BUD-2024-011 | No | Inventario de farmacia: comparte «inventory/reorder», otro dominio. |
| BUD-2024-012 | No | Ensayos clínicos. |
| BUD-2024-013 | No | IoT industrial. |
| BUD-2024-014 | **Sí** | Núcleo WMS (stock, picking, códigos de barras) y planificación de muelles. |
| BUD-2024-015 | No | Red eléctrica. |
| BUD-2024-016 | No | Core bancario. |
| BUD-2024-017 | No | MVP e-commerce. |
