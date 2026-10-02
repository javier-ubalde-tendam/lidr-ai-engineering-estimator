# Session 02 - CAG


## Funcionalidad

Estructura base del Proyecto 1: una aplicación FastAPI con un endpoint que reciba el texto de una transcripción de reunión y devuelva una estimación de software generada por un LLM, utilizando arquitectura CAG (contexto estático inyectado en el prompt).

Servicio funcional que:
- Recibe una transcripción de reunión vía API REST
- Inyecta contexto estático (ejemplos de estimaciones previas) directamente en el prompt
- Envía la petición a un LLM (OpenAI o Anthropic)
- Devuelve la estimación generada como respuesta JSON

Estructura:
- routers/ gestiona los endpoints HTTP
    - crea un endpoint POST /api/v1/estimate
    - recibe un body JSON + llame al services
    - devuelva la estimación
- services/ contiene la lógica de negocio (llamada al LLM)
    - Construye el system prompt 
    - Inyecta los ejemplos de contexto
    - Envía la transcripción del usuario
    - Devuelve la respuesta del modelo LLM
- context/ almacena los datos estáticos que inyectamos en el prompt.
- schemas/ define y valida los datos de entrada y salida
- .env tiene las variables de entorno y las API Keys. Esto no se sube al repo pero incluimos un fichero .env.- example con la estructura esperada
- main.py configura la aplicación FastAPI
    - Incluye el router de estimaciones con el prefijo /api/v1
    - Añade un endpoint GET /health
    - Configura título y descripción para la docu automática (Swagger)


## Tecnologías

- pydantic: uso de ficheros de configuración (p.ej. para variables de entorno, API_KEY, etc.)
- uv: gestor de paquetes y proyectos para Python de alto rendimiento y ultra rápido. Controla las dependencias, lanza la aplicación, etc. El comando “--reload” recarga el servidor si algún fichero cambia (muy útil para desarrollo)
- FastAPI: define la aplicación, rutas, validaciones, dependencias, etc. Pero no escucha directamente en un puerto de red. Para eso necesita un servidor ASGI como Uvicorn.
- Uvicorn: servidor web ASGI para servir el endpoint de FastAPI


## Cómo invocar

### Arrancar la aplicación
uv run uvicorn app.main:app --reload

### Lanzar petición
curl -X POST <http://localhost:8000/api/v1/estimate> \\
  -H "Content-Type: application/json" \\
  -d '{
    "transcription": "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing page con formulario de contacto, integración con su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG. El plazo ideal sería tenerlo listo en 4 semanas. El diseño ya existe en Figma."
  }'

También se puede usar el Swagger en http://127.0.0.1:8000/docs para enviar una petición de prueba