# Agente de comentarios de YouTube

Responde los comentarios de tu canal **como tú escribes**. Atiende primero a todos los
**miembros del canal** y después a unos cuantos **suscriptores**. Se ejecuta solo todos los días a
las **9:00 p. m. hora Colombia** con GitHub Actions.

## Cómo funciona

1. **Entrenamiento (`train`) — gratis, sin OpenAI.** Recorre todos los hilos de comentarios de tu
   canal y guarda **solo** los pares *comentario de otra persona → tu respuesta*. Con eso arma un
   perfil de tu estilo: largo típico, palabras que más usas, emojis, cómo empiezas y cómo cierras, y
   ~20 respuestas reales tuyas como ejemplo. No usa nada que no hayas escrito tú.
2. **Respuesta diaria (`run`).**
   - Lee los comentarios de los últimos 3 días que aún no respondiste.
   - Toma **todos los de miembros** (tope de 50) y luego **10 de suscriptores** (los que tienen más "me gusta").
   - Para cada comentario busca, localmente, tus respuestas pasadas a comentarios parecidos.
   - Le pide a `gpt-6-luna` las respuestas **en lotes de 10** y publica las que el modelo no descarta.
     Si un comentario es spam, delicado o algo que solo tú sabrías responder, el modelo lo **salta**.
   - Nunca responde dos veces el mismo hilo, ni hilos donde ya respondiste tú.
   - Nunca aprende de las respuestas que publicó el propio agente.

### Uso mínimo de la API de OpenAI

- El entrenamiento no gasta tokens: es estadística local.
- Una llamada cada 10 comentarios, con `reasoning: none` y `service_tier: flex` (mitad de precio;
  si flex falla, reintenta con el tier normal).
- Las instrucciones (tu estilo + ejemplos) son idénticas en cada llamada, así que OpenAI las
  cachea y las cobra a ~10%.
- Con ~60 respuestas al día son ~6 llamadas: **centavos de dólar al mes**. Cada reporte muestra los
  tokens usados.

## Configuración (una sola vez)

### 1. Credenciales de YouTube

1. En [Google Cloud Console](https://console.cloud.google.com/) crea un proyecto y habilita
   **YouTube Data API v3**.
2. **Pantalla de consentimiento OAuth**: tipo *Externo*, agrega tu correo como usuario de prueba y
   luego pulsa **Publicar app** (*In production*). Si la dejas en *Testing*, el token vence cada 7
   días y el agente deja de funcionar. Google mostrará "app no verificada" cuando la autorices; es
   normal porque es solo para ti.
3. **Credenciales → Crear ID de cliente OAuth → App de escritorio**. Descarga el JSON como
   `client_secret.json` (está en `.gitignore`, no lo subas).
4. En tu computador:
   ```bash
   pip install google-auth-oauthlib
   python get_refresh_token.py client_secret.json
   ```
   Entra con la cuenta del canal (si tienes varios, elige el correcto). Imprime
   `YT_CLIENT_ID`, `YT_CLIENT_SECRET` y `YT_REFRESH_TOKEN`.

### 2. Clave de OpenAI

Crea una API key en <https://platform.openai.com/api-keys>.

### 3. Secretos en GitHub

En el repo: **Settings → Secrets and variables → Actions → New repository secret**:

| Secreto | Valor |
|---|---|
| `YT_CLIENT_ID` | del paso 1 |
| `YT_CLIENT_SECRET` | del paso 1 |
| `YT_REFRESH_TOKEN` | del paso 1 |
| `OPENAI_API_KEY` | del paso 2 |

### 4. Primera prueba (modo simulación)

El agente arranca en **modo simulación**: genera las respuestas pero **no publica nada**.

1. **Actions → Agente de comentarios de YouTube → Run workflow** con `command = train`.
2. Luego otra vez con `command = run`.
3. Descarga el artefacto `reporte-…`: trae cada comentario con la respuesta que habría publicado.

### 5. Activarlo de verdad

Cuando te gusten las respuestas: **Settings → Secrets and variables → Actions → Variables →
New repository variable**:

- `DRY_RUN` = `false`. Desde esa noche publica solo a las 9 p. m.
- `TRAIN_BEFORE` = la fecha de hoy (ej. `2026-09-24`). Es una protección extra para que, si algún
  día se reentrena, nunca aprenda de respuestas escritas por el agente.

## Ajustes opcionales (variables del repositorio)

| Variable | Por defecto | Qué hace |
|---|---|---|
| `DRY_RUN` | `true` | `false` para publicar de verdad |
| `OPENAI_MODEL` | `gpt-6-luna` | Modelo de OpenAI |
| `LOOKBACK_DAYS` | `3` | Cuántos días hacia atrás buscar comentarios sin responder |
| `MAX_MEMBER_REPLIES` | `50` | Tope diario de respuestas a miembros |
| `MAX_SUBSCRIBER_REPLIES` | `10` | Respuestas diarias a suscriptores |
| `MAX_REPLIES_PER_AUTHOR` | `2` | Máximo de respuestas por persona y día |
| `INCLUDE_NON_SUBSCRIBERS` | `false` | Completar los cupos de suscriptores con otros comentaristas |
| `TRAIN_BEFORE` | — | Fecha ISO (ej. `2026-09-24`); solo aprende de respuestas anteriores a ella |

Para que aprenda de tus respuestas nuevas, corre `train` a mano cuando quieras.

## Limitaciones que debes conocer

- **Suscriptores:** YouTube solo deja ver a los suscriptores que tienen sus suscripciones
  **públicas**. Si el agente encuentra pocos, activa `INCLUDE_NON_SUBSCRIBERS=true`.
- **Miembros:** se leen con la API de membresías. Si tu canal no la tiene disponible, pon los IDs de
  canal de tus miembros en `members.txt`.
- **Cuota de YouTube:** 10.000 unidades diarias gratis. Leer es barato; **cada respuesta publicada
  cuesta 50**. El agente se frena solo en 9.000.
- **Horario:** GitHub puede retrasar unos minutos los trabajos programados. Además, en repositorios
  públicos desactiva la programación tras 60 días sin actividad en el repo; en ese caso se reactiva
  desde la pestaña Actions.
- **Privacidad:** en un repo público los logs de Actions son públicos. El agente solo escribe totales
  en los logs; el detalle va en el artefacto del reporte, que se borra a los 7 días. Aun así, conviene
  que el repositorio sea privado.

## Desarrollo

```bash
pip install -r requirements.txt pytest
python -m pytest
```
