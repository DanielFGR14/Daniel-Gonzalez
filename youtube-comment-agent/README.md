# Agente de comentarios de YouTube

Responde los comentarios de tu canal **como tú escribes**, pero con respuestas nuevas, nunca
copiadas. Atiende primero a los **miembros del canal** y después a unos cuantos comentarios de
**suscriptores y otros espectadores**. Todos los días a las **9:05 p. m. hora Colombia** prepara las
respuestas y **te las pide aprobar**; solo publica lo que tú marcas, despacio y con pausas aleatorias.

## Cómo funciona

1. **Entrenamiento (`train`) — gratis, sin OpenAI.** Recorre todos los hilos de comentarios de tu
   canal y guarda **solo** los pares *comentario de otra persona → tu respuesta*. Con eso arma un
   perfil de tu estilo: largo típico, palabras que más usas, emojis, cómo empiezas y cómo cierras, y
   ~20 respuestas reales tuyas como ejemplo. No usa nada que no hayas escrito tú.
2. **Borradores cada noche (`draft`).**
   - Lee los comentarios de los últimos 3 días que aún no respondiste.
   - Toma **primero a los miembros** (hasta 30 por noche, del comentario más viejo al más nuevo) y
     luego **5 comentarios más** (los de suscriptores visibles primero, después los que tienen más "me gusta").
   - Le pide a `gpt-6-luna` **2 opciones por comentario** en lotes de 10 y elige la mejor (ver abajo).
     Si un comentario es spam, delicado o algo que solo tú sabrías responder, el modelo lo **salta**.
   - Abre un **issue en este repositorio** con todas las respuestas propuestas y te menciona, así que
     te llega el aviso por correo y en la app de GitHub.
3. **Tú apruebas.** En el issue marcas *Publicar esta* en las que quieras (o *Aprobar TODAS*). Puedes
   editar el texto de cualquier respuesta. Cuando terminas, **cierras el issue**. Si lo cierras como
   *no planeado*, no se publica nada. Si no lo cierras en 3 días, caduca.
4. **Publicación (`publish`), al cerrar el issue.**
   - Publica solo lo que marcaste, con tu texto final.
   - **Despacio:** entre una respuesta y otra espera un tiempo al azar de **30 a 120 segundos**, como
     máximo durante 60 minutos.
   - Justo antes de cada una revisa que el comentario siga existiendo y que no lo hayas respondido
     tú mientras tanto.
   - Después de publicar comprueba que YouTube **muestre** la respuesta. Si la oculta (señal de
     posible spam), **se detiene** y la ejecución falla a propósito para que GitHub te avise por correo.
   - Al terminar **borra el contenido del issue** y deja solo los totales (regla de datos de YouTube).

### Parecida a ti, pero nunca una copia

Tus respuestas pasadas **no se pegan**: sirven para que el modelo aprenda tu forma de escribir. Cada
opción que propone pasa por dos filtros locales (sin gastar API):

- **Novedad (obligatoria).** Se compara con todas tus respuestas reales, con lo que el agente publicó
  en los últimos 30 días y con lo que ya eligió esa misma noche. Si se parece demasiado a alguna
  (`MAX_SIMILARITY`, 0.75 por defecto), se descarta. Para comparar se ignoran mayúsculas, tildes,
  signos y emojis: "Gracias!! 🙏" cuenta como copia de "gracias".
- **Estilo.** Puntaje de 0 a 1 según si usa tu vocabulario, tu largo típico, tus emojis y qué tan
  cerca queda de tu respuesta real más parecida sin copiarla. Por debajo de `MIN_STYLE_SCORE` (0.5) se descarta.

Si las dos opciones fallan, ese comentario se reintenta la noche siguiente, como máximo dos veces en
total. El reporte muestra para cada respuesta su **estilo** y su **parecido** con tu respuesta real más
cercana, así puedes ajustar los límites.

### Uso mínimo de la API de OpenAI

- El entrenamiento y los filtros no gastan tokens: son cálculos locales.
- Una llamada cada 10 comentarios, con `reasoning: none` y `service_tier: flex` (mitad de precio;
  si flex falla, reintenta con el tier normal).
- Las instrucciones (tu estilo + ejemplos) son idénticas en cada llamada, así que OpenAI las cachea
  y las cobra a ~10%.
- Las 2 opciones van en la misma llamada: más texto de salida, pero ninguna llamada extra. Con
  `VARIANTS_PER_COMMENT=1` gasta aún menos.
- Nunca vuelve a enviar un comentario que el modelo ya decidió no responder.
- Con ~35 respuestas al día son ~4 llamadas: **centavos de dólar al mes**. Cada reporte muestra los tokens usados.

## Qué permite YouTube y qué no

Resumen de las reglas que aplican a este agente, y qué hace el agente con cada una.

| Regla de YouTube / Google | Qué hace el agente |
|---|---|
| **Consentimiento y control final.** Las políticas de la API piden que el usuario haya dado su consentimiento *previo, específico y expreso* antes de automatizar comentarios, y que tenga *el control final* de lo que se publica. | **Nada se publica sin tu aprobación.** Cada noche apruebas (y puedes editar) respuesta por respuesta en un issue; solo se publica lo que marcas al cerrarlo. |
| **Spam en comentarios.** Prohibido dejar muchos comentarios idénticos, no dirigidos o repetitivos. YouTube avisa que publicar mucho en poco tiempo, repetir el mismo comentario, poner enlaces o abusar de los emojis puede marcarse como spam. | Cada respuesta es única y responde a ese comentario. Nunca pone enlaces ni hashtags, y descarta respuestas con exceso de emojis (más de 3, salvo que tú uses más). Hace pausas al azar, tiene un tope diario moderado (35) y **se detiene si YouTube oculta una respuesta**. |
| **Interacción falsa e incentivos.** Prohibido inflar métricas con sistemas automáticos y ofrecer recompensas por comentar o suscribirse. Invitar a suscribirse sí está permitido. | El modelo tiene prohibido pedir likes, suscripciones o compras y ofrecer algo a cambio. Solo responde comentarios reales de tu propio canal. |
| **Datos de la API: máximo 30 días.** Lo guardado (textos de comentarios, IDs) se debe borrar o refrescar a los 30 días. Si revocas el acceso o el token ya no se puede renovar, hay que borrar los datos. | Reentrena cada 25 días, lo que refresca los datos y hace desaparecer lo que se borró en YouTube. Borra los registros de más de 30 días. Los reportes duran 7 días. Si el token deja de servir, **borra los datos** (GitHub elimina solas las copias viejas de la caché a los 7 días sin uso). |
| **Cuota diaria.** 10.000 unidades gratis al día; cada respuesta publicada cuesta 50. | Se frena solo en 9.000 y se detiene ante errores de cuota o de permisos. |
| **Lista de miembros (`members.list`).** Solo está disponible para creadores a quienes Google les dio acceso. | Si YouTube no la entrega, usa tu lista manual del secreto `MEMBER_CHANNEL_IDS`. |
| **Suscriptores.** Las suscripciones son **privadas por defecto**, así que la API solo ve a quienes las tienen públicas. | Los suscriptores visibles van primero; los cupos restantes se llenan con otros comentaristas (`INCLUDE_NON_SUBSCRIBERS`). |
| **IA en comentarios.** La etiqueta obligatoria de contenido alterado o sintético aplica a videos realistas. No encontré ninguna regla que obligue a avisar en comentarios. YouTube mismo ofrece respuestas sugeridas por IA "con tu propio tono". | Nada especial. |
| **Uso automatizado.** Los Términos de YouTube prohíben entrar al servicio con robots o scrapers. La vía autorizada para automatizar es la API oficial. | Solo usa la YouTube Data API oficial, con tu propia autorización OAuth. |

Fuentes: [Políticas para desarrolladores de la API](https://developers.google.com/youtube/terms/developer-policies),
[Política de spam](https://support.google.com/youtube/answer/2801973),
[Interacción falsa](https://support.google.com/youtube/answer/3399767),
[Comentarios detectados como spam](https://support.google.com/youtube/answer/13209064),
[Costo de cuota](https://developers.google.com/youtube/v3/determine_quota_cost),
[members.list](https://developers.google.com/youtube/v3/docs/members/list),
[Privacidad de suscripciones](https://support.google.com/youtube/answer/7280190),
[Contenido alterado o sintético](https://support.google.com/youtube/answer/14328491),
[Términos de YouTube](https://www.youtube.com/t/terms),
[OAuth 2.0 de Google](https://developers.google.com/identity/protocols/oauth2).
Varias de estas páginas no se pudieron abrir directamente al investigarlas, así que su contenido se
tomó de extractos de búsqueda. Si alguna regla es crítica para ti, revísala en la fuente.

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
   `YT_CLIENT_ID`, `YT_CLIENT_SECRET` y `YT_REFRESH_TOKEN`. Si Google te dio acceso a la lista de
   miembros por API, agrega `--members`.

### 2. Clave de OpenAI

Crea una API key en <https://platform.openai.com/api-keys>.

### 3. Tu lista de miembros

Como `members.list` casi nunca está disponible, dale al agente la lista tú mismo: el **ID de canal**
(`UC...`) o el **@handle** de cada miembro, separados por comas o saltos de línea. El ID aparece en
el canal de cada persona, en *Acerca de → Compartir canal → Copiar ID del canal*. Actualízala cuando
entren o salgan miembros.

### 4. Secretos en GitHub

En el repo: **Settings → Secrets and variables → Actions → New repository secret**:

| Secreto | Valor |
|---|---|
| `YT_CLIENT_ID` | del paso 1 |
| `YT_CLIENT_SECRET` | del paso 1 |
| `YT_REFRESH_TOKEN` | del paso 1 |
| `OPENAI_API_KEY` | del paso 2 |
| `MEMBER_CHANNEL_IDS` | del paso 3 (va como secreto para que la lista no quede pública) |

### 5. Unir la rama a `master`

GitHub solo ejecuta los trabajos programados desde la rama principal.

### 6. Primera prueba

1. **Actions → Agente de comentarios de YouTube → Run workflow** con `command = train`. Revisa en
   el log cuántas respuestas tuyas encontró.
2. Otra vez con `command = draft`. En unos minutos aparece el issue con los borradores.
3. Revísalos. Si no te convencen, ciérralo como *no planeado* y ajusta las variables de abajo. Si
   te gustan, marca las que quieras y ciérralo: se publican.

Desde ahí se repite solo cada noche. Para apagarlo: **Actions → Agente de comentarios de YouTube →
⋯ → Disable workflow**.

## Ajustes opcionales (variables del repositorio)

| Variable | Por defecto | Qué hace |
|---|---|---|
| `OPENAI_MODEL` | `gpt-6-luna` | Modelo de OpenAI |
| `VARIANTS_PER_COMMENT` | `2` | Opciones que el modelo propone por comentario |
| `MAX_SIMILARITY` | `0.75` | Parecido máximo con cualquier respuesta existente (1 = copia exacta) |
| `MIN_STYLE_SCORE` | `0.5` | Puntaje de estilo mínimo para publicar |
| `MIN_DELAY_SECONDS` / `MAX_DELAY_SECONDS` | `30` / `120` | Pausa al azar entre respuestas |
| `MAX_RUN_MINUTES` | `60` | Tiempo máximo publicando cada noche (si lo subes de 80, sube también `timeout-minutes` en el workflow) |
| `LOOKBACK_DAYS` | `3` | Cuántos días hacia atrás buscar comentarios sin responder (y días antes de que caduque un issue sin aprobar) |
| `MAX_MEMBER_REPLIES` | `30` | Tope por noche de respuestas a miembros |
| `MAX_SUBSCRIBER_REPLIES` | `5` | Respuestas por noche a no miembros |
| `MAX_REPLIES_PER_AUTHOR` | `2` | Máximo de respuestas por persona y noche |
| `INCLUDE_NON_SUBSCRIBERS` | `true` | Completar los cupos de no miembros con comentaristas cuya suscripción no se ve |
| `TRAIN_BEFORE` | — | Fecha ISO; solo aprende de respuestas anteriores a ella |

## Limitaciones que debes conocer

- **Aprende de tus respuestas anteriores a la activación.** Una vez que el agente empieza a
  publicar, deja de aprender de respuestas nuevas del canal (incluidas las que escribas tú a mano),
  para no aprender nunca de sí mismo.
- **Minutos de GitHub Actions:** en un repo público son gratis. En uno privado con el plan Free hay
  2.000 minutos al mes; con los valores por defecto el agente usa como mucho unos 60 minutos por noche.
- **Horario:** GitHub puede retrasar los trabajos programados en horas de mucha carga. En repositorios
  públicos desactiva la programación tras 60 días sin actividad en el repo; en ese caso se reactiva
  desde la pestaña Actions.
- **Un issue a la vez:** si cierras dos issues de borradores seguidos mientras el agente publica,
  GitHub puede descartar uno de los dos en la fila. Cierra el siguiente cuando termine el anterior.
- **Privacidad:** el texto de los comentarios se envía a OpenAI para generar las respuestas (con
  `store=false`). En un repo público los logs de Actions son públicos: el agente solo escribe totales
  en los logs, y el detalle va en el artefacto del reporte, que se borra a los 7 días. Pero los
  **issues de borradores sí se ven** mientras esperan tu aprobación (comentarios, respuestas y quién
  es miembro). Por eso conviene que el repositorio sea **privado**; el modo aprobación funciona igual.

## Desarrollo

```bash
pip install -r requirements.txt pytest
python -m pytest
```
