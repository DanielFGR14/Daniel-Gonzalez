"""Ejecuta esto UNA vez en tu computador para autorizar al agente en tu canal.

    pip install google-auth-oauthlib
    python get_refresh_token.py client_secret.json

Se abre el navegador: entra con la cuenta de Google del canal (si tienes varios canales,
elige el correcto). Al final imprime los tres valores que van como secretos en GitHub.
"""

import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    # Leer comentarios y publicar respuestas.
    "https://www.googleapis.com/auth/youtube.force-ssl",
    # Leer la lista de miembros del canal.
    "https://www.googleapis.com/auth/youtube.channel-memberships.creator",
]


def main() -> None:
    secrets_file = sys.argv[1] if len(sys.argv) > 1 else "client_secret.json"
    flow = InstalledAppFlow.from_client_secrets_file(secrets_file, SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    print("\nGuarda estos valores como secretos del repositorio en GitHub:\n")
    print(f"YT_CLIENT_ID={creds.client_id}")
    print(f"YT_CLIENT_SECRET={creds.client_secret}")
    print(f"YT_REFRESH_TOKEN={creds.refresh_token}")


if __name__ == "__main__":
    main()
