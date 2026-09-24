"""Ejecuta esto UNA vez en tu computador para autorizar al agente en tu canal.

    pip install google-auth-oauthlib
    python get_refresh_token.py client_secret.json            # lo normal
    python get_refresh_token.py client_secret.json --members  # solo si Google te dio acceso a members.list

Se abre el navegador: entra con la cuenta de Google del canal (si tienes varios canales,
elige el correcto). Al final imprime los tres valores que van como secretos en GitHub.
"""

import argparse

from google_auth_oauthlib.flow import InstalledAppFlow

# Leer comentarios y publicar respuestas.
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
# Leer la lista de miembros. YouTube solo la habilita a creadores a quienes Google les dio acceso.
MEMBERS_SCOPE = "https://www.googleapis.com/auth/youtube.channel-memberships.creator"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("client_secret", nargs="?", default="client_secret.json")
    parser.add_argument("--members", action="store_true", help="pedir también acceso a la lista de miembros")
    args = parser.parse_args()

    scopes = SCOPES + ([MEMBERS_SCOPE] if args.members else [])
    flow = InstalledAppFlow.from_client_secrets_file(args.client_secret, scopes)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    print("\nGuarda estos valores como secretos del repositorio en GitHub:\n")
    print(f"YT_CLIENT_ID={creds.client_id}")
    print(f"YT_CLIENT_SECRET={creds.client_secret}")
    print(f"YT_REFRESH_TOKEN={creds.refresh_token}")


if __name__ == "__main__":
    main()
