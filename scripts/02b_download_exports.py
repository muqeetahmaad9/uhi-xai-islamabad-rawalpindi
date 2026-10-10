"""Step 2b — download finished GeoTIFF exports from Google Drive (read-only).

Uses the Earth Engine OAuth credentials (they include the Drive scope) and
saves files from the Drive folder config.DRIVE_FOLDER into data/gee_exports/.

Usage:  python scripts/02b_download_exports.py [name ...]   # default: all features_*.tif
"""
import sys
from pathlib import Path

import ee
import google.auth.transport.requests as gtr
import google.oauth2.credentials

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C  # noqa: E402

API = "https://www.googleapis.com/drive/v3/files"


def session():
    creds = google.oauth2.credentials.Credentials(None, **ee.oauth.get_credentials_arguments())
    return gtr.AuthorizedSession(creds)


def list_files(s):
    q = f"name='{C.DRIVE_FOLDER}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    folders = s.get(API, params={"q": q, "fields": "files(id)"}).json()["files"]
    if not folders:
        raise SystemExit(f"Drive folder {C.DRIVE_FOLDER!r} not found")
    files = []
    for f in folders:
        q = f"'{f['id']}' in parents and trashed=false"
        files += s.get(API, params={"q": q, "fields": "files(id,name,size,modifiedTime)",
                                    "pageSize": 1000}).json()["files"]
    return files


def main():
    wanted = {f"{n}.tif" for n in sys.argv[1:]}
    s = session()
    files = [f for f in list_files(s) if f["name"].endswith(".tif")
             and (f["name"] in wanted if wanted else f["name"].startswith("features_"))]
    if not files:
        raise SystemExit("No matching files in Drive yet.")
    # If a name occurs more than once (re-exports), keep the newest.
    latest = {}
    for f in sorted(files, key=lambda f: f["modifiedTime"]):
        latest[f["name"]] = f
    C.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, f in sorted(latest.items()):
        out = C.EXPORT_DIR / name
        with s.get(f"{API}/{f['id']}", params={"alt": "media"}, stream=True) as r:
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        print(f"{name}: {out.stat().st_size / 1e6:.1f} MB (Drive modified {f['modifiedTime']})")


if __name__ == "__main__":
    main()
