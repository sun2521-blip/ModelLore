"""Serve the built console (frontend/dist) from the API's own origin.

Serving the page and the API from one origin means the browser never makes a
cross-origin call, so CORS can stay closed. Only files inside the build folder
are served, and every response carries a strict content security policy.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = PROJECT_ROOT / "frontend" / "dist"

# The console loads only its own scripts, styles, fonts and images, and talks
# only to this API. Inline styles are allowed because React sets a few style
# attributes (bar heights, the playhead position); inline scripts are not.
CONSOLE_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

NOT_BUILT = "The console has not been built. Run `npm run build` in frontend/."

router = APIRouter(include_in_schema=False)


def ui_file(relative_path: str) -> Path | None:
    """Resolve a path inside the build folder, or None if it is outside or missing."""
    root = UI_DIR.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate


@router.get("/")
def console_page() -> FileResponse:
    page = ui_file("index.html")
    if page is None:
        raise HTTPException(status_code=404, detail=NOT_BUILT)
    return FileResponse(
        page,
        media_type="text/html",
        headers={**CONSOLE_SECURITY_HEADERS, "Cache-Control": "no-cache"},
    )


@router.get("/assets/{asset_path:path}")
def console_asset(asset_path: str) -> FileResponse:
    asset = ui_file(f"assets/{asset_path}")
    if asset is None:
        raise HTTPException(status_code=404, detail="Not Found")
    # Vite fingerprints asset names, so a given URL never changes content.
    return FileResponse(
        asset,
        headers={
            **CONSOLE_SECURITY_HEADERS,
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@router.get("/favicon.svg")
def console_favicon() -> FileResponse:
    icon = ui_file("favicon.svg")
    if icon is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(icon, media_type="image/svg+xml", headers=CONSOLE_SECURITY_HEADERS)
